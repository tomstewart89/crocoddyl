///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/ddp.hpp"

#include <iostream>
#include <numeric>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl
{

bool raiseIfNaN(const double value)
{
    if (std::isnan(value) || std::isinf(value) || value >= 1e30)
    {
        return true;
    }
    else
    {
        return false;
    }
}

SolverDDP::SolverDDP(ShootingProblem& problem) : problem_(problem)
{
    const std::size_t T = problem_.get_T();
    const std::size_t ndx = problem_.get_ndx();

    for (std::size_t t = 0; t < T; ++t)
    {
        const auto& running_model = problem_.get_runningModels()[t];
        const std::size_t nu = running_model->get_nu();

        xs_.push_back(running_model->get_state()->zero());
        xs_try_.push_back(running_model->get_state()->zero());

        us_.push_back(Eigen::VectorXd::Zero(nu));
        us_try_.push_back(Eigen::VectorXd::Zero(nu));

        Vxx_.push_back(Eigen::MatrixXd::Zero(ndx, ndx));
        Vx_.push_back(Eigen::VectorXd::Zero(ndx));
        Qu_.push_back(Eigen::VectorXd::Zero(nu));
        K_.push_back(MatrixXdRowMajor::Zero(nu, ndx));
        k_.push_back(Eigen::VectorXd::Zero(nu));
        fs_.push_back(Eigen::VectorXd::Zero(ndx));

        dx_.push_back(Eigen::VectorXd::Zero(ndx));

        Quuk_.push_back(Eigen::VectorXd(nu));
    }

    xs_try_[0] = problem_.get_x0();
    xs_.push_back(problem_.get_terminalModel()->get_state()->zero());
    xs_try_.push_back(problem_.get_terminalModel()->get_state()->zero());
    Vxx_.push_back(Eigen::MatrixXd::Zero(ndx, ndx));
    Vx_.push_back(Eigen::VectorXd::Zero(ndx));
    fs_.push_back(Eigen::VectorXd::Zero(ndx));

    std::generate_n(std::back_inserter(alphas_), 10,
                    [n = 0]() mutable { return 1.0 / pow(2., static_cast<double>(n++)); });

    if (th_stepinc_ < alphas_.back())
    {
        th_stepinc_ = alphas_.back();
        std::cerr << "Warning: th_stepinc has higher value than lowest alpha value, set to "
                  << std::to_string(alphas_.back()) << std::endl;
    }
}

bool SolverDDP::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                      const std::size_t maxiter, const bool is_feasible, const double reginit)
{
    xs_try_[0] = problem_.get_x0();  // it is needed in case that init_xs[0] is infeasible
    setCandidate(init_xs, init_us, is_feasible);

    reg_ = reginit;

    was_feasible_ = false;

    bool recalcDiff = true;

    problem_.calc(xs_, us_);

    for (std::size_t iter = 0; iter < maxiter; ++iter)
    {
        while (true)
        {
            try
            {
                if (recalcDiff)
                {
                    calcDiff();
                }

                backwardPass();
            }
            catch (std::exception& e)
            {
                recalcDiff = false;
                reg_ = std::min(reg_ * reg_incfactor_, reg_max_);

                if (reg_ == reg_max_)
                {
                    return false;
                }
                else
                {
                    continue;
                }
            }
            break;
        }
        auto d = expectedImprovement();

        // We need to recalculate the derivatives when the step length passes
        recalcDiff = false;
        for (const auto& alpha : alphas_)
        {
            steplength_ = alpha;

            try
            {
                forwardPass(steplength_);
                dV_ = cost_ - cost_try_;
            }
            catch (std::exception& e)
            {
                continue;
            }
            dVexp_ = steplength_ * (d[0] + 0.5 * steplength_ * d[1]);

            if (dVexp_ >= 0)  // descend direction
            {
                if (d[0] < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_)
                {
                    was_feasible_ = is_feasible_;
                    setCandidate(xs_try_, us_try_, true);
                    cost_ = cost_try_;
                    recalcDiff = true;
                    break;
                }
            }
        }

        if (steplength_ > th_stepdec_)
        {
            reg_ = std::max(reg_ / reg_decfactor_, reg_min_);  // decrease regularisation
        }
        if (steplength_ <= th_stepinc_)
        {
            reg_ = std::min(reg_ * reg_incfactor_, reg_max_);

            if (reg_ == reg_max_)
            {
                return false;
            }
        }

        std::cout << "it: " << iter << " " << cost_ << " reg: " << reg_ << "\n";

        if (was_feasible_ && stoppingCriteria() < th_stop_)
        {
            return true;
        }
    }

    return false;
}

void SolverDDP::setCandidate(const std::vector<Eigen::VectorXd>& xs_warm, const std::vector<Eigen::VectorXd>& us_warm,
                             bool is_feasible)
{
    const std::size_t T = problem_.get_T();

    if (xs_warm.size() == 0)
    {
        for (std::size_t t = 0; t < T; ++t)
        {
            xs_[t] = problem_.get_runningModels()[t]->get_state()->zero();
        }
        xs_.back() = problem_.get_terminalModel()->get_state()->zero();
    }
    else
    {
        if (xs_warm.size() != T + 1)
        {
            throw_pretty("Warm start state has wrong dimension, got " << xs_warm.size() << " expecting " << (T + 1));
        }
        for (std::size_t t = 0; t < T; ++t)
        {
            const std::size_t nx = problem_.get_runningModels()[t]->get_state()->get_nx();
            if (static_cast<std::size_t>(xs_warm[t].size()) != nx)
            {
                throw_pretty("Invalid argument: "
                             << "xs_init[" + std::to_string(t) + "] has wrong dimension (it should be " +
                                    std::to_string(nx) + ")");
            }
        }
        const std::size_t nx = problem_.get_terminalModel()->get_state()->get_nx();
        if (static_cast<std::size_t>(xs_warm[T].size()) != nx)
        {
            throw_pretty("Invalid argument: "
                         << "xs_init[" + std::to_string(T) + "] has wrong dimension (it should be " +
                                std::to_string(nx) + ")");
        }
        std::copy(xs_warm.begin(), xs_warm.end(), xs_.begin());
    }

    if (us_warm.size() == 0)
    {
        for (std::size_t t = 0; t < T; ++t)
        {
            us_[t] = Eigen::VectorXd::Zero(problem_.get_nu_max());
        }
    }
    else
    {
        if (us_warm.size() != T)
        {
            throw_pretty("Warm start control has wrong dimension, got " << us_warm.size() << " expecting " << T);
        }
        const std::size_t nu = problem_.get_nu_max();
        for (std::size_t t = 0; t < T; ++t)
        {
            if (static_cast<std::size_t>(us_warm[t].size()) > nu)
            {
                throw_pretty("Invalid argument: "
                             << "us_init[" + std::to_string(t) + "] has wrong dimension (it should be lower than " +
                                    std::to_string(nu) + ")");
            }
        }
        std::copy(us_warm.begin(), us_warm.end(), us_.begin());
    }
    is_feasible_ = is_feasible;
}

double SolverDDP::stoppingCriteria()
{
    return std::accumulate(Qu_.begin(), Qu_.end(), 0.0,
                           [](const auto& sum, const auto& elem) { return sum + elem.squaredNorm(); });
}

Eigen::Vector2d SolverDDP::expectedImprovement()
{
    Eigen::Vector2d d = Eigen::Vector2d::Zero();

    for (std::size_t t = 0; t < problem_.get_T(); ++t)
    {
        d[0] += Qu_[t].dot(k_[t]);
        d[1] -= k_[t].dot(Quuk_[t]);  // this is the change in the value
    }
    return d;
}

double SolverDDP::calcDiff()
{
    cost_ = problem_.calcDiff(xs_, us_);

    if (!is_feasible_)
    {
        const auto& models = problem_.get_runningModels();
        const auto& datas = problem_.get_runningDatas();

        for (std::size_t t = 0; t < problem_.get_T(); ++t)
        {
            models[t]->get_state()->diff(xs_[t + 1], datas[t]->xnext, fs_[t + 1]);
        }

        bool could_be_feasible = fs_[0].lpNorm<Eigen::Infinity>() < th_gaptol_;
        const Eigen::VectorXd& x0 = problem_.get_x0();
        problem_.get_runningModels()[0]->get_state()->diff(xs_[0], x0, fs_[0]);

        if (could_be_feasible)
        {
            for (std::size_t t = 0; t < problem_.get_T(); ++t)
            {
                if (fs_[t + 1].lpNorm<Eigen::Infinity>() >= th_gaptol_)
                {
                    could_be_feasible = false;
                    break;
                }
            }
        }
        is_feasible_ = could_be_feasible;
    }
    else if (!was_feasible_)
    {
        // closing the gaps
        for (auto& gap : fs_)
        {
            gap.setZero();
        }
    }
    return cost_;
}

void SolverDDP::backwardPass()
{
    Vxx_.back() = problem_.get_terminalData()->Lxx;
    Vx_.back() = problem_.get_terminalData()->Lx;
    Vxx_.back().diagonal().array() += reg_;

    if (!is_feasible_)
    {
        Vx_.back().noalias() += Vxx_.back() * fs_.back();
    }

    for (int t = static_cast<int>(problem_.get_T()) - 1; t >= 0; --t)
    {
        const auto& m = problem_.get_runningModels()[t];
        const auto& d = problem_.get_runningDatas()[t];
        const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
        const Eigen::VectorXd& Vx_p = Vx_[t + 1];
        const std::size_t nu = m->get_nu();

        MatrixXdRowMajor FxTVxx_p = d->Fx.transpose() * Vxx_p;

        Eigen::MatrixXd Qxx = d->Lxx + FxTVxx_p * d->Fx;
        Eigen::MatrixXd Qx = d->Lx + d->Fx.transpose() * Vx_p;

        Eigen::MatrixXd Qxu = d->Lxu + FxTVxx_p * d->Fu;
        Eigen::MatrixXd Quu = d->Luu + d->Fu.transpose() * Vxx_p * d->Fu;
        Qu_[t].head(nu) = d->Lu + d->Fu.transpose() * Vx_p;

        Quu.diagonal().array() += reg_;

        Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu);

        if (Quu_llt.info() != Eigen::Success)
        {
            std::cout << "not positive definite I guess";
            throw_pretty("backward_error");
        }

        K_[t] = Qxu.transpose();
        k_[t] = Qu_[t];

        Quu_llt.solveInPlace(K_[t]);
        Quu_llt.solveInPlace(k_[t]);

        Quuk_[t] = Quu * k_[t];
        Vx_[t] = Qx - K_[t].transpose() * Qu_[t];
        Vxx_[t] = Qxx - Qxu * K_[t];

        // Ensure symmetry of Vxx
        Eigen::MatrixXd Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
        Vxx_[t] = Vxx_tmp_;
        Vxx_[t].diagonal().array() += reg_;

        // Compute and store the Vx gradient at end of the interval (rollout state)
        if (!is_feasible_)
        {
            Vx_[t] += Vxx_[t] * fs_[t];
        }

        if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>()))
        {
            throw_pretty("backward_error");
        }
        if (raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>()))
        {
            throw_pretty("backward_error");
        }
    }
}

void SolverDDP::forwardPass(const double steplength)
{
    if (steplength > 1. || steplength < 0.)
    {
        throw_pretty("Invalid argument: "
                     << "invalid step length, value is between 0. to 1.");
    }
    cost_try_ = 0.;

    for (std::size_t t = 0; t < problem_.get_T(); ++t)
    {
        const auto& model = problem_.get_runningModels()[t];
        const auto& data = problem_.get_runningDatas()[t];

        model->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);

        if (model->get_nu() != 0)
        {
            us_try_[t] = us_[t] - k_[t] * steplength - K_[t] * dx_[t];
            model->calc(data, xs_try_[t], us_try_[t]);
        }
        else
        {
            model->calc(data, xs_try_[t]);
        }

        xs_try_[t + 1] = data->xnext;
        cost_try_ += data->cost;

        if (raiseIfNaN(cost_try_) || raiseIfNaN(xs_try_[t + 1].lpNorm<Eigen::Infinity>()))
        {
            throw_pretty("forward_error");
        }
    }

    problem_.get_terminalModel()->calc(problem_.get_terminalData(), xs_try_.back());
    cost_try_ += problem_.get_terminalData()->cost;

    if (raiseIfNaN(cost_try_))
    {
        throw_pretty("forward_error");
    }
}

}  // namespace crocoddyl
