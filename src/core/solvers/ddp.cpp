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
        K_.push_back(MatrixXdRowMajor::Zero(nu, ndx));
        k_.push_back(Eigen::VectorXd::Zero(nu));
        fs_.push_back(Eigen::VectorXd::Zero(ndx));
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

    problem_.calc(xs_, us_);
    calcDiff();

    for (std::size_t iter = 0; iter < maxiter; ++iter)
    {
        while (!backwardPass())
        {
            reg_ *= reg_incfactor_;

            if (reg_ >= reg_max_)
            {
                return false;
            }
        }

        for (const auto& alpha : alphas_)
        {
            steplength_ = alpha;

            if (!forwardPass(steplength_))
            {
                continue;
            }

            // Calculate the actual change in cost
            dV_ = cost_ - cost_try_;

            // Calculate the expected change in cost
            dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);

            // If the step is in the descent direction of the cost
            if (dVexp_ >= 0)
            {
                if (d_[0] < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_)
                {
                    was_feasible_ = is_feasible_;
                    setCandidate(xs_try_, us_try_, true);
                    cost_ = cost_try_;

                    // We need to recalculate the derivatives when the step length passes
                    calcDiff();
                    break;
                }
            }
        }

        // If we were only able to take a short step then the quadratic approximation probably isn't very accurate so
        // let's increase the regularisation
        if (steplength_ > th_stepdec_)
        {
            reg_ = std::max(reg_ / reg_decfactor_, reg_min_);
        }
        // If we were able to take a large step, then we can decrease the regularisation
        else if (steplength_ <= th_stepinc_)
        {
            reg_ *= reg_incfactor_;

            if (reg_ >= reg_max_)
            {
                return false;
            }
        }

        std::cout << "it: " << iter << " " << cost_ << " reg: " << reg_ << "\n";

        if (was_feasible_ && stop_ < th_stop_)
        {
            return true;
        }
    }

    return false;
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
        // closing the gaps (because the trajectory is feasible now)
        for (auto& gap : fs_)
        {
            gap.setZero();  // ofcourse this gap must already have an inf-norm of lower than th_gaptol_ which is crazy
                            // small so we probably needn't even do this, unless this tiny error is being accumulated
                            // somewhere
        }
    }
    return cost_;
}

/**
 * @brief
 *
 * @param steplength initially 1 but will be set to progressively more conservative values... until something(?) happens
 */
bool SolverDDP::forwardPass(const double steplength)
{
    for (std::size_t t = 0; t < problem_.get_T(); ++t)
    {
        Eigen::VectorXd dx = Eigen::VectorXd::Zero(xs_[t].size());

        problem_.get_runningModels()[t]->get_state()->diff(xs_[t], xs_try_[t], dx);
        us_try_[t] = us_[t] - k_[t] * steplength - K_[t] * dx;

        problem_.get_runningModels()[t]->calc(problem_.get_runningDatas()[t], xs_try_[t], us_try_[t]);
        xs_try_[t + 1] = problem_.get_runningDatas()[t]->xnext;

        if (raiseIfNaN(xs_try_[t + 1].lpNorm<Eigen::Infinity>()))
        {
            return false;
        }
    }

    problem_.get_terminalModel()->calc(problem_.get_terminalData(), xs_try_.back());

    cost_try_ = std::accumulate(problem_.get_runningDatas().begin(), problem_.get_runningDatas().end(),
                                problem_.get_terminalData()->cost,
                                [](double sum, const auto& data) { return sum + data->cost; });

    if (raiseIfNaN(cost_try_))
    {
        return false;
    }

    return true;
}

bool SolverDDP::backwardPass()
{
    Vxx_.back() = problem_.get_terminalData()->Lxx;
    Vx_.back() = problem_.get_terminalData()->Lx;
    Vxx_.back().diagonal().array() += reg_;

    if (!is_feasible_)
    {
        Vx_.back().noalias() +=
            Vxx_.back() *
            fs_.back();  // the Jacobian of the Value function after the deflection produced by the gap fk+1
    }

    d_ = Eigen::Vector2d::Zero();
    stop_ = 0.0;

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
        Eigen::VectorXd Qu = d->Lu + d->Fu.transpose() * Vx_p;

        Quu.diagonal().array() += reg_;

        Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu);

        if (Quu_llt.info() != Eigen::Success)
        {
            std::cout << "not positive definite I guess";
            return false;
        }

        k_[t] = Qu;
        Quu_llt.solveInPlace(k_[t]);

        K_[t] = Qxu.transpose();
        Quu_llt.solveInPlace(K_[t]);

        Vx_[t] = Qx - K_[t].transpose() * Qu;
        Vxx_[t] = Qxx - Qxu * K_[t];

        stop_ += Qu.squaredNorm();
        d_[0] += Qu.dot(k_[t]);           // don't know what this is
        d_[1] -= k_[t].dot(Quu * k_[t]);  // this is the change in the value at time t

        // Ensure symmetry of Vxx
        Eigen::MatrixXd Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
        Vxx_[t] = Vxx_tmp_;
        Vxx_[t].diagonal().array() += reg_;

        // Compute and store the Vx gradient at end of the interval (rollout state)
        if (!is_feasible_)
        {
            Vx_[t] += Vxx_[t] * fs_[t];  // if the trajectory is feasible or not, we're always free to multiply this by
                                         // a crazy small number (for a slight performance hit maybe)
        }

        if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>()))
        {
            return false;
        }
        if (raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>()))
        {
            return false;
        }
    }

    return true;
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

}  // namespace crocoddyl
