///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/ddp.hpp"

#include <iostream>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl
{

SolverDDP::SolverDDP(ShootingProblem& problem)
    : SolverAbstract(problem),
      reg_incfactor_(10.),
      reg_decfactor_(10.),
      reg_min_(1e-9),
      reg_max_(1e9),
      cost_try_(0.),
      th_grad_(1e-12),
      th_gaptol_(1e-16),
      th_stepdec_(0.5),
      th_stepinc_(0.01),
      was_feasible_(false)
{
    allocateData();

    const std::size_t n_alphas = 10;
    alphas_.resize(n_alphas);
    for (std::size_t n = 0; n < n_alphas; ++n)
    {
        alphas_[n] = 1. / pow(2., static_cast<double>(n));
    }
    if (th_stepinc_ < alphas_[n_alphas - 1])
    {
        th_stepinc_ = alphas_[n_alphas - 1];
        std::cerr << "Warning: th_stepinc has higher value than lowest alpha value, set to "
                  << std::to_string(alphas_[n_alphas - 1]) << std::endl;
    }
}

SolverDDP::~SolverDDP() {}

bool SolverDDP::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                      const std::size_t maxiter, const bool is_feasible, const double reginit)
{
    xs_try_[0] = problem_.get_x0();  // it is needed in case that init_xs[0] is infeasible
    setCandidate(init_xs, init_us, is_feasible);

    if (std::isnan(reginit))
    {
        xreg_ = reg_min_;
        ureg_ = reg_min_;
    }
    else
    {
        xreg_ = reginit;
        ureg_ = reginit;
    }
    was_feasible_ = false;

    bool recalcDiff = true;
    for (iter_ = 0; iter_ < maxiter; ++iter_)
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
                increaseRegularization();

                if (xreg_ == reg_max_)
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
        auto d = expected_improvement();

        // We need to recalculate the derivatives when the step length passes
        recalcDiff = false;
        for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it)
        {
            steplength_ = *it;

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
            decreaseRegularization();
        }
        if (steplength_ <= th_stepinc_)
        {
            increaseRegularization();

            if (xreg_ == reg_max_)
            {
                return false;
            }
        }

        const std::size_t n_callbacks = callbacks_.size();
        for (std::size_t c = 0; c < n_callbacks; ++c)
        {
            CallbackAbstract& callback = *callbacks_[c];
            callback(*this);
        }

        if (was_feasible_ && stoppingCriteria() < th_stop_)
        {
            return true;
        }
    }
    return false;
}

double SolverDDP::stoppingCriteria()
{
    double stop = 0.;

    const auto& models = problem_.get_runningModels();

    for (std::size_t t = 0; t < problem_.get_T(); ++t)
    {
        const std::size_t nu = models[t]->get_nu();
        if (nu != 0)
        {
            stop += Qu_[t].head(nu).squaredNorm();
        }
    }
    return stop;
}

Eigen::Vector2d SolverDDP::expected_improvement()
{
    Eigen::Vector2d d = Eigen::Vector2d::Zero();

    for (std::size_t t = 0; t < problem_.get_T(); ++t)
    {
        const std::size_t nu = problem_.get_runningModels()[t]->get_nu();

        if (nu != 0)
        {
            d[0] += Qu_[t].head(nu).dot(k_[t].head(nu));
            d[1] -= k_[t].head(nu).dot(Quuk_[t].head(nu));
        }
    }
    return d;
}

double SolverDDP::calcDiff()
{
    if (iter_ == 0) problem_.calc(xs_, us_);
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
    {  // closing the gaps
        for (std::vector<Eigen::VectorXd>::iterator it = fs_.begin(); it != fs_.end(); ++it)
        {
            it->setZero();
        }
    }
    return cost_;
}

void SolverDDP::backwardPass()
{
    Vxx_.back() = problem_.get_terminalData()->Lxx;
    Vx_.back() = problem_.get_terminalData()->Lx;

    if (!std::isnan(xreg_))
    {
        Vxx_.back().diagonal().array() += xreg_;
    }

    if (!is_feasible_)
    {
        Vx_.back().noalias() += Vxx_.back() * fs_.back();
    }

    const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_.get_runningModels();
    const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_.get_runningDatas();
    for (int t = static_cast<int>(problem_.get_T()) - 1; t >= 0; --t)
    {
        const boost::shared_ptr<ActionModelAbstract>& m = models[t];
        const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
        const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
        const Eigen::VectorXd& Vx_p = Vx_[t + 1];
        const std::size_t nu = m->get_nu();

        Qxx_[t] = d->Lxx;
        Qx_[t] = d->Lx;
        FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
        Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
        Qx_[t].noalias() += d->Fx.transpose() * Vx_p;

        if (nu != 0)
        {
            Qxu_[t].leftCols(nu) = d->Lxu + FxTVxx_p_ * d->Fu;
            Quu_[t].topLeftCorner(nu, nu) = d->Luu + FuTVxx_p_[t].topRows(nu) * d->Fu;
            Qu_[t].head(nu) = d->Lu + d->Fu.transpose() * Vx_p;
            FuTVxx_p_[t].topRows(nu).noalias() = d->Fu.transpose() * Vxx_p;

            if (!std::isnan(ureg_))
            {
                Quu_[t].diagonal().head(nu).array() += ureg_;
            }
        }

        computeGains(t);

        Vx_[t] = Qx_[t];
        Vxx_[t] = Qxx_[t];
        if (nu != 0)
        {
            Quuk_[t].head(nu).noalias() = Quu_[t].topLeftCorner(nu, nu) * k_[t].head(nu);
            Vx_[t].noalias() -= K_[t].topRows(nu).transpose() * Qu_[t].head(nu);
            Vxx_[t].noalias() -= Qxu_[t].leftCols(nu) * K_[t].topRows(nu);
        }
        Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
        Vxx_[t] = Vxx_tmp_;

        if (!std::isnan(xreg_))
        {
            Vxx_[t].diagonal().array() += xreg_;
        }

        // Compute and store the Vx gradient at end of the interval (rollout state)
        if (!is_feasible_)
        {
            Vx_[t].noalias() += Vxx_[t] * fs_[t];
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

void SolverDDP::computeGains(const std::size_t t)
{
    const std::size_t nu = problem_.get_runningModels()[t]->get_nu();
    if (nu > 0)
    {
        Eigen::LLT<Eigen::MatrixXd> Quu_llt(Quu_[t].topLeftCorner(nu, nu));

        if (Quu_llt.info() != Eigen::Success)
        {
            throw_pretty("backward_error");
        }

        K_[t].topRows(nu) = Qxu_[t].leftCols(nu).transpose();
        k_[t].head(nu) = Qu_[t].head(nu);

        Quu_llt.solveInPlace(K_[t].topRows(nu));
        Quu_llt.solveInPlace(k_[t].head(nu));
    }
}

void SolverDDP::increaseRegularization()
{
    xreg_ *= reg_incfactor_;
    if (xreg_ > reg_max_)
    {
        xreg_ = reg_max_;
    }
    ureg_ = xreg_;
}

void SolverDDP::decreaseRegularization()
{
    xreg_ /= reg_decfactor_;
    if (xreg_ < reg_min_)
    {
        xreg_ = reg_min_;
    }
    ureg_ = xreg_;
}

void SolverDDP::allocateData()
{
    const std::size_t T = problem_.get_T();
    Vxx_.resize(T + 1);
    Vx_.resize(T + 1);
    Qxx_.resize(T);
    Qxu_.resize(T);
    Quu_.resize(T);
    Qx_.resize(T);
    Qu_.resize(T);
    K_.resize(T);
    k_.resize(T);
    fs_.resize(T + 1);

    xs_try_.resize(T + 1);
    us_try_.resize(T);
    dx_.resize(T);

    FuTVxx_p_.resize(T);
    Quuk_.resize(T);

    const std::size_t ndx = problem_.get_ndx();
    const std::size_t nu = problem_.get_nu_max();

    for (std::size_t t = 0; t < T; ++t)
    {
        const boost::shared_ptr<ActionModelAbstract>& model = problem_.get_runningModels()[t];
        Vxx_[t] = Eigen::MatrixXd::Zero(ndx, ndx);
        Vx_[t] = Eigen::VectorXd::Zero(ndx);
        Qxx_[t] = Eigen::MatrixXd::Zero(ndx, ndx);
        Qxu_[t] = Eigen::MatrixXd::Zero(ndx, nu);
        Quu_[t] = Eigen::MatrixXd::Zero(nu, nu);
        Qx_[t] = Eigen::VectorXd::Zero(ndx);
        Qu_[t] = Eigen::VectorXd::Zero(nu);
        K_[t] = MatrixXdRowMajor::Zero(nu, ndx);
        k_[t] = Eigen::VectorXd::Zero(nu);
        fs_[t] = Eigen::VectorXd::Zero(ndx);

        if (t == 0)
        {
            xs_try_[t] = problem_.get_x0();
        }
        else
        {
            xs_try_[t] = model->get_state()->zero();
        }
        us_try_[t] = Eigen::VectorXd::Zero(nu);
        dx_[t] = Eigen::VectorXd::Zero(ndx);

        FuTVxx_p_[t] = MatrixXdRowMajor::Zero(nu, ndx);
        Quuk_[t] = Eigen::VectorXd(nu);
    }
    Vxx_.back() = Eigen::MatrixXd::Zero(ndx, ndx);
    Vxx_tmp_ = Eigen::MatrixXd::Zero(ndx, ndx);
    Vx_.back() = Eigen::VectorXd::Zero(ndx);
    xs_try_.back() = problem_.get_terminalModel()->get_state()->zero();
    fs_.back() = Eigen::VectorXd::Zero(ndx);

    FxTVxx_p_ = MatrixXdRowMajor::Zero(ndx, ndx);
    fTVxx_p_ = Eigen::VectorXd::Zero(ndx);
}

}  // namespace crocoddyl
