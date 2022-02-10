///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <stdexcept>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/timer.hpp"

namespace crocoddyl
{
struct ActionDataUnicycle : public ActionDataAbstractTpl<double>
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    template <template <typename Scalar> class Model>
    explicit ActionDataUnicycle(Model<double>* const model) : ActionDataAbstractTpl<double>(model)
    {
        Fx.diagonal().array() = 1.0;
    }
};

class ActionModelUnicycle : public ActionModelAbstractTpl<double>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef ActionDataAbstractTpl<double> ActionDataAbstract;
    typedef ActionModelAbstractTpl<double> Base;
    typedef MathBaseTpl<double> MathBase;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::Vector2s Vector2s;

    ActionModelUnicycle()
        : ActionModelAbstractTpl<double>(boost::make_shared<StateVectorTpl<double> >(3), 2, 5), dt_(double(0.1))
    {
        cost_weights_ << double(10.), double(1.);
    }

    virtual ~ActionModelUnicycle() {}

    virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                      const Eigen::Ref<const VectorXs>& u)
    {
        if (static_cast<std::size_t>(x.size()) != state_->get_nx())
        {
            throw_pretty("Invalid argument: "
                         << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
        }
        if (static_cast<std::size_t>(u.size()) != nu_)
        {
            throw_pretty("Invalid argument: "
                         << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
        }

        ActionDataUnicycle* d = static_cast<ActionDataUnicycle*>(data.get());
        const double c = cos(x[2]);
        const double s = sin(x[2]);
        d->xnext << x[0] + c * u[0] * dt_, x[1] + s * u[0] * dt_, x[2] + u[1] * dt_;
        d->r.template head<3>() = cost_weights_[0] * x;
        d->r.template tail<2>() = cost_weights_[1] * u;
        d->cost = double(0.5) * d->r.transpose() * d->r;
    }

    virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& u)
    {
        if (static_cast<std::size_t>(x.size()) != state_->get_nx())
        {
            throw_pretty("Invalid argument: "
                         << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
        }
        if (static_cast<std::size_t>(u.size()) != nu_)
        {
            throw_pretty("Invalid argument: "
                         << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
        }

        ActionDataUnicycle* d = static_cast<ActionDataUnicycle*>(data.get());

        // Cost derivatives
        const double w_x = cost_weights_[0] * cost_weights_[0];
        const double w_u = cost_weights_[1] * cost_weights_[1];
        d->Lx = x * w_x;
        d->Lu = u * w_u;
        d->Lxx.diagonal().setConstant(w_x);
        d->Luu.diagonal().setConstant(w_u);

        // Dynamic derivatives
        const double c = cos(x[2]);
        const double s = sin(x[2]);
        d->Fx(0, 2) = -s * u[0] * dt_;
        d->Fx(1, 2) = c * u[0] * dt_;
        d->Fu(0, 0) = c * dt_;
        d->Fu(1, 0) = s * dt_;
        d->Fu(2, 1) = dt_;
    }

    virtual boost::shared_ptr<ActionDataAbstract> createData()
    {
        return boost::allocate_shared<ActionDataUnicycle>(Eigen::aligned_allocator<ActionDataUnicycle>(), this);
    }

    virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data)
    {
        boost::shared_ptr<ActionDataUnicycle> d = boost::dynamic_pointer_cast<ActionDataUnicycle>(data);
        if (d != NULL)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void print(std::ostream& os) const { os << "ActionModelUnicycle {dt=" << dt_ << "}"; }

    const Vector2s& get_cost_weights() const { return cost_weights_; }

    void set_cost_weights(const typename MathBase::Vector2s& weights) { cost_weights_ = weights; }

    double get_dt() const { return dt_; }

    void set_dt(const double dt)
    {
        if (dt <= 0) throw_pretty("Invalid argument: dt should be strictly positive.");
        dt_ = dt;
    }

   private:
    Vector2s cost_weights_;
    double dt_;
};

}  // namespace crocoddyl

int main(int argc, char* argv[])
{
    unsigned int N = 200;  // number of nodes
    unsigned int T = 5e3;  // number of trials
    unsigned int MAXITER = 1;
    if (argc > 1)
    {
        T = atoi(argv[1]);
    }

    // Creating the action models and warm point for the unicycle system
    Eigen::VectorXd x0 = Eigen::Vector3d(1., 0., 0.);
    boost::shared_ptr<crocoddyl::ActionModelAbstract> model = boost::make_shared<crocoddyl::ActionModelUnicycle>();
    std::vector<Eigen::VectorXd> xs(N + 1, x0);
    std::vector<Eigen::VectorXd> us(N, Eigen::Vector2d::Zero());
    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels(N, model);

    // Formulating the optimal control problem
    boost::shared_ptr<crocoddyl::ShootingProblem> problem =
        boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels, model);

    crocoddyl::SolverDDP ddp(problem);

    // Solving the optimal control problem
    Eigen::ArrayXd duration(T);
    for (unsigned int i = 0; i < T; ++i)
    {
        crocoddyl::Timer timer;
        ddp.solve(xs, us, MAXITER);
        duration[i] = timer.get_duration();
    }

    double avrg_duration = duration.sum() / T;
    double min_duration = duration.minCoeff();
    double max_duration = duration.maxCoeff();
    std::cout << "  DDP.solve [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
              << std::endl;

    // Running calc
    for (unsigned int i = 0; i < T; ++i)
    {
        crocoddyl::Timer timer;
        problem->calc(xs, us);
        duration[i] = timer.get_duration();
    }

    avrg_duration = duration.sum() / T;
    min_duration = duration.minCoeff();
    max_duration = duration.maxCoeff();
    std::cout << "  ShootingProblem.calc [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
              << std::endl;

    // Running calcDiff
    for (unsigned int i = 0; i < T; ++i)
    {
        crocoddyl::Timer timer;
        problem->calcDiff(xs, us);
        duration[i] = timer.get_duration();
    }

    avrg_duration = duration.sum() / T;
    min_duration = duration.minCoeff();
    max_duration = duration.maxCoeff();
    std::cout << "  ShootingProblem.calcDiff [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration
              << ")" << std::endl;
}
