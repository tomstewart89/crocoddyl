///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/timer.hpp"

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <example-robot-data/path.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl
{
namespace benchmark
{

template <typename Scalar>
void build_arm_action_models(boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& runningModel,
                             boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& terminalModel)
{
    typedef typename crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar>
        DifferentialActionModelFreeFwdDynamics;
    typedef typename crocoddyl::IntegratedActionModelEulerTpl<Scalar> IntegratedActionModelEuler;
    typedef typename crocoddyl::ActuationModelFullTpl<Scalar> ActuationModelFull;
    typedef typename crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
    typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModelAbstract;
    typedef typename crocoddyl::CostModelResidualTpl<Scalar> CostModelResidual;
    typedef typename crocoddyl::ResidualModelStateTpl<Scalar> ResidualModelState;
    typedef typename crocoddyl::ResidualModelFramePlacementTpl<Scalar> ResidualModelFramePlacement;
    typedef typename crocoddyl::ResidualModelControlTpl<Scalar> ResidualModelControl;
    typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
    typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
    typedef typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s Matrix3s;

    // because urdf is not supported with all scalar types.
    pinocchio::ModelTpl<double> modeld;
    pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf", modeld);
    pinocchio::srdf::loadReferenceConfigurations(modeld, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
                                                 false);

    pinocchio::ModelTpl<Scalar> model_full(modeld.cast<Scalar>()), model;
    std::vector<pinocchio::JointIndex> locked_joints;
    locked_joints.push_back(5);
    locked_joints.push_back(6);
    locked_joints.push_back(7);
    pinocchio::buildReducedModel(model_full, locked_joints, VectorXs::Zero(model_full.nq), model);

    boost::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
        boost::make_shared<crocoddyl::StateMultibodyTpl<Scalar> >(
            boost::make_shared<pinocchio::ModelTpl<Scalar> >(model));

    boost::shared_ptr<CostModelAbstract> goalTrackingCost = boost::make_shared<CostModelResidual>(
        state, boost::make_shared<ResidualModelFramePlacement>(
                   state, model.getFrameId("gripper_left_joint"),
                   pinocchio::SE3Tpl<Scalar>(Matrix3s::Identity(), Vector3s(Scalar(0.), Scalar(0), Scalar(.3)))));
    boost::shared_ptr<CostModelAbstract> xRegCost =
        boost::make_shared<CostModelResidual>(state, boost::make_shared<ResidualModelState>(state));
    boost::shared_ptr<CostModelAbstract> uRegCost =
        boost::make_shared<CostModelResidual>(state, boost::make_shared<ResidualModelControl>(state));

    // Create a cost model per the running and terminal action model.
    boost::shared_ptr<CostModelSum> runningCostModel = boost::make_shared<CostModelSum>(state);
    boost::shared_ptr<CostModelSum> terminalCostModel = boost::make_shared<CostModelSum>(state);

    // Then let's added the running and terminal cost functions
    runningCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
    runningCostModel->addCost("xReg", xRegCost, Scalar(1e-4));
    runningCostModel->addCost("uReg", uRegCost, Scalar(1e-4));
    terminalCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));

    // We define an actuation model
    boost::shared_ptr<ActuationModelFull> actuation = boost::make_shared<ActuationModelFull>(state);

    // Next, we need to create an action model for running and terminal knots. The
    // forward dynamics (computed using ABA) are implemented
    // inside DifferentialActionModelFullyActuated.
    boost::shared_ptr<DifferentialActionModelFreeFwdDynamics> runningDAM =
        boost::make_shared<DifferentialActionModelFreeFwdDynamics>(state, actuation, runningCostModel);

    // VectorXs armature(state->get_nq());
    // armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.;
    // runningDAM->set_armature(armature);
    // terminalDAM->set_armature(armature);
    runningModel = boost::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(1e-3));
    terminalModel = boost::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(0.));
}

}  // namespace benchmark
}  // namespace crocoddyl

int main()
{
    unsigned int N = 100;  // number of nodes
    unsigned int MAXITER = 100;

    // Building the running and terminal models
    boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModel, terminalModel;
    crocoddyl::benchmark::build_arm_action_models(runningModel, terminalModel);

    // Get the initial state
    boost::shared_ptr<crocoddyl::StateMultibody> state =
        boost::static_pointer_cast<crocoddyl::StateMultibody>(runningModel->get_state());
    std::cout << "NQ: " << state->get_nq() << std::endl;
    std::cout << "Number of nodes: " << N << std::endl << std::endl;
    Eigen::VectorXd q0 = Eigen::VectorXd::Random(state->get_nq());
    Eigen::VectorXd x0(state->get_nx());
    x0 << q0, Eigen::VectorXd::Random(state->get_nv());

    // For this optimal control problem, we define 100 knots (or running action
    // models) plus a terminal knot
    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels(N, runningModel);
    boost::shared_ptr<crocoddyl::ShootingProblem> problem =
        boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels, terminalModel);
    std::vector<Eigen::VectorXd> xs(N + 1, x0);
    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(runningModel->get_nu()));
    for (unsigned int i = 0; i < N; ++i)
    {
        const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem->get_runningModels()[i];
        const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = problem->get_runningDatas()[i];
        model->quasiStatic(data, us[i], x0);
    }

    // Formulating the optimal control problem
    crocoddyl::SolverDDP ddp(*problem);
    crocoddyl::Timer timer;
    ddp.solve(xs, us, MAXITER, false, 0.1);

    std::cout << "cost: " << ddp.get_cost() << "\ntime: " << timer.get_duration() << "\n";
}