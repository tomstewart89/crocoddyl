///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <Eigen/Core>
#include <example-robot-data/path.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

using boost::make_shared;
using boost::shared_ptr;
using namespace crocoddyl;

std::tuple<shared_ptr<ActionModelAbstractTpl<double>>, shared_ptr<ActionModelAbstractTpl<double>>>
build_arm_action_models()
{
    typedef typename MathBaseTpl<double>::Vector3s Vector3s;
    typedef typename MathBaseTpl<double>::Matrix3s Matrix3s;

    // because urdf is not supported with all double types.
    pinocchio::ModelTpl<double> modeld;
    pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf", modeld);
    pinocchio::srdf::loadReferenceConfigurations(modeld, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
                                                 false);

    pinocchio::ModelTpl<double> model_full(modeld.cast<double>()), model;
    std::vector<pinocchio::JointIndex> locked_joints;
    locked_joints.push_back(5);
    locked_joints.push_back(6);
    locked_joints.push_back(7);
    pinocchio::buildReducedModel(model_full, locked_joints, Eigen::VectorXd::Zero(model_full.nq), model);

    shared_ptr<StateMultibodyTpl<double>> state =
        make_shared<StateMultibodyTpl<double>>(make_shared<pinocchio::ModelTpl<double>>(model));

    // We define an actuation model
    auto actuation = make_shared<ActuationModelFloatingBaseTpl<double>>(state);

    ContactModelMultiple contact_model(state, actuation->get_nu());

    contact_model.addContact("left_contact", make_shared<ContactModel6D>(state, model.getFrameId("left_sole_link"),
                                                                         pinocchio::SE3Tpl<double>::Identity()));

    contact_model.addContact("right_contact", make_shared<ContactModel6D>(state, model.getFrameId("right_sole_link"),
                                                                          pinocchio::SE3Tpl<double>::Identity()));

    auto goalTrackingCost = make_shared<CostModelResidualTpl<double>>(
        state, make_shared<ResidualModelFramePlacementTpl<double>>(
                   state, model.getFrameId("gripper_left_joint"),
                   pinocchio::SE3Tpl<double>(Matrix3s::Identity(), Vector3s(double(0), double(0), double(.4)))));
    auto xRegCost = make_shared<CostModelResidualTpl<double>>(state, make_shared<ResidualModelStateTpl<double>>(state));
    auto uRegCost =
        make_shared<CostModelResidualTpl<double>>(state, make_shared<ResidualModelControlTpl<double>>(state));

    // Create a cost model per the running and terminal action model.
    auto runningCostModel = make_shared<CostModelSumTpl<double>>(state);
    auto terminalCostModel = make_shared<CostModelSumTpl<double>>(state);

    // Then let's added the running and terminal cost functions
    runningCostModel->addCost("gripperPose", goalTrackingCost, double(1));
    runningCostModel->addCost("xReg", xRegCost, double(1e-4));
    runningCostModel->addCost("uReg", uRegCost, double(1e-4));
    terminalCostModel->addCost("gripperPose", goalTrackingCost, double(1));

    // Next, we need to create an action model for running and terminal knots. The
    // forward dynamics (computed using ABA) are implemented
    // inside DifferentialActionModelFullyActuated.
    auto runningDAM =
        make_shared<DifferentialActionModelFreeFwdDynamicsTpl<double>>(state, actuation, runningCostModel);

    auto runningModel = make_shared<IntegratedActionModelEulerTpl<double>>(runningDAM, double(1e-3));
    auto terminalModel = make_shared<IntegratedActionModelEulerTpl<double>>(runningDAM, double(0.));

    return {runningModel, terminalModel};
}

int main()
{
    unsigned int N = 100;  // number of nodes
    unsigned int T = 5e3;  // number of trials
    unsigned int MAXITER = 1;

    // Building the running and terminal models
    auto [runningModel, terminalModel] = build_arm_action_models();

    // Get the initial state
    shared_ptr<StateMultibody> state = boost::static_pointer_cast<StateMultibody>(runningModel->get_state());

    Eigen::VectorXd q0 = Eigen::VectorXd::Random(state->get_nq());
    Eigen::VectorXd x0(state->get_nx());

    x0 << q0, Eigen::VectorXd::Random(state->get_nv());

    // For this optimal control problem, we define 100 knots (or running action
    // models) plus a terminal knot
    std::vector<shared_ptr<ActionModelAbstract>> runningModels(N, runningModel);
    ShootingProblem problem(x0, runningModels, terminalModel);

    std::vector<Eigen::VectorXd> xs(N + 1, x0);
    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(runningModel->get_nu()));

    for (unsigned int i = 0; i < N; ++i)
    {
        const shared_ptr<ActionModelAbstract> &model = problem.get_runningModels()[i];
        const shared_ptr<ActionDataAbstract> &data = problem.get_runningDatas()[i];
        model->quasiStatic(data, us[i], x0);
    }

    // Formulating the optimal control problem
    SolverDDP ddp(problem);

    // Solving the optimal control problem
    ddp.solve(xs, us, MAXITER, false, 0.1);
}

// import sys

// import crocoddyl
// import numpy as np
// import example_robot_data
// import pinocchio

// # Load robot
// robot = example_robot_data.load("talos")

// # Create data structures
// rdata = robot.model.createData()
// state = crocoddyl.StateMultibody(robot.model)
// actuation = crocoddyl.ActuationModelFloatingBase(state)

// # Set integration time
// DT = 5e-2
// T = 60
// target = np.array([0.5, 0, 1.8])

// # Initialize reference state, target and reference CoM
// rightFoot = "right_sole_link"
// leftFoot = "left_sole_link"
// endEffector = "gripper_left_joint"
// endEffectorId = robot.model.getFrameId(endEffector)
// rightFootId = robot.model.getFrameId(rightFoot)
// leftFootId = robot.model.getFrameId(leftFoot)
// q0 = robot.model.referenceConfigurations["half_sitting"]
// x0 = np.concatenate([q0, np.zeros(robot.model.nv)])
// pinocchio.forwardKinematics(robot.model, rdata, q0)
// pinocchio.updateFramePlacements(robot.model, rdata)

// # Cost for self-collision
// maxfloat = sys.float_info.max
// xlb = np.concatenate(
//     [
//         -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
//         robot.model.lowerPositionLimit[7:],
//         -maxfloat * np.ones(state.nv),
//     ]
// )
// xub = np.concatenate(
//     [
//         maxfloat * np.ones(6),  # dimension of the SE(3) manifold
//         robot.model.upperPositionLimit[7:],
//         maxfloat * np.ones(state.nv),
//     ]
// )
// bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
// xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
// xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
// limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)

// # Cost for state and control
// xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
// xActivation = crocoddyl.ActivationModelWeightedQuad(
//     np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv) ** 2
// )
// uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
// xTActivation = crocoddyl.ActivationModelWeightedQuad(
//     np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv) ** 2
// )
// xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
// uRegCost = crocoddyl.CostModelResidual(state, uResidual)
// xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

// # Cost for target reaching
// framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
//     state, endEffectorId, pinocchio.SE3(np.eye(3), target), actuation.nu
// )
// framePlacementActivation = crocoddyl.ActivationModelWeightedQuad(
//     np.array([1] * 3 + [0.0001] * 3) ** 2
// )
// goalTrackingCost = crocoddyl.CostModelResidual(
//     state, framePlacementActivation, framePlacementResidual
// )

// # Create cost model per each action model
// runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
// terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

// # Then let's added the running and terminal cost functions
// runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
// runningCostModel.addCost("stateReg", xRegCost, 1e-3)
// runningCostModel.addCost("ctrlReg", uRegCost, 1e-4)
// runningCostModel.addCost("limitCost", limitCost, 1e3)

// terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
// terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
// terminalCostModel.addCost("limitCost", limitCost, 1e3)

// # Create the action model
// dmodelRunningLeft = crocoddyl.DifferentialActionModelContactFwdDynamics(
//     state, actuation, contactModelLeft, runningCostModel
// )
// dmodelTerminalLeft = crocoddyl.DifferentialActionModelContactFwdDynamics(
//     state, actuation, contactModelLeft, terminalCostModel
// )

// dmodelRunning = crocoddyl.DifferentialActionModelContactFwdDynamics(
//     state, actuation, contactModelRight, runningCostModel
// )
// dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
//     state, actuation, contactModelRight, terminalCostModel
// )
// runningModelLeft = crocoddyl.IntegratedActionModelEuler(dmodelRunningLeft, DT)
// runningModel = crocoddyl.IntegratedActionModelEuler(dmodelRunning, DT)
// terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

// # Problem definition
// x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
// problem = crocoddyl.ShootingProblem(x0, [runningModel] * T + [runningModelLeft] * T, terminalModel)

// # Creating the DDP solver for this OC problem, defining a logger
// solver = crocoddyl.SolverFDDP(problem)
// solver.setCallbacks(
//     [
//         crocoddyl.CallbackVerbose(),
//         crocoddyl.CallbackDisplay(
//             crocoddyl.GepettoDisplay(robot, 4, 4, frameNames=[rightFoot, leftFoot])
//         ),
//     ]
// )

// # Solving it with the FDDP algorithm
// xs = [x0] * (solver.problem.T + 1)
// us = solver.problem.quasiStatic([x0] * solver.problem.T)
// solver.solve(xs, us, 500, False, 0.1)

// # Visualizing the solution in gepetto-viewer
// display.displayFromSolver(solver)

// # Get final state and end effector position
// xT = solver.xs[-1]
// pinocchio.forwardKinematics(robot.model, rdata, xT[: state.nq])
// pinocchio.updateFramePlacements(robot.model, rdata)
// com = pinocchio.centerOfMass(robot.model, rdata, xT[: state.nq])
// finalPosEff = np.array(rdata.oMf[robot.model.getFrameId("gripper_left_joint")].translation.T.flat)

// print("Finally reached = ", finalPosEff)
// print("Distance between hand and target = ", np.linalg.norm(finalPosEff - target))
// print("Distance to default state = ", np.linalg.norm(x0 - np.array(xT.flat)))
// # print("XY distance to CoM reference = ", np.linalg.norm(com[:2] - comRef[:2]))