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
#include "crocoddyl/multibody/actuations/full.hpp"
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
    pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf", modeld);
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

    // We define an actuation model
    auto actuation = make_shared<ActuationModelFullTpl<double>>(state);

    // Next, we need to create an action model for running and terminal knots. The
    // forward dynamics (computed using ABA) are implemented
    // inside DifferentialActionModelFullyActuated.
    auto runningDAM =
        make_shared<DifferentialActionModelFreeFwdDynamicsTpl<double>>(state, actuation, runningCostModel);

    auto runningModel = make_shared<IntegratedActionModelEulerTpl<double>>(runningDAM, double(1e-3));
    auto terminalModel = make_shared<IntegratedActionModelEulerTpl<double>>(runningDAM, double(0.));

    return {runningModel, terminalModel};
}

int main(int argc, char *argv[])
{
    unsigned int N = 100;  // number of nodes
    unsigned int T = 5e3;  // number of trials
    unsigned int MAXITER = 1;

    if (argc > 1)
    {
        T = atoi(argv[1]);
    }

    // Building the running and terminal models
    auto [runningModel, terminalModel] = build_arm_action_models();

    // Get the initial state
    shared_ptr<StateMultibody> state = boost::static_pointer_cast<StateMultibody>(runningModel->get_state());
    std::cout << "NQ: " << state->get_nq() << std::endl;
    std::cout << "Number of nodes: " << N << std::endl << std::endl;
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
    Eigen::ArrayXd duration(T);
    for (unsigned int i = 0; i < T; ++i)
    {
        Timer timer;
        ddp.solve(xs, us, MAXITER, false, 0.1);
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
        Timer timer;
        problem.calc(xs, us);
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
        Timer timer;
        problem.calcDiff(xs, us);
        duration[i] = timer.get_duration();
    }

    avrg_duration = duration.sum() / T;
    min_duration = duration.minCoeff();
    max_duration = duration.maxCoeff();
    std::cout << "  ShootingProblem.calcDiff [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration
              << ")" << std::endl;
}
