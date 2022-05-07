///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_DDP_HPP_
#define CROCODDYL_CORE_SOLVERS_DDP_HPP_

#include <Eigen/Cholesky>
#include <vector>

#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl
{
static std::vector<Eigen::VectorXd> DEFAULT_VECTOR;

/**
 * @brief Differential Dynamic Programming (DDP) solver
 *
 * The DDP solver computes an optimal trajectory and control commands by iterates running `backwardPass()` and
 * `forwardPass()`. The backward-pass updates locally the quadratic approximation of the problem and computes descent
 * direction. If the warm-start is feasible, then it computes the gaps \f$\mathbf{\bar{f}}_s\f$ and run a modified
 * Riccati sweep:
 * \f{eqnarray*}
 *   \mathbf{Q}_{\mathbf{x}_k} &=& \mathbf{l}_{\mathbf{x}_k} + \mathbf{f}^\top_{\mathbf{x}_k} (V_{\mathbf{x}_{k+1}} +
 * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
 *   \mathbf{Q}_{\mathbf{u}_k} &=& \mathbf{l}_{\mathbf{u}_k} + \mathbf{f}^\top_{\mathbf{u}_k} (V_{\mathbf{x}_{k+1}} +
 * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
 *   \mathbf{Q}_{\mathbf{xx}_k} &=& \mathbf{l}_{\mathbf{xx}_k} + \mathbf{f}^\top_{\mathbf{x}_k} V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{x}_k},\\
 *   \mathbf{Q}_{\mathbf{xu}_k} &=& \mathbf{l}_{\mathbf{xu}_k} + \mathbf{f}^\top_{\mathbf{x}_k} V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{u}_k},\\
 *   \mathbf{Q}_{\mathbf{uu}_k} &=& \mathbf{l}_{\mathbf{uu}_k} + \mathbf{f}^\top_{\mathbf{u}_k} V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{u}_k}.
 * \f}
 * Then, the forward-pass rollouts this new policy by integrating the system dynamics along a tuple of optimized
 * control commands \f$\mathbf{u}^*_s\f$, i.e.
 * \f{eqnarray}
 *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0,\\
 *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k + \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\
 *   \mathbf{\hat{x}}_{k+1} &=& \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k).
 * \f}
 *
 * \sa `backwardPass()` and `forwardPass()`
 */
class SolverDDP
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef typename MathBaseTpl<double>::MatrixXsRowMajor MatrixXdRowMajor;

    /**
     * @brief Initialize the DDP solver
     *
     * @param[in] problem  Shooting problem
     */
    explicit SolverDDP(ShootingProblem& problem);
    ~SolverDDP() = default;

    bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
               const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t maxiter = 100,
               const bool is_feasible = false, const double regInit = 1e-9);
    double stoppingCriteria();

    Eigen::Vector2d expectedImprovement();

    /**
     * @brief Update the Jacobian and Hessian of the optimal control problem
     *
     * These derivatives are computed around the guess state and control trajectory. These trajectory can be set by
     * using `setCandidate()`.
     *
     * @return  The total cost around the guess trajectory
     */
    double calcDiff();

    /**
     * @brief Run the backward pass (Riccati sweep)
     *
     * It assumes that the Jacobian and Hessians of the optimal control problem have been compute (i.e. `calcDiff()`).
     * The backward pass handles infeasible guess through a modified Riccati sweep:
     * \f{eqnarray*}
     *   \mathbf{Q}_{\mathbf{x}_k} &=& \mathbf{l}_{\mathbf{x}_k} + \mathbf{f}^\top_{\mathbf{x}_k} (V_{\mathbf{x}_{k+1}}
     * +
     * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
     *   \mathbf{Q}_{\mathbf{u}_k} &=& \mathbf{l}_{\mathbf{u}_k} + \mathbf{f}^\top_{\mathbf{u}_k} (V_{\mathbf{x}_{k+1}}
     * +
     * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
     *   \mathbf{Q}_{\mathbf{xx}_k} &=& \mathbf{l}_{\mathbf{xx}_k} + \mathbf{f}^\top_{\mathbf{x}_k}
     * V_{\mathbf{xx}_{k+1}}
     * \mathbf{f}_{\mathbf{x}_k},\\
     *   \mathbf{Q}_{\mathbf{xu}_k} &=& \mathbf{l}_{\mathbf{xu}_k} + \mathbf{f}^\top_{\mathbf{x}_k}
     * V_{\mathbf{xx}_{k+1}}
     * \mathbf{f}_{\mathbf{u}_k},\\
     *   \mathbf{Q}_{\mathbf{uu}_k} &=& \mathbf{l}_{\mathbf{uu}_k} + \mathbf{f}^\top_{\mathbf{u}_k}
     * V_{\mathbf{xx}_{k+1}} \mathbf{f}_{\mathbf{u}_k}, \f} where
     * \f$\mathbf{l}_{\mathbf{x}_k}\f$,\f$\mathbf{l}_{\mathbf{u}_k}\f$,\f$\mathbf{f}_{\mathbf{x}_k}\f$ and
     * \f$\mathbf{f}_{\mathbf{u}_k}\f$ are the Jacobians of the cost function and dynamics,
     * \f$\mathbf{l}_{\mathbf{xx}_k}\f$,\f$\mathbf{l}_{\mathbf{xu}_k}\f$ and \f$\mathbf{l}_{\mathbf{uu}_k}\f$ are the
     * Hessians of the cost function, \f$V_{\mathbf{x}_{k+1}}\f$ and \f$V_{\mathbf{xx}_{k+1}}\f$ defines the
     * linear-quadratic approximation of the Value function, and \f$\mathbf{\bar{f}}_{k+1}\f$ describes the gaps of the
     * dynamics.
     */
    void backwardPass();

    /**
     * @brief Run the forward pass or rollout
     *
     * It rollouts the action model given the computed policy (feedforward terns and feedback gains) by the
     * `backwardPass()`:
     * \f{eqnarray}
     *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0,\\
     *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k + \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\
     *   \mathbf{\hat{x}}_{k+1} &=& \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k).
     * \f}
     * We can define different step lengths \f$\alpha\f$.
     *
     * @param  stepLength  applied step length (\f$0\leq\alpha\leq1\f$)
     */
    void forwardPass(const double stepLength);

    /**
     * @brief Set the solver candidate warm-point values \f$(\mathbf{x}_s,\mathbf{u}_s)\f$
     *
     * The solver candidates are defined as a state and control trajectories \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ of
     * \f$T+1\f$ and \f$T\f$ elements, respectively. Additionally, we need to define is
     * \f$(\mathbf{x}_s,\mathbf{u}_s)\f$ pair is feasible, this means that the dynamics rollout give us produces
     * \f$\mathbf{x}_s\f$.
     *
     * @param[in]  xs          state trajectory of \f$T+1\f$ elements (default [])
     * @param[in]  us          control trajectory of \f$T\f$ elements (default [])
     * @param[in]  isFeasible  true if the \p xs are obtained from integrating the \p us (rollout)
     */
    void setCandidate(const std::vector<Eigen::VectorXd>& xs_warm = DEFAULT_VECTOR,
                      const std::vector<Eigen::VectorXd>& us_warm = DEFAULT_VECTOR, const bool is_feasible = false);

    /**
     * @brief Return the total cost
     */
    double get_cost() const { return cost_; }

   protected:
    double reg_incfactor_ = 10.0;  //!< Regularization factor used to increase the damping value
    double reg_decfactor_ = 10.0;  //!< Regularization factor used to decrease the damping value
    double reg_min_ = 1e-9;        //!< Minimum allowed regularization value
    double reg_max_ = 1e9;         //!< Maximum allowed regularization value

    double cost_try_ = 0.0;                //!< Total cost computed by line-search procedure
    std::vector<Eigen::VectorXd> xs_try_;  //!< State trajectory computed by line-search procedure
    std::vector<Eigen::VectorXd> us_try_;  //!< Control trajectory computed by line-search procedure
    std::vector<Eigen::VectorXd> dx_;

    // allocate data
    std::vector<Eigen::MatrixXd> Vxx_;  //!< Hessian of the Value function
    std::vector<Eigen::VectorXd> Vx_;   //!< Gradient of the Value function
    std::vector<Eigen::VectorXd> Qu_;   //!< Gradient of the Hamiltonian
    std::vector<MatrixXdRowMajor> K_;   //!< Feedback gains
    std::vector<Eigen::VectorXd> k_;    //!< Feed-forward terms
    std::vector<Eigen::VectorXd> fs_;   //!< Gaps/defects between shooting nodes

    Eigen::VectorXd xnext_;              //!< Next state
    std::vector<Eigen::VectorXd> Quuk_;  //!< Quuk term
    std::vector<double> alphas_;         //!< Set of step lengths using by the line-search procedure
    double th_grad_ = 1e-12;             //!< Tolerance of the expected gradient used for testing the step
    double th_gaptol_ = 1e-16;           //!< Threshold limit to check non-zero gaps
    double th_stepdec_ = 0.5;            //!< Step-length threshold used to decrease regularization
    double th_stepinc_ = 0.01;           //!< Step-length threshold used to increase regularization
    bool was_feasible_ = false;          //!< Label that indicates in the previous iterate was feasible

    ShootingProblem& problem_;         //!< optimal control problem
    std::vector<Eigen::VectorXd> xs_;  //!< State trajectory
    std::vector<Eigen::VectorXd> us_;  //!< Control trajectory
    bool is_feasible_ = false;         //!< Label that indicates is the iteration is feasible
    double cost_ = 0.0;                //!< Total cost
    double reg_ = 1e-9;                //!< Current regularization value
    double steplength_ = 1.0;          //!< Current applied step-length
    double dV_ = 0.0;                  //!< Cost reduction obtained by `tryStep()`
    double dVexp_ = 0.0;               //!< Expected cost reduction
    double th_acceptstep_ = 0.1;       //!< Threshold used for accepting step
    double th_stop_ = 1e-9;            //!< Tolerance for stopping the algorithm
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_DDP_HPP_
