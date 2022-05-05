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
#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl
{

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
class SolverDDP : public SolverAbstract
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
    virtual ~SolverDDP();

    virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                       const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t maxiter = 100,
                       const bool is_feasible = false, const double regInit = 1e-9);
    virtual void computeDirection(const bool) {}
    virtual double tryStep(const double) { return 0; }
    virtual double stoppingCriteria();
    virtual const Eigen::Vector2d& expectedImprovement() { return d_; }

    Eigen::Vector2d expected_improvement();

    /**
     * @brief Update the Jacobian and Hessian of the optimal control problem
     *
     * These derivatives are computed around the guess state and control trajectory. These trajectory can be set by
     * using `setCandidate()`.
     *
     * @return  The total cost around the guess trajectory
     */
    virtual double calcDiff();

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
    virtual void backwardPass();

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
    virtual void forwardPass(const double stepLength);

    /**
     * @brief Compute the feedforward and feedback terms using a Cholesky decomposition
     *
     * To compute the feedforward \f$\mathbf{k}_k\f$ and feedback \f$\mathbf{K}_k\f$ terms, we use a Cholesky
     * decomposition to solve \f$\mathbf{Q}_{\mathbf{uu}_k}^{-1}\f$ term:
     * \f{eqnarray}
     * \mathbf{k}_k &=& \mathbf{Q}_{\mathbf{uu}_k}^{-1}\mathbf{Q}_{\mathbf{u}},\\
     * \mathbf{K}_k &=& \mathbf{Q}_{\mathbf{uu}_k}^{-1}\mathbf{Q}_{\mathbf{ux}}.
     * \f}
     *
     * Note that if the Cholesky decomposition fails, then we re-start the backward pass and increase the
     * state and control regularization values.
     */
    virtual void computeGains(const std::size_t t);

    /**
     * @brief Increase the state and control regularization values by a `regfactor_` factor
     */
    void increaseRegularization();

    /**
     * @brief Decrease the state and control regularization values by a `regfactor_` factor
     */
    void decreaseRegularization();

    /**
     * @brief Allocate all the internal data needed for the solver
     */
    virtual void allocateData();

   protected:
    double reg_incfactor_;  //!< Regularization factor used to increase the damping value
    double reg_decfactor_;  //!< Regularization factor used to decrease the damping value
    double reg_min_;        //!< Minimum allowed regularization value
    double reg_max_;        //!< Maximum allowed regularization value

    double cost_try_;                      //!< Total cost computed by line-search procedure
    std::vector<Eigen::VectorXd> xs_try_;  //!< State trajectory computed by line-search procedure
    std::vector<Eigen::VectorXd> us_try_;  //!< Control trajectory computed by line-search procedure
    std::vector<Eigen::VectorXd> dx_;

    // allocate data
    std::vector<Eigen::MatrixXd> Vxx_;  //!< Hessian of the Value function
    Eigen::MatrixXd Vxx_tmp_;           //!< Temporary variable for ensuring symmetry of Vxx
    std::vector<Eigen::VectorXd> Vx_;   //!< Gradient of the Value function
    std::vector<Eigen::MatrixXd> Qxx_;  //!< Hessian of the Hamiltonian
    std::vector<Eigen::MatrixXd> Qxu_;  //!< Hessian of the Hamiltonian
    std::vector<Eigen::MatrixXd> Quu_;  //!< Hessian of the Hamiltonian
    std::vector<Eigen::VectorXd> Qx_;   //!< Gradient of the Hamiltonian
    std::vector<Eigen::VectorXd> Qu_;   //!< Gradient of the Hamiltonian
    std::vector<MatrixXdRowMajor> K_;   //!< Feedback gains
    std::vector<Eigen::VectorXd> k_;    //!< Feed-forward terms
    std::vector<Eigen::VectorXd> fs_;   //!< Gaps/defects between shooting nodes

    Eigen::VectorXd xnext_;                   //!< Next state
    MatrixXdRowMajor FxTVxx_p_;               //!< fxTVxx_p_
    std::vector<MatrixXdRowMajor> FuTVxx_p_;  //!< fuTVxx_p_
    Eigen::VectorXd fTVxx_p_;                 //!< fTVxx_p term
    std::vector<Eigen::VectorXd> Quuk_;       //!< Quuk term
    std::vector<double> alphas_;              //!< Set of step lengths using by the line-search procedure
    double th_grad_;                          //!< Tolerance of the expected gradient used for testing the step
    double th_gaptol_;                        //!< Threshold limit to check non-zero gaps
    double th_stepdec_;                       //!< Step-length threshold used to decrease regularization
    double th_stepinc_;                       //!< Step-length threshold used to increase regularization
    bool was_feasible_;                       //!< Label that indicates in the previous iterate was feasible
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_DDP_HPP_
