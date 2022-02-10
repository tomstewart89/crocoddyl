///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/optctrl/shooting.hpp"

#include <iostream>

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif  // CROCODDYL_WITH_MULTITHREADING

namespace crocoddyl
{

ShootingProblem::ShootingProblem(const VectorXs& x0,
                                 const std::vector<boost::shared_ptr<ActionModelAbstract> >& running_models,
                                 boost::shared_ptr<ActionModelAbstract> terminal_model)
    : cost_(double(0.)),
      T_(running_models.size()),
      x0_(x0),
      terminal_model_(terminal_model),
      running_models_(running_models),
      nx_(running_models[0]->get_state()->get_nx()),
      ndx_(running_models[0]->get_state()->get_ndx()),
      nu_max_(running_models[0]->get_nu()),
      nthreads_(1)
{
    for (std::size_t i = 1; i < T_; ++i)
    {
        const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
        const std::size_t nu = model->get_nu();
        if (nu_max_ < nu)
        {
            nu_max_ = nu;
        }
    }
    if (static_cast<std::size_t>(x0.size()) != nx_)
    {
        throw_pretty("Invalid argument: "
                     << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    for (std::size_t i = 1; i < T_; ++i)
    {
        const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
        if (model->get_state()->get_nx() != nx_)
        {
            throw_pretty("Invalid argument: "
                         << "nx in " << i << " node is not consistent with the other nodes")
        }
        if (model->get_state()->get_ndx() != ndx_)
        {
            throw_pretty("Invalid argument: "
                         << "ndx in " << i << " node is not consistent with the other nodes")
        }
    }
    if (terminal_model_->get_state()->get_nx() != nx_)
    {
        throw_pretty("Invalid argument: "
                     << "nx in terminal node is not consistent with the other nodes")
    }
    if (terminal_model_->get_state()->get_ndx() != ndx_)
    {
        throw_pretty("Invalid argument: "
                     << "ndx in terminal node is not consistent with the other nodes")
    }
    allocateData();

#ifdef CROCODDYL_WITH_MULTITHREADING
    nthreads_ = CROCODDYL_WITH_NTHREADS;
#endif
}

ShootingProblem::ShootingProblem(const VectorXs& x0,
                                 const std::vector<boost::shared_ptr<ActionModelAbstract> >& running_models,
                                 boost::shared_ptr<ActionModelAbstract> terminal_model,
                                 const std::vector<boost::shared_ptr<ActionDataAbstract> >& running_datas,
                                 boost::shared_ptr<ActionDataAbstract> terminal_data)
    : cost_(double(0.)),
      T_(running_models.size()),
      x0_(x0),
      terminal_model_(terminal_model),
      terminal_data_(terminal_data),
      running_models_(running_models),
      running_datas_(running_datas),
      nx_(running_models[0]->get_state()->get_nx()),
      ndx_(running_models[0]->get_state()->get_ndx()),
      nu_max_(running_models[0]->get_nu()),
      nthreads_(1)
{
    for (std::size_t i = 1; i < T_; ++i)
    {
        const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
        const std::size_t nu = model->get_nu();
        if (nu_max_ < nu)
        {
            nu_max_ = nu;
        }
    }
    if (static_cast<std::size_t>(x0.size()) != nx_)
    {
        throw_pretty("Invalid argument: "
                     << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    const std::size_t Td = running_datas.size();
    if (Td != T_)
    {
        throw_pretty("Invalid argument: "
                     << "the number of running models and datas are not the same (" + std::to_string(T_) +
                            " != " + std::to_string(Td) + ")")
    }
    for (std::size_t i = 0; i < T_; ++i)
    {
        const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
        const boost::shared_ptr<ActionDataAbstract>& data = running_datas_[i];
        if (model->get_state()->get_nx() != nx_)
        {
            throw_pretty("Invalid argument: "
                         << "nx in " << i << " node is not consistent with the other nodes")
        }
        if (model->get_state()->get_ndx() != ndx_)
        {
            throw_pretty("Invalid argument: "
                         << "ndx in " << i << " node is not consistent with the other nodes")
        }
        if (!model->checkData(data))
        {
            throw_pretty("Invalid argument: "
                         << "action data in " << i << " node is not consistent with the action model")
        }
    }
    if (!terminal_model->checkData(terminal_data))
    {
        throw_pretty("Invalid argument: "
                     << "terminal action data is not consistent with the terminal action model")
    }

#ifdef CROCODDYL_WITH_MULTITHREADING
    nthreads_ = CROCODDYL_WITH_NTHREADS;
#endif
}

ShootingProblem::ShootingProblem(const ShootingProblem& problem)
    : cost_(double(0.)),
      T_(problem.get_T()),
      x0_(problem.get_x0()),
      terminal_model_(problem.get_terminalModel()),
      terminal_data_(problem.get_terminalData()),
      running_models_(problem.get_runningModels()),
      running_datas_(problem.get_runningDatas()),
      nx_(problem.get_nx()),
      ndx_(problem.get_ndx()),
      nu_max_(problem.get_nu_max())
{
}

ShootingProblem::~ShootingProblem() {}

double ShootingProblem::calc(const std::vector<VectorXs>& xs, const std::vector<VectorXs>& us)
{
    if (xs.size() != T_ + 1)
    {
        throw_pretty("Invalid argument: "
                     << "xs has wrong dimension (it should be " + std::to_string(T_ + 1) + ")");
    }
    if (us.size() != T_)
    {
        throw_pretty("Invalid argument: "
                     << "us has wrong dimension (it should be " + std::to_string(T_) + ")");
    }

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(nthreads_)
#endif
    for (std::size_t i = 0; i < T_; ++i)
    {
        const std::size_t nu = running_models_[i]->get_nu();
        if (nu != 0)
        {
            running_models_[i]->calc(running_datas_[i], xs[i], us[i].head(nu));
        }
        else
        {
            running_models_[i]->calc(running_datas_[i], xs[i]);
        }
    }
    terminal_model_->calc(terminal_data_, xs.back());

    cost_ = double(0.);
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp simd reduction(+ : cost_)
#endif
    for (std::size_t i = 0; i < T_; ++i)
    {
        cost_ += running_datas_[i]->cost;
    }
    cost_ += terminal_data_->cost;
    return cost_;
}

double ShootingProblem::calcDiff(const std::vector<VectorXs>& xs, const std::vector<VectorXs>& us)
{
    if (xs.size() != T_ + 1)
    {
        throw_pretty("Invalid argument: "
                     << "xs has wrong dimension (it should be " + std::to_string(T_ + 1) + ")");
    }
    if (us.size() != T_)
    {
        throw_pretty("Invalid argument: "
                     << "us has wrong dimension (it should be " + std::to_string(T_) + ")");
    }

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(nthreads_)
#endif
    for (std::size_t i = 0; i < T_; ++i)
    {
        if (running_models_[i]->get_nu() != 0)
        {
            const std::size_t nu = running_models_[i]->get_nu();
            running_models_[i]->calcDiff(running_datas_[i], xs[i], us[i].head(nu));
        }
        else
        {
            running_models_[i]->calcDiff(running_datas_[i], xs[i]);
        }
    }
    terminal_model_->calcDiff(terminal_data_, xs.back());

    cost_ = double(0.);
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp simd reduction(+ : cost_)
#endif
    for (std::size_t i = 0; i < T_; ++i)
    {
        cost_ += running_datas_[i]->cost;
    }
    cost_ += terminal_data_->cost;

    return cost_;
}

void ShootingProblem::rollout(const std::vector<VectorXs>& us, std::vector<VectorXs>& xs)
{
    if (xs.size() != T_ + 1)
    {
        throw_pretty("Invalid argument: "
                     << "xs has wrong dimension (it should be " + std::to_string(T_ + 1) + ")");
    }
    if (us.size() != T_)
    {
        throw_pretty("Invalid argument: "
                     << "us has wrong dimension (it should be " + std::to_string(T_) + ")");
    }

    xs[0] = x0_;
    for (std::size_t i = 0; i < T_; ++i)
    {
        const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
        const boost::shared_ptr<ActionDataAbstract>& data = running_datas_[i];
        const VectorXs& x = xs[i];
        const std::size_t nu = running_models_[i]->get_nu();
        if (model->get_nu() != 0)
        {
            const VectorXs& u = us[i];
            model->calc(data, x, u.head(nu));
        }
        else
        {
            model->calc(data, x);
        }
        xs[i + 1] = data->xnext;
    }
    terminal_model_->calc(terminal_data_, xs.back());
}

std::vector<typename MathBaseTpl<double>::VectorXs> ShootingProblem::rollout_us(const std::vector<VectorXs>& us)
{
    std::vector<VectorXs> xs;
    xs.resize(T_ + 1);
    rollout(us, xs);
    return xs;
}

void ShootingProblem::quasiStatic(std::vector<VectorXs>& us, const std::vector<VectorXs>& xs)
{
    if (xs.size() != T_)
    {
        throw_pretty("Invalid argument: "
                     << "xs has wrong dimension (it should be " + std::to_string(T_) + ")");
    }
    if (us.size() != T_)
    {
        throw_pretty("Invalid argument: "
                     << "us has wrong dimension (it should be " + std::to_string(T_) + ")");
    }

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(nthreads_)
#endif
    for (std::size_t i = 0; i < T_; ++i)
    {
        const std::size_t nu = running_models_[i]->get_nu();
        running_models_[i]->quasiStatic(running_datas_[i], us[i].head(nu), xs[i]);
    }
}

std::vector<typename MathBaseTpl<double>::VectorXs> ShootingProblem::quasiStatic_xs(const std::vector<VectorXs>& xs)
{
    std::vector<VectorXs> us;
    us.resize(T_);
    for (std::size_t i = 0; i < T_; ++i)
    {
        us[i] = VectorXs::Zero(running_models_[i]->get_nu());
    }
    quasiStatic(us, xs);
    return us;
}

void ShootingProblem::circularAppend(boost::shared_ptr<ActionModelAbstract> model,
                                     boost::shared_ptr<ActionDataAbstract> data)
{
    if (!model->checkData(data))
    {
        throw_pretty("Invalid argument: "
                     << "action data is not consistent with the action model")
    }
    if (model->get_state()->get_nx() != nx_)
    {
        throw_pretty("Invalid argument: "
                     << "nx is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_)
    {
        throw_pretty("Invalid argument: "
                     << "ndx node is not consistent with the other nodes")
    }
    if (model->get_nu() > nu_max_)
    {
        throw_pretty("Invalid argument: "
                     << "nu node is greater than the maximum nu")
    }

    for (std::size_t i = 0; i < T_ - 1; ++i)
    {
        running_models_[i] = running_models_[i + 1];
        running_datas_[i] = running_datas_[i + 1];
    }
    running_models_.back() = model;
    running_datas_.back() = data;
}

void ShootingProblem::circularAppend(boost::shared_ptr<ActionModelAbstract> model)
{
    if (model->get_state()->get_nx() != nx_)
    {
        throw_pretty("Invalid argument: "
                     << "nx is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_)
    {
        throw_pretty("Invalid argument: "
                     << "ndx node is not consistent with the other nodes")
    }
    if (model->get_nu() > nu_max_)
    {
        throw_pretty("Invalid argument: "
                     << "nu node is greater than the maximum nu")
    }

    for (std::size_t i = 0; i < T_ - 1; ++i)
    {
        running_models_[i] = running_models_[i + 1];
        running_datas_[i] = running_datas_[i + 1];
    }
    running_models_.back() = model;
    running_datas_.back() = model->createData();
}

void ShootingProblem::updateNode(const std::size_t i, boost::shared_ptr<ActionModelAbstract> model,
                                 boost::shared_ptr<ActionDataAbstract> data)
{
    if (i >= T_ + 1)
    {
        throw_pretty("Invalid argument: "
                     << "i is bigger than the allocated horizon (it should be less than or equal to " +
                            std::to_string(T_ + 1) + ")");
    }
    if (!model->checkData(data))
    {
        throw_pretty("Invalid argument: "
                     << "action data is not consistent with the action model")
    }
    if (model->get_state()->get_nx() != nx_)
    {
        throw_pretty("Invalid argument: "
                     << "nx is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_)
    {
        throw_pretty("Invalid argument: "
                     << "ndx node is not consistent with the other nodes")
    }
    if (model->get_nu() > nu_max_)
    {
        throw_pretty("Invalid argument: "
                     << "nu node is greater than the maximum nu")
    }

    if (i == T_)
    {
        terminal_model_ = model;
        terminal_data_ = data;
    }
    else
    {
        running_models_[i] = model;
        running_datas_[i] = data;
    }
}

void ShootingProblem::updateModel(const std::size_t i, boost::shared_ptr<ActionModelAbstract> model)
{
    if (i >= T_ + 1)
    {
        throw_pretty("Invalid argument: "
                     << "i is bigger than the allocated horizon (it should be lower than " + std::to_string(T_ + 1) +
                            ")");
    }
    if (model->get_state()->get_nx() != nx_)
    {
        throw_pretty("Invalid argument: "
                     << "nx is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_)
    {
        throw_pretty("Invalid argument: "
                     << "ndx is not consistent with the other nodes")
    }
    if (model->get_nu() > nu_max_)
    {
        throw_pretty("Invalid argument: "
                     << "nu node is greater than the maximum nu")
    }

    if (i == T_)
    {
        terminal_model_ = model;
        terminal_data_ = terminal_model_->createData();
    }
    else
    {
        running_models_[i] = model;
        running_datas_[i] = model->createData();
    }
}

std::size_t ShootingProblem::get_T() const { return T_; }

const typename MathBaseTpl<double>::VectorXs& ShootingProblem::get_x0() const { return x0_; }

void ShootingProblem::allocateData()
{
    running_datas_.resize(T_);
    for (std::size_t i = 0; i < T_; ++i)
    {
        const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
        running_datas_[i] = model->createData();
    }
    terminal_data_ = terminal_model_->createData();
}

const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<double> > >& ShootingProblem::get_runningModels()
    const
{
    return running_models_;
}

const boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<double> >& ShootingProblem::get_terminalModel() const
{
    return terminal_model_;
}

const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<double> > >& ShootingProblem::get_runningDatas()
    const
{
    return running_datas_;
}

const boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<double> >& ShootingProblem::get_terminalData() const
{
    return terminal_data_;
}

void ShootingProblem::set_x0(const VectorXs& x0_in)
{
    if (x0_in.size() != x0_.size())
    {
        throw_pretty("Invalid argument: "
                     << "invalid size of x0 provided: Expected " << x0_.size() << ", received " << x0_in.size());
    }
    x0_ = x0_in;
}

void ShootingProblem::set_runningModels(const std::vector<boost::shared_ptr<ActionModelAbstract> >& models)
{
    for (std::size_t i = 0; i < T_; ++i)
    {
        const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
        if (model->get_state()->get_nx() != nx_)
        {
            throw_pretty("Invalid argument: "
                         << "nx in " << i << " node is not consistent with the other nodes")
        }
        if (model->get_state()->get_ndx() != ndx_)
        {
            throw_pretty("Invalid argument: "
                         << "ndx in " << i << " node is not consistent with the other nodes")
        }
        if (model->get_nu() > nu_max_)
        {
            throw_pretty("Invalid argument: "
                         << "nu node is greater than the maximum nu")
        }
    }

    T_ = models.size();
    running_models_.clear();
    running_datas_.clear();
    for (std::size_t i = 0; i < T_; ++i)
    {
        const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
        running_datas_.push_back(model->createData());
    }
}

void ShootingProblem::set_terminalModel(boost::shared_ptr<ActionModelAbstract> model)
{
    if (model->get_state()->get_nx() != nx_)
    {
        throw_pretty("Invalid argument: "
                     << "nx is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_)
    {
        throw_pretty("Invalid argument: "
                     << "ndx is not consistent with the other nodes")
    }
    terminal_model_ = model;
    terminal_data_ = terminal_model_->createData();
}

void ShootingProblem::set_nthreads(const int nthreads)
{
#ifndef CROCODDYL_WITH_MULTITHREADING
    (void)nthreads;
    std::cerr << "Warning: the number of threads won't affect the computational performance as multithreading "
                 "support is not enabled."
              << std::endl;
#else
    if (nthreads < 1)
    {
        nthreads_ = CROCODDYL_WITH_NTHREADS;
    }
    else
    {
        nthreads_ = static_cast<std::size_t>(nthreads);
    }
#endif
}

std::size_t ShootingProblem::get_nx() const { return nx_; }

std::size_t ShootingProblem::get_ndx() const { return ndx_; }

std::size_t ShootingProblem::get_nu_max() const { return nu_max_; }

std::size_t ShootingProblem::get_nthreads() const
{
#ifndef CROCODDYL_WITH_MULTITHREADING
    std::cerr << "Warning: the number of threads won't affect the computational performance as multithreading "
                 "support is not enabled."
              << std::endl;
#endif
    return nthreads_;
}

std::ostream& operator<<(std::ostream& os, const ShootingProblem& problem)
{
    os << "ShootingProblem (T=" << problem.get_T() << ", nx=" << problem.get_nx() << ", ndx=" << problem.get_ndx()
       << ", nu_max=" << problem.get_nu_max() << ") " << std::endl;
    os << "  Models:" << std::endl;
    const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<double> > >& runningModels =
        problem.get_runningModels();
    for (std::size_t t = 0; t < problem.get_T(); ++t)
    {
        os << "    " << t << ": " << *runningModels[t] << std::endl;
    }
    os << "    " << problem.get_T() << ": " << *problem.get_terminalModel();
    return os;
}

}  // namespace crocoddyl
