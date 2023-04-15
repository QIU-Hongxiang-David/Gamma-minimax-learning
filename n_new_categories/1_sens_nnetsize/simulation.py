import numpy as np
from numpy.random import uniform
from scipy.optimize import linprog
from scipy.stats import binom
import torch
import matplotlib.pyplot as plt
import pickle

# All inputs that can be array-like or a floating number (i.e. except counts and bools) should be tensors
# All user defined function inputs and outputs should be tensors
# A sample is a vector of length sample_size with the i-th entry being the number of species with i occurrences (i=1 to sample_size)

default_dtype = torch.double
use_cuda = False
if use_cuda:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


sample_size = 100
new_sample_size = 200


# functions and classes for estimators and Gamma-minimax estimators

def sufficient_statistic(multinomial_sample):
    global sample_size
    return np.bincount(multinomial_sample, minlength=sample_size + 1)[1:]


class Gamma_minimax_Problem(object):
    '''holds the estimator, an array of distributions and their prior probs, their true parameters and the summaries of distributions appearing in the constraints.'''

    def __init__(self, estimator, parameter_constraint_fun, b_ub=None, b_eq=None, Risk_fun=lambda estimates, true_parameters: torch.sum((estimates - true_parameters)**2, dim=2).mean(dim=1), p_init=None, n_distr=2000, init_prior_prob=None, MCMC_MC_size=100, parameter=None):
        '''n_distr: number of random distributions drawn initially
        parameter_constraint_fun: function that takes in distributions (array of tensors on simplexes) and returns 3 components: (1) parameter (2) summaries that appear in upper bound constraints (3) summaries that appear in equality constraints. Each component should be None (no constraint) or a 2D-tensor with each entry in the 0th dimension corresponding to each distribution. The parameter component cannot be None.
        b_ub, b_eq: vector of <= and == constraints in linear programming. can be None or numpy arrays or 1D iterables.
        estimator: initial estimator parameterized in PyTorch
        Risk_fun: function that takes in a 3D tensor of estimates (number of distributions X number of samples X dimension of estimate) and a 3D tensor of true parameters (number of distributions X number of samples X dimension of estimate) and outputs the Risk=average loss for each distribution. output should be a 1D tensor. default to squared-error loss
        init_prior_prob: initial probs for the distributions. if not None, its length should be n_distr. defaults to a point mass at the initial point
        MCMC_MC_size: Monte Caro size to estimate Risk in MCMC
        parameter: function that takes in distributions (array of tensors on simplexes) and returns the parameter. defaults to run parameter_constraint_fun and take the first output'''
        self.estimator = estimator
        self.parameter_constraint_fun = parameter_constraint_fun
        if parameter is None:
            def parameter(ps):
                true_parameters, dummy1, dummy2 = parameter_constraint_fun(ps)
                return true_parameters
            self.parameter = parameter
        else:
            self.parameter = parameter

        self.Risk_fun = Risk_fun
        self.MCMC_MC_size = MCMC_MC_size
        self.b_ub = np.array(b_ub) if b_ub is not None else None
        self.b_eq = np.concatenate((np.array(b_eq), np.ones(1))) if b_eq is not None else np.ones(1)

        if p_init is not None:
            self.distrs = [p_init]
            self.distrs.extend(self.generate_ps(n_distr - 1, p_init))
        else:
            self.distrs = self.generate_ps(n_distr, p_init)
        self.n_distr = n_distr

        if init_prior_prob is None:
            self.prior_prob = torch.zeros(self.n_distr)
            self.prior_prob[0] = 1
        else:
            self.prior_prob = torch.as_tensor(init_prior_prob)
        
        self.eval_parameter_constraint()

    def eval_parameter_constraint(self, index_of_first_new_distr=0):
        '''index_of_first_new_distr: index of the first new random distribution'''
        true_parameters, A_ubT, A_eqT = self.parameter_constraint_fun(self.distrs[index_of_first_new_distr:])
        if index_of_first_new_distr == 0:
            self.true_parameters = true_parameters
        else:
            self.true_parameters = torch.cat((self.true_parameters,true_parameters), dim=0)

        if A_ubT is None:
            self.A_ubT = None
        elif index_of_first_new_distr == 0:
            self.A_ubT = A_ubT
        else:
            self.A_ubT = torch.cat((self.A_ubT,A_ubT), dim=0)

        if A_eqT is None:
            self.A_eqT = None
        elif index_of_first_new_distr == 0:
            self.A_eqT = A_eqT
        else:
            self.A_eqT = torch.cat((self.A_eqT, A_eqT), dim=0)

    def enlarge_distr_grid(self, n_distr=1000):
        '''generate n_distr more distributions starting from the last distribution with positive prior prob; appends 0 to prior; calculate parameter and constraints'''
        non_zero_indices = self.prior_prob.nonzero().squeeze(1)
        p_init = self.distrs[non_zero_indices[-1]]
        self.distrs.extend(self.generate_ps(n_distr, p_init))
        self.prior_prob = torch.cat((self.prior_prob, torch.zeros(n_distr)))
        self.eval_parameter_constraint(self.n_distr)
        self.n_distr = len(self.distrs)

    def get_constraint_matrices(self):
        '''return a tuple of (1) 2D numpy array A_ub (2) 2D numpy array A_eq
        A_eq stacks the constraint that prior probability sums to 1 at the end'''
        np_A_ub=self.A_ubT.cpu().numpy().transpose() if self.A_ubT is not None else None

        if self.A_eqT is not None:
            np_A_eq = self.A_eqT.cpu().numpy().transpose()
            np_A_eq = np.vstack((np_A_eq, np.ones((1, self.prior_prob.size()[0]))))
        else:
            np_A_eq = np.ones((1, self.prior_prob.size()[0]))

        return np_A_ub, np_A_eq
    
    def draw_sample(self, n_sample, distributions=None):
        '''returns a 3D tensor (number of distributions X number of samples X sample_size). when distributions = None, use the distributions in the object'''
        global sample_size, default_dtype
        if distributions is None:
            distributions = self.distrs
        samples = torch.empty((len(distributions), n_sample, sample_size))
        for i, p in enumerate(distributions):
            #work on CPU so that numpy functions can be used
            multinomial_samples = torch.distributions.Multinomial(sample_size, p.cpu()).sample((n_sample,)).type(torch.int64)
            sufficient_stats = np.apply_along_axis(sufficient_statistic, 1, multinomial_samples)
            samples[i,:,:] = torch.as_tensor(sufficient_stats, dtype=default_dtype)
        return samples
    
    def calc_Risks_tensor(self, n_sample=30, estimator=None, distributions=None, distr_indices=None):
        '''calculate Risks of distributions and estimator via Monte Carlo. when estimator=None, use the current estimator in the object. when distributions=None, use the distributions in the object. distr_indices is the indices in distributions for which Risks are to be calculated; when distr_indices=None, calculate Risks for all distributions'''
        global sample_size, new_sample_size, default_dtype
        if distributions is None:
            distributions = self.distrs
        if estimator is None:
            estimator = self.estimator
        if distr_indices is None:
            distr_indices = range(len(distributions))
        n_distr = len(distr_indices)
        samples = torch.empty((n_distr, n_sample, sample_size))
        true_prediction_mean = torch.empty((n_distr, n_sample, 1))
        for i, p_index in enumerate(distr_indices):
            p = distributions[p_index]
            #work on CPU so that numpy functions can be used
            multinomial_samples = torch.distributions.Multinomial(sample_size, p.cpu()).sample((n_sample,))
            sufficient_stats = np.apply_along_axis(sufficient_statistic, 1, multinomial_samples.type(torch.int64))
            samples[i,:,:] = torch.as_tensor(sufficient_stats, dtype=default_dtype)
            prob_occur_new_sample = 1. - (1. - p) ** new_sample_size
            true_prediction_mean[i,:,:] = (multinomial_samples == 0).type(default_dtype).mm(prob_occur_new_sample.unsqueeze(1).cpu())
        estimates = estimator(samples.reshape(n_distr*n_sample, -1)).view(n_distr, n_sample, -1)
        Risks = self.Risk_fun(estimates, true_prediction_mean)
        return Risks
    
    def calc_Risks_tensor_memeff(self, n_sample=2000, estimator=None, distributions=None, distr_indices=None):
        '''memory efficient version of Gamma_minimax_Problem.calc_Risks_tensor. may be slower and have trouble when used with autograd. may be preferrable when evaluating Risk or Bayes risk with a large number of distributions and large n_sample'''
        global sample_size, new_sample_size, default_dtype, use_cuda
        if distributions is None:
            distributions = self.distrs
        if estimator is None:
            estimator = self.estimator
        if distr_indices is None:
            distr_indices = range(len(distributions))
        n_distr = len(distr_indices)
        Risks = torch.empty(n_distr)
        for i, p_index in enumerate(distr_indices):
            p = distributions[p_index]
            #work on CPU so that numpy functions can be used
            multinomial_samples = torch.distributions.Multinomial(sample_size, p.cpu()).sample((n_sample,))
            sufficient_stats = np.apply_along_axis(sufficient_statistic, 1, multinomial_samples.type(torch.int64))
            samples = torch.as_tensor(sufficient_stats, dtype=default_dtype).unsqueeze(0)
            prob_occur_new_sample = 1. - (1. - p) ** new_sample_size
            true_prediction_mean = torch.as_tensor((multinomial_samples == 0).type(default_dtype).mm(prob_occur_new_sample.unsqueeze(1).cpu())).unsqueeze(0)
            if use_cuda:
                true_prediction_mean = true_prediction_mean.cuda()
            estimates = estimator(samples.reshape(n_sample, -1)).view(1, n_sample, -1)
            Risks[i] = self.Risk_fun(estimates, true_prediction_mean)
        return Risks
    
    def log_pseudo_prior(self, p):
        global MCMC_normal_distribution, MCMC_negbinomial_distribution, default_dtype
        return torch.tensor(MCMC_normal_distribution.log_prob(self.parameter((p,))).item() * 30. + MCMC_negbinomial_distribution.log_prob(torch.tensor(p.size(), dtype=default_dtype)).item() * 10.)

    def generate_ps(self, n_distr, p_init=None):
        '''generate a list of n_distr tensors of multinomial probabilities'''
        global MCMC_Poisson_distribution

        if n_distr == 0:
            return []
        
        if p_init is None:
            global sample_size
            k_init = max(np.ceil(.5 * sample_size), 2)
            p_init = torch.distributions.dirichlet.Dirichlet(torch.ones(int(k_init))).sample()
        k_init = p_init.size()[0]
        log_pseudo_prior_init = self.log_pseudo_prior(p_init)
        with torch.no_grad():
            Risk_init = self.calc_Risks_tensor(n_sample=self.MCMC_MC_size, distributions=(p_init,))[0]

        ps = [None]*n_distr
        i = 0
        while(True):
            #within dimension jump
            to_prop_dirichlet = torch.distributions.dirichlet.Dirichlet(1. * p_init + 1.)
            p_prop = to_prop_dirichlet.sample()
            log_pseudo_prior_prop = self.log_pseudo_prior(p_prop)
            with torch.no_grad():
                Risk_prop = self.calc_Risks_tensor(n_sample=self.MCMC_MC_size, distributions=(p_prop,))[0]
            to_init_dirichlet = torch.distributions.dirichlet.Dirichlet(1. * p_prop + 1.)
            ratio = (Risk_prop / Risk_init) * torch.exp(log_pseudo_prior_prop - log_pseudo_prior_init + to_init_dirichlet.log_prob(p_init) - to_prop_dirichlet.log_prob(p_prop))
            if ratio >= 1 or uniform() <= ratio:
                ps[i] = p_prop
                p_init = p_prop
                log_pseudo_prior_init = log_pseudo_prior_prop
                Risk_init = Risk_prop
                i += 1
                if i == n_distr:
                    break
            
            #across dimension jump
            if k_init==1:
                k_prop = 2
            else:
                k_prop = torch.distributions.bernoulli.Bernoulli(torch.tensor(0.5)).sample().item() * 2 - 1 + k_init
            
            if k_prop > k_init:
                flat_dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(int(k_prop - k_init + 1)))
                u = flat_dirichlet.sample()
                p_prop = torch.cat((p_init[:-1], p_init[-1] * u))
                log_pseudo_prior_prop = self.log_pseudo_prior(p_prop)
                with torch.no_grad():
                    Risk_prop = self.calc_Risks_tensor(n_sample=self.MCMC_MC_size, distributions=(p_prop,))[0]
                if k_init==1:
                    ratio = torch.exp(log_pseudo_prior_prop - log_pseudo_prior_init + MCMC_Poisson_distribution.log_prob(torch.tensor(k_init, dtype=default_dtype)).item() - MCMC_Poisson_distribution.log_prob(torch.tensor(k_prop, dtype=default_dtype)).item()  - np.log(2.) - flat_dirichlet.log_prob(u)) * (Risk_prop / Risk_init)
                else:
                    ratio = torch.exp(log_pseudo_prior_prop - log_pseudo_prior_init + MCMC_Poisson_distribution.log_prob(torch.tensor(k_init, dtype=default_dtype)).item() - MCMC_Poisson_distribution.log_prob(torch.tensor(k_prop, dtype=default_dtype)).item() + torch.log(torch.arange(k_init,k_prop)).sum() - flat_dirichlet.log_prob(u)) * (Risk_prop / Risk_init)
                if ratio >= 1 or uniform() <= ratio:
                    ps[i] = p_prop
                    p_init = p_prop
                    k_init = k_prop
                    log_pseudo_prior_init = log_pseudo_prior_prop
                    Risk_init = Risk_prop
                    i += 1
                    if i == n_distr:
                        break
            elif k_prop < k_init:
                flat_dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(int(k_init - k_prop + 1)))
                p_prop = torch.cat((p_init[:int(k_prop - 1)], torch.sum(p_init[int(k_prop-1):]).unsqueeze(0)))
                u = p_init[int(k_prop-1):] / p_prop[-1]
                log_pseudo_prior_prop = self.log_pseudo_prior(p_prop)
                with torch.no_grad():
                    Risk_prop = self.calc_Risks_tensor(n_sample=self.MCMC_MC_size, distributions=(p_prop,))[0]
                ratio = torch.exp(log_pseudo_prior_prop - log_pseudo_prior_init + MCMC_Poisson_distribution.log_prob(torch.tensor(k_init, dtype=default_dtype)).item() - MCMC_Poisson_distribution.log_prob(torch.tensor(k_prop, dtype=default_dtype)).item() + flat_dirichlet.log_prob(u) + torch.log(torch.arange(k_prop,k_init)).sum()) * (Risk_prop / Risk_init)
                if ratio >= 1 or uniform() <= ratio:
                    ps[i] = p_prop
                    p_init = p_prop
                    k_init = k_prop
                    log_pseudo_prior_init = log_pseudo_prior_prop
                    Risk_init = Risk_prop
                    i += 1
                    if i == n_distr:
                        break
        return ps
    
    def SGDmax(self, n_iter=2000, use_init_prior=True, optimizer=None, n_sample=30):
        '''use SGDmax to find the Gamma_l-minimax estimator; updates estimator and Prior_Risk_Constraint_object in place; returns a tuple of "lower bounds" and "upper bounds" (not really because only a small Monte Carlo sample size is used and we update the estimator with one SGD step in each iteration) of the Gamma_l-minimax risk
        n_iter: number of iterations
        use_init_prior: whether to use the prior in Prior_Risk_Constraint_object as the initial point in linear programming
        optimizer: optimizer to update estimator
        n_sample: number of samples drawn for each distribution to estimate Risk.
        b_ub, b_eq: constraints used in linprog that define the restricted set of priors Gamma. default to None. b_eq should include the constraint that prior probabilities sum to 1'''
        risk_lower = []
        risk_upper = []
        if optimizer is None:
            optimizer = torch.optim.SGD(self.estimator.parameters(), lr=0.005)

        np_A_ub, np_A_eq = self.get_constraint_matrices()

        with torch.no_grad():
            Risks = self.calc_Risks_tensor(n_sample=n_sample)
        risk_lower.append(Risks.dot(self.prior_prob).item())
        if use_init_prior:
            linear_prog_result = linprog(-Risks.cpu().numpy(), A_ub=np_A_ub, b_ub=self.b_ub, A_eq=np_A_eq,b_eq=self.b_eq, bounds=(0., 1.), method='revised simplex', x0=self.prior_prob.cpu().numpy())
            if not linear_prog_result.success:
                linear_prog_result = linprog(-Risks.cpu().numpy(), A_ub=np_A_ub,b_ub=self.b_ub, A_eq=np_A_eq, b_eq=self.b_eq, bounds=(0., 1.), method='revised simplex')
        else:
            linear_prog_result = linprog(-Risks.cpu().numpy(), A_ub=np_A_ub,b_ub=self.b_ub, A_eq=np_A_eq, b_eq=self.b_eq, bounds=(0., 1.), method='revised simplex')
        if not linear_prog_result.success:
            raise RuntimeError(linear_prog_result.message)
        self.prior_prob = torch.as_tensor(linear_prog_result.x)
        nonzero_distr_indices = self.prior_prob.nonzero().squeeze(1)
        Risks = self.calc_Risks_tensor(n_sample=n_sample, distr_indices=nonzero_distr_indices)
        risk = Risks.dot(self.prior_prob[nonzero_distr_indices])
        risk_upper.append(risk.item())

        optimizer.zero_grad()
        risk.backward()
        optimizer.step()

        for _ in range(n_iter):
            with torch.no_grad():
                Risks = self.calc_Risks_tensor(n_sample=n_sample)
            risk_lower.append(Risks.dot(self.prior_prob).item())
            linear_prog_result = linprog(-Risks.cpu().numpy(), A_ub=np_A_ub, b_ub=self.b_ub, A_eq=np_A_eq,b_eq=self.b_eq, bounds=(0., 1.), method='revised simplex', x0=self.prior_prob.cpu().numpy())
            if not linear_prog_result.success:
                linear_prog_result = linprog(-Risks.cpu().numpy(), A_ub=np_A_ub, b_ub=self.b_ub, A_eq=np_A_eq, b_eq=self.b_eq, bounds=(0., 1.), method='revised simplex')
            if not linear_prog_result.success:
                raise RuntimeError(linear_prog_result.message)
            self.prior_prob = torch.as_tensor(linear_prog_result.x)
            nonzero_distr_indices = self.prior_prob.nonzero().squeeze(1)
            Risks = self.calc_Risks_tensor(n_sample=n_sample, distr_indices=nonzero_distr_indices)
            risk = Risks.dot(self.prior_prob[nonzero_distr_indices])
            risk_upper.append(risk.item())
            optimizer.zero_grad()
            risk.backward()
            optimizer.step()

        return (np.array(risk_lower), np.array(risk_upper))
    
    def calc_Gamma_minimax_estimator(self, n_SGDmax_iter=200, max_enlarge_iter=10, optimizer=None, n_SGDmax_sample=30, n_accurate_Risk_sample=2000, tol=1e-4, relative_tol=0.02, n_new_distr=1000, save_SGDmax_result=True):
        '''calculate the Gamma-minimax estimator; estimator and prior are updated in place. returns a tuple of (1) status where 0 means success and 1 means divergence (2) list of "lower bounds" of Gamma_l-minimax risk from SGDmax (3) list of "upper bounds" of Gamma_l-minimax risk from SGDmax (4) 2D array of estimated Gamma_l-minimax risks in each iteration (indexed by l) of the old prior (1st column) and new prior (2nd column)
        n_SGDmax_iter: number of iterations in SGDmax (use 20 X n_SGDmax_iter for the first training)
        max_enlarge_iter: max number of iterations to enlarge grid
        optimizer: optimizer to update estimator
        n_SGD_max_sample: number of samples drawn for each distribution to estimate Risk in SGDmax
        n_accurate_Risk_sample: number of samples drawn for each distribution to accurately estimate Risk
        tol, relative_tol: tolerance in increment of [min max risk] to stop enlarging the grid of distributions
        max_enlarge_iter: max number of iteration to enlarge the grid of distributions
        n_new_distr: numbder of new distributions when enlarging the grid'''
        risk_lower = []
        risk_upper = []
        risk_iter = []
        if optimizer is None:
            optimizer = torch.optim.SGD(estimator.parameters(), lr=0.005)

        lower, upper = self.SGDmax(n_iter=n_SGDmax_iter * 20,use_init_prior=False, optimizer=optimizer, n_sample=n_SGDmax_sample)
        if save_SGDmax_result:
            with open("l0.pkl", "wb") as saved_file:
                pickle.dump({"estimator": estimator, "lower": lower, "upper": upper}, saved_file)
        risk_lower.append(lower)
        risk_upper.append(upper)

        for l in range(max_enlarge_iter):
            with torch.no_grad():
                Risks = self.calc_Risks_tensor_memeff(n_sample=n_accurate_Risk_sample)
            np_A_ub, np_A_eq = self.get_constraint_matrices()
            linear_prog_result = linprog(-Risks.cpu().numpy(), A_ub=np_A_ub, b_ub=self.b_ub, A_eq=np_A_eq, b_eq=self.b_eq, bounds=(0., 1.), method='revised simplex', x0=self.prior_prob.cpu().numpy())
            if not linear_prog_result.success:
                linear_prog_result = linprog(-Risks.cpu().numpy(), A_ub=np_A_ub, b_ub=self.b_ub, A_eq=np_A_eq, b_eq=self.b_eq, bounds=(0., 1.), method='revised simplex')
            if not linear_prog_result.success:
                raise RuntimeError(linear_prog_result.message)
            self.prior_prob = torch.as_tensor(linear_prog_result.x)

            old_n_distr = self.n_distr
            old_prior = self.prior_prob.clone()
            self.enlarge_distr_grid(n_distr=n_new_distr)

            with torch.no_grad():
                Risks = self.calc_Risks_tensor_memeff(n_sample=n_accurate_Risk_sample)
            np_A_ub, np_A_eq = self.get_constraint_matrices()
            linear_prog_result = linprog(-Risks.cpu().numpy(), A_ub=np_A_ub, b_ub=self.b_ub, A_eq=np_A_eq, b_eq=self.b_eq, bounds=(0., 1.), method='revised simplex', x0=self.prior_prob.cpu().numpy())
            if not linear_prog_result.success:
                linear_prog_result = linprog(-Risks.cpu().numpy(), A_ub=np_A_ub, b_ub=self.b_ub, A_eq=np_A_eq, b_eq=self.b_eq, bounds=(0., 1.), method='revised simplex')
            if not linear_prog_result.success:
                raise RuntimeError(linear_prog_result.message)
            self.prior_prob = torch.as_tensor(linear_prog_result.x)

            with torch.no_grad():
                eval_Risks = self.calc_Risks_tensor_memeff(n_sample=n_accurate_Risk_sample)
            new_risk = eval_Risks.dot(self.prior_prob).item()
            old_risk = eval_Risks[:old_n_distr].dot(old_prior).item()

            risk_iter.append([old_risk, new_risk])

            if new_risk - old_risk <= tol or (new_risk - old_risk) / new_risk <= relative_tol:
                return (0, risk_lower, risk_upper, np.array(risk_iter))
            
            lower, upper = self.SGDmax(n_iter=n_SGDmax_iter, use_init_prior=True, optimizer=optimizer, n_sample=n_SGDmax_sample)
            if save_SGDmax_result:
                with open("l"+str(l+1)+".pkl", "wb") as saved_file:
                    pickle.dump({"estimator": estimator, "lower": lower, "upper": upper}, saved_file)
            risk_lower.append(lower)
            risk_upper.append(upper)

        return (1, risk_lower, risk_upper, np.array(risk_iter))











def parameter_constraint_fun(ps):
    '''parameters and eq_constraint: expected number of new species in the new sample; no ub_constraint'''
    global sample_size, new_sample_size, prior_credible_range
    output = torch.empty((len(ps), 1))
    for i, p in enumerate(ps):
        one_minus_p = 1 - p
        output[i, 0] = torch.sum(one_minus_p ** sample_size * (1 - one_minus_p ** new_sample_size))
    return (output, torch.cat((-((output >= prior_credible_range[0]) & (output <= prior_credible_range[1])).type(default_dtype), output, -output), 1), None)

def parameter(ps):
    global sample_size, new_sample_size
    output = torch.empty((len(ps), 1))
    for i, p in enumerate(ps):
        one_minus_p = 1 - p
        output[i, 0] = torch.sum(one_minus_p ** sample_size * (1 - one_minus_p ** new_sample_size))
    return output

def Risk_fun(estimates, true_parameters):
    return torch.sum((estimates - true_parameters)**2, dim=2).mean(dim=1)


class OrlitskySureshWu_Estimator(object):
    def __init__(self,sample_size,new_sample_size):
        t = new_sample_size/sample_size
        weights = - (-t) ** np.arange(1, sample_size + 1)
        if t > 1:
            weights *= binom.sf(np.arange(sample_size), np.floor(.5*np.log(sample_size * new_sample_size / (t - 1)) / np.log(3)), 2 / (t + 2))
        self.weights = torch.as_tensor(weights, dtype=default_dtype)
    def __call__(self,samples):
        return torch.sum(samples * self.weights.expand(samples.size()[0], -1), dim=1).unsqueeze(1)

class ShenChaoLin_Estimator(object):
    def __init__(self,sample_size,new_sample_size,k=10):
        if k > sample_size:
            raise ValueError("k must be smaller than sample size")
        self.new_sample_size = new_sample_size
        self.sample_size = sample_size
        self.k = k
    def __call__(self,samples):
        global default_dtype
        global b_eq
        S_rare = torch.sum(samples[:,:self.k], dim=1)
        n_rare = (samples[:, :self.k] @ torch.arange(1, self.k+1, dtype=default_dtype).unsqueeze(1)).squeeze(1)
        C_tilde = 1 - samples[:,0]/n_rare
        C_hat = 1 - samples[:,0]/self.sample_size
        C_tilde = torch.where((C_tilde == 0) | torch.isnan(C_tilde), C_hat, C_tilde)
        C_tilde = torch.where(C_tilde == 0, torch.ones_like(C_tilde) * 0.1, C_tilde)
        gamma2 = S_rare / C_tilde * (samples[:, :self.k] @ (torch.arange(self.k, dtype=default_dtype) * torch.arange(1, self.k+1, dtype=default_dtype)).unsqueeze(1)).squeeze(1) / n_rare / (n_rare - 1.) - 1
        gamma2 = torch.max(gamma2, torch.zeros_like(gamma2))
        w = S_rare / C_tilde + samples[:, 0] / C_tilde * gamma2 - S_rare
        w = torch.where(torch.isnan(w), torch.ones_like(w) * b_eq[0], w)
        estimates = torch.where(w == 0, w, w * (1 - (1 - (1 - C_hat) / w) ** self.new_sample_size))
        return estimates.unsqueeze(1)



class nnet_estimator(torch.nn.Module):
    def __init__(self, naive_estimators, naive_estimators_scale=.02, init_output_params="naive mean"):
        '''naive_estimators_scale: initial estimates are multiplied by naive_estimators_scale before feeding to the neural net to stabilize gradients
        init_output_params: initial parameters, vector of length >=2. First component is intercept; second component is slope of previous hidden layers; other components are the slope of naive estimators. default "naive mean" takes the mean of naive_estimators. Set to None to randomized initialization.'''
        global sample_size
        super().__init__()
        self.naive_estimators = naive_estimators
        self.n_naive_estimators = len(naive_estimators)
        self.naive_estimators_scale = naive_estimators_scale
        self.hidden1 = torch.nn.Linear(sample_size + self.n_naive_estimators, 100, bias=True)
        self.hidden2 = torch.nn.Linear(100 + self.n_naive_estimators, 100, bias=True)
        self.hidden3 = torch.nn.Linear(100 + self.n_naive_estimators, 1, bias=True)
        self.output = torch.nn.Linear(1 + self.n_naive_estimators, 1, bias=True)
        if init_output_params == "naive mean":
            self.output.bias.data = torch.zeros(1)
            self.output.weight.data = torch.cat((torch.zeros(1), torch.ones(self.n_naive_estimators) / self.n_naive_estimators / self.naive_estimators_scale)).unsqueeze(0)
        elif init_output_params is not None:
            self.output.bias.data = torch.tensor(init_output_params[0]).unsqueeze(0)
            self.output.weight.data = torch.tensor(init_output_params[1:]).unsqueeze(0)
    def forward(self, input):
        naive_estimator_node = torch.cat([est(input) for est in self.naive_estimators], dim=1) * self.naive_estimators_scale
        input_aug = torch.cat((input, naive_estimator_node), dim=1)
        hidden1 = torch.relu(self.hidden1(input_aug))
        hidden1_aug = torch.cat((hidden1, naive_estimator_node), dim=1)
        hidden2 = torch.relu(self.hidden2(hidden1_aug))
        hidden2_aug = torch.cat((hidden2, naive_estimator_node), dim=1)
        hidden3 = torch.relu(self.hidden3(hidden2_aug))
        hidden3_aug = torch.cat((hidden3, naive_estimator_node), dim=1)
        return self.output(hidden3_aug)


# hyperparameters
pseudo_prior_mean = 47.5
b_eq = np.array([pseudo_prior_mean])
b_ub = np.array([-.95, 50., -45.])
prior_credible_range = torch.tensor([40., 55.])
MCMC_normal_distribution = torch.distributions.normal.Normal(torch.tensor(pseudo_prior_mean), torch.as_tensor((prior_credible_range[1] - prior_credible_range[0]) * .5 / torch.distributions.normal.Normal(torch.tensor(0.), torch.tensor(1.)).icdf(torch.tensor(.975))))
MCMC_negbinomial_distribution = torch.distributions.negative_binomial.NegativeBinomial(torch.tensor(2.), torch.tensor(.995))
MCMC_Poisson_distribution = torch.distributions.poisson.Poisson(torch.tensor(150.))

# initialize grid
torch.manual_seed(893)
np.random.seed(5784)

estimator = nnet_estimator((OrlitskySureshWu_Estimator(sample_size, new_sample_size), ShenChaoLin_Estimator(sample_size, new_sample_size)))
p_init = torch.cat((torch.ones(100), torch.ones(25) * 2, torch.ones(9) * 3, torch.ones(5) * 4, torch.ones(3) * 5, torch.ones(2) * 6, torch.ones(1) * 7, torch.ones(1) * 9))
p_init /= p_init.sum()
Gamma_minimax_Problem_object=Gamma_minimax_Problem(estimator=estimator, parameter_constraint_fun=parameter_constraint_fun, b_ub=b_ub, Risk_fun=Risk_fun, parameter=parameter, p_init=p_init)

# compute Gamma-minimax estimator
result = Gamma_minimax_Problem_object.calc_Gamma_minimax_estimator()

# save Gamma-minimax estimator
import pickle
with open("minimax_estimator.pkl", "wb") as saved_file:
    pickle.dump({"Gamma_minimax_Problem_object":Gamma_minimax_Problem_object, "result":result}, saved_file)

# load Gamma-minimax estimator
import pickle
with open("minimax_estimator.pkl", "rb") as saved_file:
    results = pickle.load(saved_file)
result = results["result"]
Gamma_minimax_Problem_object = results["Gamma_minimax_Problem_object"]
estimator = Gamma_minimax_Problem_object.estimator

result

plt.plot(result[1][0])
plt.plot(result[2][0])

list(estimator.named_parameters())

#resample simulation to estimate worst-case Bayes risks and Risks
torch.manual_seed(5678934)
np.random.seed(2758)
data = torch.cat((torch.ones(61) * 1, torch.ones(35) * 2, torch.ones(18) * 3, torch.ones(12) * 4, torch.ones(15) * 5, torch.ones(4) * 6, torch.ones(8) * 7, torch.ones(4) * 8, torch.ones(5) * 9, torch.ones(5) * 10, torch.ones(1) * 11, torch.ones(2) * 12, torch.ones(1) * 13, torch.ones(2) * 14, torch.ones(3) * 15, torch.ones(2) * 16, torch.ones(1) * 19, torch.ones(2) * 20, torch.ones(1) * 22, torch.ones(1) * 29, torch.ones(1) * 32, torch.ones(1) * 40, torch.ones(1) * 43, torch.ones(1) * 48, torch.ones(1) * 67))
p = data / data.sum()
simulation_object = Gamma_minimax_Problem(estimator=estimator, parameter_constraint_fun=parameter_constraint_fun, b_ub=b_ub, p_init=p, n_distr=1)

Gamma_minimax_Risk = simulation_object.calc_Risks_tensor(n_sample=20000, distributions=(p,)).item()
OrlitskySureshWu_Risk = simulation_object.calc_Risks_tensor(n_sample=20000, estimator=OrlitskySureshWu_Estimator(sample_size, new_sample_size), distributions=(p,)).item()
ShenChaoLin_Risk = simulation_object.calc_Risks_tensor(n_sample=20000, estimator=ShenChaoLin_Estimator(sample_size, new_sample_size), distributions=(p,)).item()
