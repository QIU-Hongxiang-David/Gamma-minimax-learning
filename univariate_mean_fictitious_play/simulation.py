import torch
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# All inputs that can be array-like or a floating number (i.e. except counts and bools) should be tensors
# All user defined function inputs and outputs should be tensors

torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_tensor_type(torch.cuda.DoubleTensor)


# functions and classes for estimators and Gamma-minimax estimators
def init_generate_distr(support, n_random_distr, new_support_index=None):
    '''support: a 1D tensor with support points
    n_random_distr: number of random distributions to be generated (besides point masses at support points)
    new_support_index: index of support points for which point masses to be generated; set to [] if no new points'''
    if new_support_index is None:
        new_support_index = torch.arange(len(support))
    output = torch.empty((len(new_support_index) + n_random_distr, len(support)))
    output[:len(new_support_index), :] = 0
    for i, index in enumerate(new_support_index):
        output[i, index] = 1

    alpha =torch.ones(len(support))
    output[len(new_support_index):, :] = torch.distributions.dirichlet.Dirichlet(alpha).sample((n_random_distr,))
    return output


class Distrs(object):
    def __init__(self, support, n_random_distr, new_support_index=None):
        '''support: a 1D tensor with support points
        n_random_distr: number of random distributions to be generated (besides point masses at support points)
        new_support_index: index of support points for which point masses to be generated; set to [] if no new points'''
        self.support = support
        self.n_random_distr = n_random_distr
        if new_support_index is None:
            self.new_support_index = torch.arange(support.size()[0])
        else:
            self.new_support_index = new_support_index
        self.distributions = init_generate_distr(self.support, self.n_random_distr, self.new_support_index)
        self.n_distr = self.distributions.size()[0]
        self.true_vars = self.true_parameters = self.A_ubT = self.A_eqT = self.Risks = None

    def generate_distr(self, n_random_distr):
        '''n_random_distr: number of random distributions to be generated (besides point masses at support points)'''
        alpha = torch.ones(len(self.support))
        new_distributions = torch.distributions.dirichlet.Dirichlet(alpha).sample((n_random_distr,))
        self.distributions = torch.cat((self.distributions, new_distributions), dim=0)
        self.n_distr += n_random_distr
        self.n_random_distr += n_random_distr


    def eval_parameter_constraint(self, parameter, ub_constraint_fun=None, eq_constraint_fun=None, sample_size=10):
        '''parameter: true mean. function that takes in a tensor of support and a tensor of probabilities (number of distributions X number of support points) and returns a 2D tensor with each entry in the 0th dimension corresponding to each probability distribution
        ub_constraint_fun,eq_contraint_fun: either None or functions that takes in a tensor of support and a tensor of probabilities and returns a 2D tensor with each entry in the 0th dimension corresponding to each probability distribution, which is used in A_ub and A_eq respectively. defaults to no constraint, i.e. None'''
        self.true_parameters = parameter(self.support, self.distributions)
        self.true_vars = self.distributions.mm(self.support.unsqueeze(1) ** 2) - self.true_parameters ** 2
        
        #"X^T X" and "X^T Y" for each distribution in least squares to calculate the optimal estimator
        self.XTX = torch.cat((torch.ones((self.n_distr, 1)), self.true_parameters, self.true_parameters, self.true_parameters ** 2 + self.true_vars / sample_size), dim=1).reshape(self.n_distr, 2, 2)
        self.XTY = torch.cat((self.true_parameters, self.true_parameters ** 2), dim=1).reshape(self.n_distr, 2, 1)

        if ub_constraint_fun is None:
            self.A_ubT = None
        elif ub_constraint_fun is parameter:
            self.A_ubT = self.true_parameters
        else:
            self.A_ubT = ub_constraint_fun(self.support, self.distributions)

        if eq_constraint_fun is None:
            self.A_eqT = None
        elif eq_constraint_fun is parameter:
            self.A_eqT = self.true_parameters
        else:
            self.A_eqT = eq_constraint_fun(self.support, self.distributions)

    def add_parameter_constraint(self, n_added, parameter, ub_constraint_fun=None, eq_constraint_fun=None, sample_size=10):
        self.true_parameters = torch.cat((self.true_parameters, parameter(self.support, self.distributions[-n_added:])), dim=0)
        self.true_vars = torch.cat((self.true_vars, self.distributions[-n_added:].mm(self.support.unsqueeze(1) ** 2) - self.true_parameters[-n_added:] ** 2), dim=0)

        #"X^T X" and "X^T Y" for each distribution in least squares to calculate the optimal estimator
        self.XTX = torch.cat((torch.ones((self.n_distr, 1)), self.true_parameters, self.true_parameters, self.true_parameters ** 2 + self.true_vars / sample_size), dim=1).reshape(self.n_distr, 2, 2)
        self.XTY = torch.cat((self.true_parameters, self.true_parameters ** 2), dim=1).reshape(self.n_distr, 2, 1)

        if ub_constraint_fun is None:
            pass
        elif ub_constraint_fun is parameter:
            self.A_ubT = self.true_parameters
        else:
            self.A_ubT = torch.cat((self.A_ubT, ub_constraint_fun(self.support, self.distributions[-n_added:])), dim=0)

        if eq_constraint_fun is None:
            pass
        elif eq_constraint_fun is parameter:
            self.A_eqT = self.true_parameters
        else:
            self.A_eqT = torch.cat(self.A_eqT, eq_constraint_fun(self.support, self.distributions[-n_added:]), dim=0)

    def eval_Risks(self, estimators_coef, sample_size=10):
        '''evaluate (via Monte Carlo) Risks at each distribution after true_parameters and constraints are computed
        estimators: a list of models. each model takes in a tensor of samples and outputs a tensor of estimators
        estimators_coef: a 2D tensor (# of estimators X 2) of coefficients of estimators.'''
        if self.true_parameters is None or self.true_vars is None:
            raise AttributeError("true_parameters and true variances not computed. Run eval_parameter_constraint() method first!")

        self.Risks = (self.true_vars / sample_size * estimators_coef[:, 1] ** 2 + (estimators_coef[:, 0] - (1 - estimators_coef[:, 1]) * self.true_parameters) ** 2).mean(dim=1)


class Prior_Risk_Constraint(object):
    '''holds an array of Distrs objects and their prior probs, their Risks evaluated by Monte Carlo, and the summaries of distributions appearing in the constraints.'''

    def __init__(self, support, parameter, ub_constraint_fun=None, eq_constraint_fun=None,\
        Risk_fun=lambda estimates, true_parameters: torch.sum((estimates - true_parameters)**2, dim=2).mean(dim=1),\
            n_random_distr=2000, init_prior_prob=None):
        '''support,n_random_distr: see Distrs class
        init_prior_prob: initial probs for the distributions. if not None, its length should be [# of support points]+[n_random_distr]. defaults to a point mass at the first support point
        parameter: parameter of interest. function that takes in a tensor of support and a tensor of probabilities and returns a 2D tensor with each entry in the 0th dimension corresponding to each probability distribution
        Risk_fun: function that takes in a 3D tensor of estimates (number of distributions X number of samples X dimension of estimate) and a 3D tensor of true parameters (number of distributions X number of samples X dimension of estimate) and outputs the Risk=average loss for each distribution. output should be a 1D tensor. default to squared-error loss
        ub_constraint_fun,eq_contraint_fun: either None or functions that takes in a tensor of support and a tensor of probabilities and returns a 2D tensor with each entry in the 0th dimension corresponding to each probability distribution, which is used in A_ub and A_eq respectively. defaults to no constraint, i.e. None
        distrs is the array containing Distrs objects
        Distrs_n_distr is an array containing n_distr of Distr objects of distrs
        Distrs_indices is an array containing indices of the first distribution of each Distrs object in the vector with each entry corresponding to a distribution'''
        self.distrs = np.array(Distrs(support, n_random_distr), ndmin=1)

        if init_prior_prob is None:
            self.prior_prob = torch.zeros(self.distrs[0].n_distr)
            self.prior_prob[0] = 1
        else:
            self.prior_prob = torch.as_tensor(init_prior_prob)
        self.Distrs_n_distr = torch.tensor((self.distrs[0].n_distr,))
        self.Distrs_indices = torch.tensor((0,))

        self.parameter = parameter
        self.ub_constraint_fun = ub_constraint_fun
        self.eq_constraint_fun = eq_constraint_fun

        self.distrs[0].eval_parameter_constraint(self.parameter, self.ub_constraint_fun, self.eq_constraint_fun)
        self.Risk_fun = Risk_fun

    def enlarge_distr_grid(self, n_new_distr_old_support=500, n_new_support_points=10, n_random_new_distr_new_support=500):
        '''generate a finer grid by generating more distributions on existing Distrs objects and a new Distrs object with more support points; update distrs, true_parameters, A_ubT, A_eqT, Distrs_n_distr and Distrs_indices accordingly; update prior_prob by putting zero probabilities to new distributions
        n_new_distr_old_support: number of new distributions on the old support
        n_new_support_points: number of new support points
        n_random_new_distr_new_support: number of new distributions on the new support'''
        new_prior_list = []
        for i, distr in enumerate(self.distrs):
            new_prior_list.extend([self.prior_prob[self.Distrs_indices[i] : self.Distrs_indices[i] + distr.n_distr], torch.zeros(n_new_distr_old_support)])
            distr.generate_distr(n_new_distr_old_support)
            distr.add_parameter_constraint(n_added=n_new_distr_old_support, parameter=self.parameter, ub_constraint_fun=self.ub_constraint_fun, eq_constraint_fun=self.eq_constraint_fun)
        self.Distrs_n_distr += n_new_distr_old_support

        old_support = self.distrs[-1].support
        new_support = torch.cat((old_support, torch.distributions.uniform.Uniform(0,1).sample((n_new_support_points,))), dim=0)
        new_support_index = torch.arange(len(old_support), len(new_support))
        new_Distr_object = Distrs(new_support, n_random_new_distr_new_support, new_support_index)
        new_Distr_object.eval_parameter_constraint(parameter=self.parameter, ub_constraint_fun=self.ub_constraint_fun, eq_constraint_fun=self.eq_constraint_fun)

        self.distrs = np.concatenate((self.distrs, np.array(new_Distr_object, ndmin=1)))
        self.Distrs_n_distr = torch.cat((self.Distrs_n_distr, torch.tensor((new_Distr_object.n_distr,))))
        new_prior_list.append(torch.zeros(new_Distr_object.n_distr))
        self.prior_prob = torch.cat(new_prior_list, dim = 0)
        self.Distrs_indices = torch.cat((torch.tensor((0,)), torch.cumsum(self.Distrs_n_distr[:-1], dim=0)))

    def get_constraint_matrices(self):
        '''aggregate constraint matrices from Distrs objects and return a tuple of (1) 2D numpy array A_ub (2) 2D numpy array A_eq
        A_eq stacks the constraint that prior probability sums to 1 at the end'''
        np_A_ub = []
        np_A_eq = []
        for distr in self.distrs:
            np_A_ub.append(distr.A_ubT)
            np_A_eq.append(distr.A_eqT)

        np_A_ub=torch.cat(np_A_ub, dim=0).cpu().numpy().transpose() if np_A_ub[0] is not None else None

        if np_A_eq[0] is not None:
            np_A_eq = torch.cat(np_A_eq, dim=0).cpu().numpy().transpose()
            np_A_eq = np.vstack((np_A_eq, np.ones((1, self.prior_prob.size()[0]))))
        else:
            np_A_eq = np.ones((1, self.prior_prob.size()[0]))

        return np_A_ub, np_A_eq
    
    def get_XTX_XTY(self):
        '''aggregate "X^T X" and "X^T Y" matrices from Distrs objects and return a tuple of (1) 3D tensor of "X^T X" (2) 3D tensor of "X^T Y"'''
        XTX = []
        XTY = []
        for distr in self.distrs:
            XTX.append(distr.XTX)
            XTY.append(distr.XTY)
        
        XTX = torch.cat(XTX, dim=0)
        XTY = torch.cat(XTY, dim=0)
        return XTX, XTY

    def calc_Risks_tensor(self, estimators):
        '''calculate via Monte Carlo and aggregate Risks into a 1D tensor'''
        Risks = []
        for distr in self.distrs:
            distr.eval_Risks(estimators)
            Risks.append(distr.Risks)
        return torch.cat(Risks, dim=0)
    


def fictitious_play(estimators_coef, Prior_Risk_Constraint_object, n_max_iter=10000, tol=1e-4, relative_tol=0.01, use_init_prior=True, b_ub=None, b_eq=None):
    '''use fictitious play to find the Gamma_l-minimax estimator; updates Prior_Risk_Constraint_object in place; returns a tuple of updated estimators_coef, success status (0 for success; 1 for failure), lower bounds and upper bounds of the Gamma_l-minimax risk
    estimators_coef: a 2D tensor (# estimators X 2) of coefficients of estimators such that the empirical distribution of this list is the initial stochastic estimator
    Prior_Risk_Constraint_object: Prior_Risk_Constraint object that contains information about Gamma_l
    n_max_iter: max number of iterations
    tol, relative_tol: tolerance in difference between upper and lower bounds of minimax risk to stop fictitious play iteration
    use_init_prior: whether to use the prior in Prior_Risk_Constraint_object as the initial point in linear programming
    b_ub, b_eq: constraints used in linprog that define the restricted set of priors Gamma. default to None. b_eq should include the constraint that prior probabilities sum to 1'''
    risk_lower = []
    risk_upper = []

    np_A_ub, np_A_eq = Prior_Risk_Constraint_object.get_constraint_matrices()
    XTX, XTY = Prior_Risk_Constraint_object.get_XTX_XTY()
    
    Risks = Prior_Risk_Constraint_object.calc_Risks_tensor(estimators_coef)
    risk_lower.append(Risks.dot(Prior_Risk_Constraint_object.prior_prob).item())
    if use_init_prior:
        linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex', x0=Prior_Risk_Constraint_object.prior_prob.cpu().numpy())
        if not linear_prog_result.success:
            linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex')
    else:
        linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex')
    if not linear_prog_result.success:
        raise RuntimeError(linear_prog_result.message)
    t_iter = 1
    init_prior = torch.as_tensor(linear_prog_result.x)
    Prior_Risk_Constraint_object.prior_prob = init_prior

    risk_upper.append(Risks.dot(init_prior).item())

    for _ in range(n_max_iter):
        reshaped_prior_prob = Prior_Risk_Constraint_object.prior_prob.reshape(-1, 1, 1)
        prior_mean_XTX = (reshaped_prior_prob * XTX).sum(dim=0)
        prior_mean_XTY = (reshaped_prior_prob * XTY).sum(dim=0)
        new_estimator_coef = torch.solve(prior_mean_XTY, prior_mean_XTX)[0].squeeze(1)
        estimators_coef = torch.cat((estimators_coef, new_estimator_coef.unsqueeze(0)), dim=0)

        new_Risks = Prior_Risk_Constraint_object.calc_Risks_tensor(new_estimator_coef.unsqueeze(0))
        n_estimators = estimators_coef.size()[0]
        Risks = (n_estimators - 1.) / n_estimators * Risks + new_Risks / n_estimators
        risk_lower.append(new_Risks.dot(Prior_Risk_Constraint_object.prior_prob).item())
        
        linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex', x0=init_prior.cpu().numpy())
        if not linear_prog_result.success:
            linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex')
        if not linear_prog_result.success:
            raise RuntimeError(linear_prog_result.message)
        t_iter += 1
        init_prior = torch.as_tensor(linear_prog_result.x)
        Prior_Risk_Constraint_object.prior_prob = (t_iter - 1.) / t_iter * Prior_Risk_Constraint_object.prior_prob + init_prior / t_iter
        risk_upper.append(Risks.dot(init_prior).item())
        
        if risk_upper[-1] - risk_lower[-1] <= tol or (risk_upper[-1] - risk_lower[-1]) / risk_upper[-1] <= relative_tol:
            return (estimators_coef, 0, np.array(risk_lower), np.array(risk_upper))
        
        reshaped_prior_prob = Prior_Risk_Constraint_object.prior_prob.reshape(-1, 1, 1)
        prior_mean_XTX = (reshaped_prior_prob * XTX).sum(dim=0)
        prior_mean_XTY = (reshaped_prior_prob * XTY).sum(dim=0)
        new_estimator_coef = torch.solve(prior_mean_XTY, prior_mean_XTX)[0].squeeze(1)
        estimators_coef = torch.cat((estimators_coef, new_estimator_coef.unsqueeze(0)), dim=0)
    
    return (estimators_coef, 1, np.array(risk_lower), np.array(risk_upper))


def calc_Gamma_minimax_estimator(estimator_coef, Prior_Risk_Constraint_object, b_ub=None, b_eq=None, n_fictitious_play_max_iter=10000, max_enlarge_iter=10, tol=0.0005, fictitious_play_tol=1e-4, fictitious_play_relative_tol=0.02, n_new_distr_old_support=1000, n_new_support_points=10, n_random_new_distr_new_support=500):
    '''calculate the Gamma-minimax estimator starting from initial estimator and Prior_Risk_Constraint_object; Prior_Risk_Constraint_object is updated in place. returns a tuple of (1) Gamma-minimax estimator (2) status where 0 means success and 1 means divergence (this does not account for success status of fictitious play) (3) list of "lower bounds" of Gamma_l-minimax risk from SGDmax (4) list of "upper bounds" of Gamma_l-minimax risk from SGDmax (5) success status of fictitious play iteration (6) 2D array of estimated Gamma_l-minimax risks in each iteration (indexed by l) of the old prior (1st column) and new prior (2nd column)
    estimator_coef: coefficients of the initial estimator. a 1D tensor of length 2
    Prior_Risk_Constraint_object: Prior_Risk_Constraint object
    n_fictitious_play_max_iter: max number of iterations in fictitious play
    use_init_prior: whether to use the prior in Prior_Risk_Constraint_object as the initial point in linear programming
    tol: tolerance in increment of [min max risk] to stop enlarging the grid of distributions
    fictitious_play_tol, fictitious_play_relative_tol: tolerance for fictitious play
    max_enlarge_iter: max number of iteration to enlarge the grid of distributions
    b_ub, b_eq: constraints used in linprog that define the restricted set of priors Gamma. default to None
    n_new_distr_old_support, n_new_support_points, n_random_new_distr_new_support: see Prior_Risk_Constraint.enlarge_distr_grid'''
    risk_lower = []
    risk_upper = []
    fictitious_play_success = []
    risk_iter = []
    b_eq = np.concatenate((b_eq, np.ones(1))) if b_eq is not None else np.ones(1)

    estimators_coef = estimator_coef.unsqueeze(0)
    estimators_coef, success, lower, upper = fictitious_play(estimators_coef, Prior_Risk_Constraint_object, n_max_iter=n_fictitious_play_max_iter, use_init_prior=False, b_ub=b_ub, b_eq=b_eq, tol=fictitious_play_tol, relative_tol=fictitious_play_relative_tol)
    estimator_coef = estimators_coef.mean(dim=0)
    risk_lower.append(lower)
    risk_upper.append(upper)
    fictitious_play_success.append(success)

    for l in range(max_enlarge_iter):
        Prior_Risk_Constraint_object.enlarge_distr_grid(n_new_distr_old_support=n_new_distr_old_support, n_new_support_points=n_new_support_points, n_random_new_distr_new_support=n_random_new_distr_new_support)
        old_prior = Prior_Risk_Constraint_object.prior_prob.clone()
        
        Risks = Prior_Risk_Constraint_object.calc_Risks_tensor(estimator_coef.unsqueeze(0))
        np_A_ub, np_A_eq = Prior_Risk_Constraint_object.get_constraint_matrices()

        linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex', x0=Prior_Risk_Constraint_object.prior_prob.cpu().numpy())
        if not linear_prog_result.success:
            linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex')
        if not linear_prog_result.success:
            raise RuntimeError(linear_prog_result.message)
        Prior_Risk_Constraint_object.prior_prob = torch.as_tensor(linear_prog_result.x)

        new_risk = Risks.dot(Prior_Risk_Constraint_object.prior_prob)
        old_risk = Risks.dot(old_prior)
        risk_iter.append([old_risk,new_risk])
        
        if new_risk - old_risk <= tol:
            return (estimator_coef, 0, risk_lower, risk_upper, fictitious_play_success, np.array(risk_iter))
        else:
            estimators_coef = estimator_coef.unsqueeze(0)
            estimators_coef, success, lower, upper = fictitious_play(estimators_coef, Prior_Risk_Constraint_object, n_max_iter=n_fictitious_play_max_iter, use_init_prior=True, b_ub=b_ub, b_eq=b_eq, tol=fictitious_play_tol, relative_tol=fictitious_play_relative_tol)
            estimator_coef = estimators_coef.mean(dim=0)
            risk_lower.append(lower)
            risk_upper.append(upper)
            fictitious_play_success.append(success)
    
    return (estimator_coef, 1, risk_lower, risk_upper, fictitious_play_success, np.array(risk_iter))














def estimate(samples, estimator_coef):
    '''given samples, which is a 2D tensor (# samples X sample size), compute the estimates according to the coefficient. the output is also a 2D tensor (# samples X 1)'''
    return (estimator_coef[0] + samples.mean(dim=1) * estimator_coef[1]).unsqueeze(1)

def parameter(support, distribution):
    '''mean'''
    return distribution.mm(support.unsqueeze(1))

def Risk_fun(estimates, true_parameters):
    return torch.sum((estimates - true_parameters)**2, dim=2).mean(dim=1)

# hyperparameter
b_eq = np.array([0.3])


# initialize
torch.manual_seed(893)
np.random.seed(5784)

estimator_coef = torch.as_tensor((0., 1.))
Prior_Risk_Constraint_object = Prior_Risk_Constraint(support=torch.tensor([0., 1.]), parameter=parameter, eq_constraint_fun=parameter, Risk_fun=Risk_fun)

# compute Gamma-minimax estimator
result = calc_Gamma_minimax_estimator(estimator_coef, Prior_Risk_Constraint_object, b_eq=b_eq)

estimator_coef = result[0]

# save Gamma-minimax estimator
import pickle
with open("minimax_estimator.pkl", "wb") as saved_file:
    pickle.dump({"result": result, "Prior_Risk_Constraint_object": Prior_Risk_Constraint_object}, saved_file)

# load Gamma-minimax estimator
import pickle
with open("minimax_estimator.pkl", "rb") as saved_file:
    results = pickle.load(saved_file)
result = results["result"]
estimator_coef = result[0]
Prior_Risk_Constraint_object = results["Prior_Risk_Constraint_object"]

# theoretical Gamma-minimax estimator
mu = b_eq[0]
n = 10
theoretical_coef = torch.as_tensor([mu/(1+np.sqrt(n)),np.sqrt(n)/(1+np.sqrt(n))])
print(theoretical_coef)
print(estimator_coef)

# simulation to estimate worst-case Bayes risks
torch.manual_seed(893)
np.random.seed(5784)
Beta_prior = Prior_Risk_Constraint(support=torch.tensor([0., 1.]), parameter=parameter, eq_constraint_fun=parameter, Risk_fun=Risk_fun)
Beta_prior.distrs[0].distributions = torch.distributions.dirichlet.Dirichlet(torch.tensor([(1 - mu) * np.sqrt(n), mu * np.sqrt(n)])).sample((1000,))
Beta_prior.distrs[0].n_distr = 1000
Beta_prior.prior_prob = torch.ones(1000)/1000
Beta_prior.Distrs_n_distr = 1000
Beta_prior.distrs[0].eval_parameter_constraint(parameter, eq_constraint_fun=parameter)
risk = Beta_prior.calc_Risks_tensor(estimator_coef.unsqueeze(0)).dot(Beta_prior.prior_prob)
theoretical_risk = Beta_prior.calc_Risks_tensor(theoretical_coef.unsqueeze(0)).dot(Beta_prior.prior_prob)

Risks = Beta_prior.calc_Risks_tensor(estimator_coef.unsqueeze(0))
dummy, np_A_eq = Beta_prior.get_constraint_matrices()
linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_eq=np_A_eq, b_eq=np.concatenate((b_eq, np.ones(1))), bounds=(0., 1.), method='revised simplex')
Beta_prior.prior_prob = torch.as_tensor(linear_prog_result.x)
risk = Risks.dot(Beta_prior.prior_prob)

Risks = Beta_prior.calc_Risks_tensor(theoretical_coef.unsqueeze(0))
linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_eq=np_A_eq, b_eq=np.concatenate((b_eq, np.ones(1))), bounds=(0., 1.), method='revised simplex')
Beta_prior.prior_prob = torch.as_tensor(linear_prog_result.x)
theoretical_risk = Risks.dot(Beta_prior.prior_prob)

Prior_Risk_Constraint_object.calc_Risks_tensor(estimator_coef.unsqueeze(0)).dot(Prior_Risk_Constraint_object.prior_prob)
Prior_Risk_Constraint_object.calc_Risks_tensor(theoretical_coef.unsqueeze(0)).dot(Prior_Risk_Constraint_object.prior_prob)

result
