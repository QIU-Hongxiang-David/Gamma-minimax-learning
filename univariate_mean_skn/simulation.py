import torch
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import pickle

# All inputs that can be array-like or a floating number (i.e. except counts and bools) should be tensors
# All user defined function inputs and outputs should be tensors

torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_tensor_type(torch.cuda.DoubleTensor)

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
        self.distributions = init_generate_distr(
            self.support, self.n_random_distr, self.new_support_index)
        self.n_distr = self.distributions.size()[0]
        self.samples = self.true_parameters = self.A_ubT = self.A_eqT = self.Risks = None

    def generate_distr(self, n_random_distr):
        '''n_random_distr: number of random distributions to be generated (besides point masses at support points)'''
        alpha = torch.ones(len(self.support))
        new_distributions = torch.distributions.dirichlet.Dirichlet(alpha).sample((n_random_distr,))
        self.distributions = torch.cat((self.distributions, new_distributions), dim=0)
        self.n_distr += n_random_distr
        self.n_random_distr += n_random_distr

    def draw_sample(self, n_sample, sample_size=10):
        "sample_size: size of each sample; n_sample: number of samples per distribution"
        self.samples = torch.empty([self.n_distr, n_sample, sample_size])
        for i, distribution in enumerate(self.distributions):
            self.samples[i, :, :] = self.support[torch.distributions.categorical.Categorical(distribution).sample((sample_size * n_sample,))].reshape(n_sample, sample_size)

    def eval_parameter_constraint(self, parameter, ub_constraint_fun=None, eq_constraint_fun=None):
        '''parameter: parameter of interest. function that takes in a tensor of support and a tensor of probabilities (number of distributions X number of support points) and returns a 2D tensor with each entry in the 0th dimension corresponding to each probability distribution
        ub_constraint_fun,eq_contraint_fun: either None or functions that takes in a tensor of support and a tensor of probabilities and returns a 2D tensor with each entry in the 0th dimension corresponding to each probability distribution, which is used in A_ub and A_eq respectively. defaults to no constraint, i.e. None'''
        self.true_parameters = parameter(self.support, self.distributions)

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

    def add_parameter_constraint(self, n_added, parameter, ub_constraint_fun=None, eq_constraint_fun=None):
        self.true_parameters = torch.cat((self.true_parameters,
                                        parameter(self.support, self.distributions[-n_added:])), dim=0)

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
            self.A_eqT = torch.cat(self.A_eqT,
                                    eq_constraint_fun(self.support, self.distributions[-n_added:]), dim=0)

    def eval_Risks(self, estimator, n_sample=50,\
        Risk_fun=lambda estimates, true_parameters: torch.sum((estimates - true_parameters)**2, dim=2).mean(dim=1)):
        '''evaluate (via Monte Carlo) Risks at each distribution after true_parameters and constraints are computed
        estimator: a model that takes in a tensor of samples and outputs a tensor of estimators
        n_sample: number of new samples drawn to evaluate Risks. use a small number in SGD-type algorithms returns a tensor of Risks
        Risk_fun: function that takes in a 3D tensor of estimates (number of distributions X number of samples X dimension of estimate) and a 3D tensor of true parameters (number of distributions X number of samples X dimension of estimate) and outputs the Risk=average loss for each distribution. output should be a 1D tensor. default to squared-error loss'''
        if self.true_parameters is None:
            raise AttributeError("true_parameters not computed. Run eval_parameter_constraint() method first!")
        
        self.draw_sample(n_sample=n_sample)
        
        true_parameters = self.true_parameters.unsqueeze(1).expand(-1, n_sample, -1)
        estimates = estimator(self.samples.reshape(self.n_distr*n_sample, -1)).view(self.n_distr, n_sample, -1)
        self.Risks = Risk_fun(estimates, true_parameters)
        self.samples = None
    
    def eval_Risks_memeff(self, estimator, n_sample=1000,\
        Risk_fun=lambda estimates, true_parameters: torch.sum((estimates - true_parameters)**2, dim=2).mean(dim=1), sample_size=10):
        '''memory efficient version of eval_Risks; might not work with autograd'''
        if self.true_parameters is None:
            raise AttributeError("true_parameters not computed. Run eval_parameter_constraint() method first!")

        self.Risks = torch.empty(self.n_distr)
        for i, distribution in enumerate(self.distributions):
            samples = self.support[torch.distributions.categorical.Categorical(distribution).sample((sample_size * n_sample,))].reshape(n_sample, sample_size)
            estimates = estimator(samples)
            self.Risks[i] = Risk_fun(estimates.unsqueeze(0), self.true_parameters[i])


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
        '''aggregate constraint matrices from Distrs objects and return a tuple of (1) 1D numpy array of Risks (2) 2D numpy array A_ub (3) 2D numpy array A_eq
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

    def calc_Risks_tensor(self, estimator, n_sample=50):
        '''calculate via Monte Carlo and aggregate Risks into a 1D tensor'''
        Risks = []
        for distr in self.distrs:
            distr.eval_Risks(estimator, n_sample=n_sample, Risk_fun=self.Risk_fun)
            Risks.append(distr.Risks)
        return torch.cat(Risks, dim=0)
    
    def calc_Risks_tensor_memeff(self, estimator, n_sample=50):
        '''memory efficient version of calc_Risks_tensor; might not work with autograd'''
        Risks = []
        for distr in self.distrs:
            distr.eval_Risks_memeff(estimator, n_sample=n_sample, Risk_fun=self.Risk_fun)
            Risks.append(distr.Risks)
        return torch.cat(Risks, dim=0)





def SGDmax(estimator, Prior_Risk_Constraint_object, n_iter=1500,\
    use_init_prior=True, optimizer=None, n_sample=50, b_ub=None, b_eq=None):
    '''use SGDmax to find the Gamma_l-minimax estimator; updates estimator and Prior_Risk_Constraint_object in place; returns a tuple of "lower bounds" and "upper bounds" (not really because only a small Monte Carlo sample size is used and we update the estimator with one SGD step in each iteration) of the Gamma_l-minimax risk
    estimator: a Pytorch model
    Prior_Risk_Constraint_object: Prior_Risk_Constraint object that contains information about Gamma_l
    n_iter: number of iterations
    use_init_prior: whether to use the prior in Prior_Risk_Constraint_object as the initial point in linear programming
    optimizer: optimizer to update estimator
    n_sample: number of samples drawn for each distribution to estimate Risk.
    b_ub, b_eq: constraints used in linprog that define the restricted set of priors Gamma. default to None. b_eq should include the constraint that prior probabilities sum to 1'''
    risk_lower = []
    risk_upper = []
    if optimizer is None:
        optimizer = torch.optim.SGD(estimator.parameters(), lr=0.005)

    np_A_ub, np_A_eq = Prior_Risk_Constraint_object.get_constraint_matrices()

    with torch.no_grad():
        Risks = Prior_Risk_Constraint_object.calc_Risks_tensor(estimator, n_sample=n_sample)
    risk_lower.append(Risks.dot(Prior_Risk_Constraint_object.prior_prob).item())
    if use_init_prior:
        linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex', x0=Prior_Risk_Constraint_object.prior_prob.cpu().numpy())
        if not linear_prog_result.success:
            linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex')
    else:
        linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex')
    Prior_Risk_Constraint_object.prior_prob = torch.as_tensor(linear_prog_result.x)
    Risks = Prior_Risk_Constraint_object.calc_Risks_tensor(estimator, n_sample=n_sample)
    risk = Risks.dot(Prior_Risk_Constraint_object.prior_prob)
    risk_upper.append(risk.item())
    
    optimizer.zero_grad()
    risk.backward()
    optimizer.step()

    for _ in range(n_iter):
        with torch.no_grad():
            Risks = Prior_Risk_Constraint_object.calc_Risks_tensor(estimator, n_sample = n_sample)
        risk_lower.append(Risks.dot(Prior_Risk_Constraint_object.prior_prob).item())
        linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex', x0=Prior_Risk_Constraint_object.prior_prob.cpu().numpy())
        if not linear_prog_result.success:
            linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex')
        Prior_Risk_Constraint_object.prior_prob = torch.as_tensor(linear_prog_result.x)
        Risks = Prior_Risk_Constraint_object.calc_Risks_tensor(estimator, n_sample=n_sample)
        risk = Risks.dot(Prior_Risk_Constraint_object.prior_prob)
        risk_upper.append(risk.item())
        optimizer.zero_grad()
        risk.backward()
        optimizer.step()
    
    return (np.array(risk_lower), np.array(risk_upper))



def calc_Gamma_minimax_estimator(estimator, Prior_Risk_Constraint_object, b_ub=None, b_eq=None,\
    n_SGDmax_iter=200, max_enlarge_iter=5, optimizer=None, n_SGDmax_sample=50, n_accurate_Risk_sample=2000, tol=1e-4,\
        n_new_distr_old_support=1000, n_new_support_points=10, n_random_new_distr_new_support=500, save_SGDmax_result=True):
    '''calculate the Gamma-minimax estimator starting from initial estimator and Prior_Risk_Constraint_object; these two arguments are updated in place. returns a tuple of (1) status where 0 means success and 1 means divergence (2) list of "lower bounds" of Gamma_l-minimax risk from SGDmax (3) list of "upper bounds" of Gamma_l-minimax risk from SGDmax (4) 2D array of estimated Gamma_l-minimax risks in each iteration (indexed by l) of the old prior (1st column) and new prior (2nd column)
    estimator: a Pytorch model
    Prior_Risk_Constraint_object: Prior_Risk_Constraint object
    n_SGDmax_iter: number of iterations in SGDmax (use 10 X n_SGDmax_iter for the initial grid)
    max_enlarge_iter: max number of iterations to enlarge grid
    optimizer: optimizer to update estimator
    n_SGD_max_sample: number of samples drawn for each distribution to estimate Risk in SGDmax
    n_accurate_Risk_sample: number of samples drawn for each distribution to accurately estimate Risk
    tol: tolerance in increment of [min max risk] to stop enlarging the grid of distributions
    max_enlarge_iter: max number of iteration to enlarge the grid of distributions
    b_ub, b_eq: constraints used in linprog that define the restricted set of priors Gamma. default to None
    n_new_distr_old_support, n_new_support_points, n_random_new_distr_new_support: see Prior_Risk_Constraint.enlarge_distr_grid
    save_SGDmax_result: whether SGDmax results are saved after running SGDmax each time'''
    risk_lower = []
    risk_upper = []
    risk_iter = []
    b_eq = np.concatenate((b_eq, np.ones(1))) if b_eq is not None else np.ones(1)
    if optimizer is None:
        optimizer = torch.optim.SGD(estimator.parameters(), lr=0.005)

    lower, upper = SGDmax(estimator, Prior_Risk_Constraint_object, n_iter=n_SGDmax_iter * 10, use_init_prior=False, optimizer=optimizer, n_sample=n_SGDmax_sample, b_ub=b_ub, b_eq=b_eq)
    if save_SGDmax_result:
        with open("l0.pkl", "wb") as saved_file:
            pickle.dump({"estimator": estimator, "lower": lower, "upper": upper}, saved_file)
    risk_lower.append(lower)
    risk_upper.append(upper)

    for l in range(max_enlarge_iter):
        with torch.no_grad():
            Risks = Prior_Risk_Constraint_object.calc_Risks_tensor(estimator, n_sample=n_accurate_Risk_sample)
        np_A_ub, np_A_eq = Prior_Risk_Constraint_object.get_constraint_matrices()
        linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex', x0=Prior_Risk_Constraint_object.prior_prob.cpu().numpy())
        if not linear_prog_result.success:
            linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex')
        Prior_Risk_Constraint_object.prior_prob = torch.as_tensor(linear_prog_result.x)
        
        Prior_Risk_Constraint_object.enlarge_distr_grid(n_new_distr_old_support=n_new_distr_old_support, n_new_support_points=n_new_support_points, n_random_new_distr_new_support=n_random_new_distr_new_support)
        old_prior = Prior_Risk_Constraint_object.prior_prob.clone()

        with torch.no_grad():
            Risks = Prior_Risk_Constraint_object.calc_Risks_tensor(estimator, n_sample=n_accurate_Risk_sample)
        np_A_ub, np_A_eq = Prior_Risk_Constraint_object.get_constraint_matrices()
        linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex', x0=Prior_Risk_Constraint_object.prior_prob.cpu().numpy())
        if not linear_prog_result.success:
            linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_ub=np_A_ub, b_ub=b_ub, A_eq=np_A_eq, b_eq=b_eq, bounds=(0., 1.), method='revised simplex')
        Prior_Risk_Constraint_object.prior_prob = torch.as_tensor(linear_prog_result.x)

        with torch.no_grad():
            Risks = Prior_Risk_Constraint_object.calc_Risks_tensor(estimator, n_sample=n_accurate_Risk_sample)
        new_risk = Risks.dot(Prior_Risk_Constraint_object.prior_prob)
        old_risk = Risks.dot(old_prior)
        risk_iter.append([old_risk,new_risk])
        
        if new_risk - old_risk <= tol:
            return (0, risk_lower, risk_upper, np.array(risk_iter))
        else:
            lower, upper = SGDmax(estimator, Prior_Risk_Constraint_object, n_iter=n_SGDmax_iter, use_init_prior=True, optimizer=optimizer, n_sample=n_SGDmax_sample, b_ub=b_ub, b_eq=b_eq)
            if save_SGDmax_result:
                with open("l"+str(l+1)+".pkl", "wb") as saved_file:
                    pickle.dump({"estimator": estimator, "lower": lower, "upper": upper}, saved_file)
            risk_lower.append(lower)
            risk_upper.append(upper)
    
    return (1, risk_lower, risk_upper, np.array(risk_iter))








def sample_mean(samples):
    '''samples: 2D tensor (# samples X sample size)'''
    return torch.mean(samples, dim=1).unsqueeze(1)

class nnet_estimator(torch.nn.Module):
    def __init__(self, n_transform_hidden_nodes=10, n_transformed_nodes=5, n_aggregated_hidden_nodes=10, activation=torch.relu, init_params=np.array((0., 0., 1.))):
        '''n_transform_hidden_nodes: number of hidden nodes for tranforming each single observation
        n_aggregated_hidden_nodes: number of hidden nodes after transformed observations are summed up (excluding the node for sample mean)
        activation: activation function
        init_params: initial parameters, vector of length 3. First component is intercept; second component is the slope of the previous hidden layer; third component is the slope of sample mean. Set to None for random initialization'''
        super().__init__()
        self.n_transform_hidden_nodes = n_transform_hidden_nodes
        self.n_transformed_nodes = n_transformed_nodes
        self.n_aggregated_hidden_nodes = n_aggregated_hidden_nodes
        self.activation = activation

        self.transform_hidden = torch.nn.Linear(1, self.n_transform_hidden_nodes, bias=True)
        self.transform_output = torch.nn.Linear(self.n_transform_hidden_nodes, self.n_transformed_nodes, bias=True)
        self.aggregate_hidden = torch.nn.Linear(self.n_transformed_nodes + 1, self.n_aggregated_hidden_nodes, bias=True)
        self.aggregate_output = torch.nn.Linear(self.n_aggregated_hidden_nodes + 1, 1, bias=True)
        self.output = torch.nn.Linear(2, 1, bias=True)
        if init_params is not None:
            self.output.bias.data = torch.as_tensor((init_params[0],))
            self.output.weight.data = torch.as_tensor(((init_params[1], init_params[2]),))
    def forward(self, input):
        Xbar = sample_mean(input)
        n_sample = input.size()[0]
        transform_hidden = self.activation(self.transform_hidden(input.reshape(-1,1)))
        transformed = self.activation(self.transform_output(transform_hidden))
        aggregated = transformed.reshape(n_sample, -1, self.n_transformed_nodes).mean(dim=1)
        aggregated_aug = torch.cat((aggregated, Xbar), dim=1)
        aggregate_hidden = self.activation(self.aggregate_hidden(aggregated_aug))
        aggregate_hidden_aug = torch.cat((aggregate_hidden, Xbar), dim=1)
        aggregate_output = self.activation(self.aggregate_output(aggregate_hidden_aug))
        aggregate_output_aug = torch.cat((aggregate_output, Xbar), dim=1)
        return self.output(aggregate_output_aug)


class Linear_mean_estimator(torch.nn.Module):
    def __init__(self, init_params=np.array((0, 1))):
        '''init_params: initial parameters, vector of length 2. First component is intercept; second component is slope.'''
        super().__init__()
        self.output_linear=torch.nn.Linear(1, 1, bias = True)
        if init_params is not None:
            self.output_linear.bias.data.fill_(init_params[0])
            self.output_linear.weight.data.fill_(init_params[1])
    def forward(self, input):
        return self.output_linear(sample_mean(input))



def parameter(support, distribution):
    '''mean'''
    return distribution.mm(support.unsqueeze(1))

def ub_constraint_fun(support, distribution):
    true_parameter = parameter(support, distribution)
    return torch.cat((true_parameter, -true_parameter), dim=1)

def Risk_fun(estimates, true_parameters):
    return torch.sum((estimates - true_parameters)**2, dim=2).mean(dim=1)

torch.manual_seed(893)
np.random.seed(5784)

estimator = nnet_estimator()
Prior_Risk_Constraint_object = Prior_Risk_Constraint(support=torch.tensor([0., 1.]), parameter=parameter, eq_constraint_fun=parameter, Risk_fun=Risk_fun)
b_eq = np.array([0.3])

result = calc_Gamma_minimax_estimator(estimator, Prior_Risk_Constraint_object, b_eq=b_eq)

import pickle
with open("minimax_estimator.pkl", "wb") as saved_file:
    pickle.dump({"estimator": estimator, "result": result, "Prior_Risk_Constraint_object": Prior_Risk_Constraint_object}, saved_file)

with open("minimax_estimator.pkl", "rb") as saved_file:
    results = pickle.load(saved_file)
result = results["result"]
estimator = results["estimator"]
Prior_Risk_Constraint_object = results["Prior_Risk_Constraint_object"]


mu = b_eq[0]
n = 10
theoretical_parameters = np.array([mu/(1+np.sqrt(n)),np.sqrt(n)/(1+np.sqrt(n))])
print(theoretical_parameters)
print(list(estimator.named_parameters()))
theoretical_estimator = Linear_mean_estimator(theoretical_parameters)


torch.manual_seed(893)
np.random.seed(5784)
Beta_prior = Prior_Risk_Constraint(support=torch.tensor([0., 1.]), parameter=parameter, eq_constraint_fun=parameter, Risk_fun=Risk_fun)
Beta_prior.distrs[0].distributions = torch.distributions.dirichlet.Dirichlet(torch.tensor([(1 - mu) * np.sqrt(n), mu * np.sqrt(n)])).sample((1000,))
Beta_prior.distrs[0].n_distr = 1000
Beta_prior.prior_prob = torch.ones(1000)/1000
Beta_prior.Distrs_n_distr = 1000
Beta_prior.distrs[0].eval_parameter_constraint(parameter, eq_constraint_fun=parameter)
risk = Beta_prior.calc_Risks_tensor(estimator,2000).dot(Beta_prior.prior_prob)
theoretical_risk = Beta_prior.calc_Risks_tensor(theoretical_estimator, 2000).dot(Beta_prior.prior_prob)

Risks = Beta_prior.calc_Risks_tensor(estimator, n_sample = 2000)
dummy, np_A_eq = Beta_prior.get_constraint_matrices()
linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_eq=np_A_eq, b_eq=np.concatenate((b_eq, np.ones(1))), bounds=(0., 1.), method='revised simplex')
Beta_prior.prior_prob = torch.as_tensor(linear_prog_result.x)
risk = Risks.dot(Beta_prior.prior_prob)

Risks = Beta_prior.calc_Risks_tensor(theoretical_estimator, n_sample = 2000)
linear_prog_result = linprog(-Risks.clone().detach().cpu().numpy(), A_eq=np_A_eq, b_eq=np.concatenate((b_eq, np.ones(1))), bounds=(0., 1.), method='revised simplex')
Beta_prior.prior_prob = torch.as_tensor(linear_prog_result.x)
theoretical_risk = Risks.dot(Beta_prior.prior_prob)

Prior_Risk_Constraint_object.calc_Risks_tensor(estimator,2000).dot(Prior_Risk_Constraint_object.prior_prob)
Prior_Risk_Constraint_object.calc_Risks_tensor(theoretical_estimator, 2000).dot(Prior_Risk_Constraint_object.prior_prob)

result
