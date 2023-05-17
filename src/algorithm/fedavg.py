import torch

from .basealgorithm import BaseOptimizer

import numpy as np

def lagrange_basis(x, idx, x_eval):
    """Compute Lagrange basis polynomial for the given data points and evaluation points."""
    n = len(x)
    result = 1
    for j in range(n):
        if j != idx:
            result *= (x_eval - x[j]) / (x[idx] - x[j])
    return result

# def lagrange_basis(x, idx, x_eval):
#     """Compute Lagrange basis polynomial for the given data points and evaluation points."""
#     n = len(x)
#     result = 1
#     for j in range(n):
#         if j != idx:
#             result *= (x_eval - x[j]) / (x[idx] - x[j])
#     return result

def get_mean_std_and_sample(weights, use_sampling):
    mean = torch.mean(weights, dim=0)
    std = torch.std(weights, dim=0, unbiased=False)
    # print("mean" , mean)
    # print("std" , std)
    # check if all stds are zero
    if torch.all(std == 0):
        std += 1e-9
    if use_sampling:
        return torch.normal(mean, std)
    else:
        return mean
    
def sigmoid_kernel(x):
    return torch.sigmoid(x)

class FedavgOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        self.lr = kwargs.get('lr')
        self.momentum = kwargs.get('momentum', 0.)
        defaults = dict(lr=self.lr, momentum=self.momentum)
        super(FedavgOptimizer, self).__init__(params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta = param.grad.data
                if beta > 0.:
                    if 'momentum_buffer' not in self.state[param]:
                        self.state[param]['momentum_buffer'] = torch.zeros_like(param).detach()
                    self.state[param]['momentum_buffer'].mul_(beta).add_(delta.mul(1. - beta)) # \beta * v + (1 - \beta) * grad
                    delta = self.state[param]['momentum_buffer']
                param.data.sub_(delta)
        return loss

    def accumulate(self, mixing_coefficient, local_param_iterator, partial_agg_condition=lambda name: None, lagrage = False, use_sampling = False, ):
        for group in self.param_groups:
            for server_param, (name, local_param) in zip(group['params'], local_param_iterator):
                if partial_agg_condition(name):
                    continue
                if server_param.grad is None:
                    server_param.grad = server_param.data.sub(local_param.data).mul(mixing_coefficient)
                else:
                    server_param.grad.add_(server_param.data.sub(local_param.data).mul(mixing_coefficient))


    def sample_model(self, local_param_iterator):
        use_sampling = True
        weight_dict = {}
        for name, local_param in local_param_iterator:
            if name not in weight_dict:
                weight_dict[name] = []
            weight_dict[name].append(local_param.data.clone()) 
            
        for name in weight_dict:
            weights = torch.stack(weight_dict[name])
            # print 10 elements of each weight
            print(name)
            for weight in weights:
                print(weight[:2])
            
            mean = torch.mean(weights, dim=0)
            std = torch.std(weights, dim=0)

            new_weight_value = get_mean_std_and_sample(weights, use_sampling)
        
            for group in self.param_groups:
                for server_param in group['params']:
                    if server_param.name == name:
                        server_param.data = new_weight_value
        
    

    # def accumulate(self, mixing_coefficient, local_param_iterator, partial_agg_condition=lambda name: None, lagrage = False, use_sampling = False, ):
    #     # Hardcoded evaluation points
        
    #     if lagrage:
    #         evaluation_points = np.array([-2, -1, 0, 1, 2])

    #         for group in self.param_groups:
    #             for server_param, (name, local_params) in zip(group['params'], local_param_iterator):
    #                 if partial_agg_condition(name):
    #                     continue

    #                     num_clients = len(local_params)

    #                     if server_param.grad is None:
    #                         server_param.grad = server_param.data.new_zeros(server_param.data.shape)

    #                     client_indices = np.arange(num_clients)

    #                     for x_eval in evaluation_points:
    #                         mixed_weight = 0
    #                         for idx, local_param in enumerate(local_params):
    #                             basis = lagrange_basis(client_indices, idx, x_eval)
    #                             mixed_weight += mixing_coefficient * basis * (server_param.data - local_param.data)
    #                         server_param.grad.add_(mixed_weight / len(evaluation_points))
        
    #     elif use_sampling:
    #         self.sample_model(local_param_iterator)
        
    #     else:
            
    #         for group in self.param_groups:
    #             for server_param, (name, local_param) in zip(group['params'], local_param_iterator):
    #                 if partial_agg_condition(name):
    #                     continue
    #                 if server_param.grad is None:
    #                     server_param.grad = server_param.data.sub(local_param.data).mul(mixing_coefficient)
    #                 else:
    #                     server_param.grad.add_(server_param.data.sub(local_param.data).mul(mixing_coefficient))





    # def accumulate(self, mixing_coefficient, local_param_iterator, partial_agg_condition=lambda name: None, lagrage = False, use_sampling = False, ):
    #     for group in self.param_groups:
    #         for server_param, (name, local_param) in zip(group['params'], local_param_iterator):
    #             if partial_agg_condition(name):
    #                 continue
    #             if server_param.grad is None:
    #                 server_param.grad = sigmoid_kernel(server_param.data.sub(local_param.data))
    #             else:
    #                 server_param.grad.add_(sigmoid_kernel(server_param.data.sub(local_param.data)))