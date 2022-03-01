import numpy as np
import scipy
import torch
device = 'cpu'

# Adapted from Matt Levine's GitHub at https://github.com/mattlevine22/contRNN
def L63_torch(t, S, sigma=10.0, rho=28.0, beta=8.0/3):
    """ Lorenz-63 dynamical model implemented for torch. """
    x_1 = sigma*(S[1]-S[0])
    x_2 = S[0]*(rho-S[2])-S[1]
    x_3 = S[0]*S[1] - beta*S[2]
    dS  = torch.hstack((x_1,x_2,x_3))
    return dS

def L63_torch_modified(t, S, eta, sigma=10.0, rho=28.0, beta=8.0/3):
    """ Lorenz-63 dynamical model with added terms implemented for torch. """
    x_1 = sigma*(S[1]-S[0])
    x_2 = S[0]*(rho-S[2])-S[1]
    x_3 = S[0]*S[1] - beta*S[2]
    dS  = torch.hstack((x_1,x_2,x_3))
    terms = torch.tensor([S[1]*S[2], S[0]**2])
    dS += torch.matmul(eta, terms)
    return dS


def L63_torch_modified_vec(t, S, eta, sigma=10.0, rho=28.0, beta=8.0/3):
    """ Lorenz-63 dynamical model with added terms implemented for torch. """
    x_1 = sigma*(S[:,1]-S[:,0])
    x_2 = S[:,0]*(rho-S[:,2])-S[:,1]
    x_3 = S[:,0]*S[:,1] - beta*S[:,2]
    dS  = torch.vstack((x_1,x_2,x_3))
    terms = torch.vstack([S[:,1]*S[:,2], S[:,0]**2])
    m = torch.matmul(eta, terms)
    dS = dS + m
    return dS.T


def make_predictions(t, x_true, optlor):
    y_pred = optlor.forward(t, x_true)[1:,:]
    return y_pred


class DiffLoss(torch.nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, a, b):

        a_diff = torch.diff(a, axis=0)
        b_diff = torch.diff(b, axis=0)

        return torch.mean((b_diff - a_diff)**2)



# OptimizeLorenz and code using it adapted from examples by the makers
# of torchdiffeq, in particular, the one which can be found at 
# https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py


class OptimizeLorenz(torch.nn.Module):

    def __init__(self, x0, t_space, n_terms, eta0=None):
        super(OptimizeLorenz, self).__init__()
        self.x0 = x0
        self.t_space = t_space
        self.n_terms = n_terms

        if eta0 == None:
            self.eta = torch.tensor(np.random.normal(0, 0.01, (3,self.n_terms)), dtype=torch.float).to(device)
        else:
            self.eta = eta0
        self.eta.requires_grad_()
        

    def rhs(self, x, t):
        return L63_torch_modified_vec(t, x, self.eta)
        
    def forward(self, t, x):
        return self.rhs(x, t)

