import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

import torch
device = 'cpu'
from torchdiffeq import odeint

from plotly.offline import iplot
import plotly.graph_objs as go

from matplotlib import rcParams

rcParams['figure.figsize'] = 8,8



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



# OptimizeLorenz and code using it adapted from examples by the makers
# of torchdiffeq, in particular, the one which can be found at 
# https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py


class OptimizeLorenz(torch.nn.Module):

    def __init__(self, x0, t_space, n_terms):
        super(OptimizeLorenz, self).__init__()
        self.x0 = x0
        self.t_space = t_space
        self.n_terms = n_terms
        
        self.eta = torch.tensor(np.random.normal(0, 0.01, (3,self.n_terms)), dtype=torch.float).to(device)
        self.eta.requires_grad_()
        

    def rhs(self, x, t):
        return L63_torch_modified(t, x, self.eta)
        
    def forward(self, t, x):
        return self.rhs(x, t)



if __name__ == '__main__':
    max_it = 100

    x0 = torch.tensor([8.0, 0.0, 30.0]).to(device)
    x0.requires_grad_()

    t_space = torch.linspace(0, 10, 1000).to(device)
    
    

    with torch.no_grad():
        true_eta = torch.tensor([[0, 0.003], [0, 0], [0.005, 0]], dtype=torch.float)
        true_l63 = lambda t, x : L63_torch_modified(t, x, true_eta)
        true_soln = odeint(true_l63, x0, t_space)
        
    optlor = OptimizeLorenz(x0, t_space, 2).to(device)
    eta0 = torch.tensor(np.random.normal(0, 0.01, (3,2)), dtype=torch.float).to(device)

    optimizer = torch.optim.Adam([optlor.eta], lr=1e-6)
    loss = torch.nn.MSELoss().to(device)


    loss_vec = []
    
    np.set_printoptions(precision=4)
    for it in range(max_it):
    
        optimizer.zero_grad()
    
        pred_soln = odeint(optlor, x0, t_space).to(device)
        
        loss_curr = loss(pred_soln, true_soln)
        loss_curr.retain_grad()
        loss_curr.backward()

        if it > 0:
            for i in range(optlor.eta.shape[0]):
                for j in range(optlor.eta.shape[1]):
                    print('Backprop Derivative for {},{} = {:0f}'.format(i, j, optlor.eta.grad[i,j]))
                    print('Checked Derivative for {},{} = {:0f}\n'.format(i, j, grad_check[i,j]))
            raise
        print('\n')

        print('Iterarion {}'.format(it+1))
        print('eta = \n{}'.format(optlor.eta.detach().numpy()))
        print('loss = {}\n'.format(loss_curr))
        loss_vec.append(loss_curr.detach().numpy())
        print('===Derivative Check===')

        grad_check = np.zeros((optlor.eta.shape[0], optlor.eta.shape[1]))
        for i in range(optlor.eta.shape[0]):
            for j in range(optlor.eta.shape[1]):
                eps = 10**(-3)
                eta_check_0 = optlor.eta.detach().clone()
                eta_check_1 = optlor.eta.detach().clone()
                eta_check_0[i,j] += eps
                eta_check_1[i,j] -= eps
        
                l63_check_0 = lambda t, S : L63_torch_modified(t, S, eta_check_0)
                x_pred_check_0 = odeint(l63_check_0, x0, t_space)
                l63_check_1 = lambda t, S : L63_torch_modified(t, S, eta_check_1)
                x_pred_check_1 = odeint(l63_check_1, x0, t_space)
                
                L0 = loss(x_pred_check_0, true_soln)
                L1 = loss(x_pred_check_1, true_soln)
                grad_check[i,j] = ((L0 - L1)/(2*eps)).item()
        
        optimizer.step()

    fig, ax = plt.subplots(1,1)
    ax.plot(range(max_it), loss_vec)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Loss as a Function of Iterations')
    ax.set_xlim((0, max_it))
    ax.set_ylim((0, max(loss_vec) + 10))
    plt.grid()
    plt.show()
