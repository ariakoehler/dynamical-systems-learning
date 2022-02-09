import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

import torch
device = 'cpu'
from torchdiffeq import odeint

from plotly.offline import iplot
import plotly.graph_objs as go


from torch_params_utils import *



if __name__ == '__main__':
    check_grad = False
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

        if check_grad:
            if it > 0:
                for i in range(optlor.eta.shape[0]):
                    for j in range(optlor.eta.shape[1]):
                        print('Backprop Derivative for {},{} = {:.2f}'.format(i, j, optlor.eta.grad[i,j]))
                        print('Checked Derivative for {},{} = {:.2f}'.format(i, j, grad_check[i,j]))
                        print('Error Ratio = {:.2f}'.format(optlor.eta.grad[i,j] / grad_check[i,j]))
                        print()
            print('\n')

        print('Iterarion {}'.format(it+1))
        print('eta = \n{}'.format(optlor.eta.detach().numpy()))
        print('loss = {:.2f}\n\n'.format(loss_curr))
        
        loss_vec.append(loss_curr.detach().numpy())

        if check_grad:

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
