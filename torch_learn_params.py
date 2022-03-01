import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse

import torch
device = 'cpu'
from torchdiffeq import odeint

from torch_params_utils import *



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--eta_method', type=str, choices=['random', 'zeros', 'actual'], default='random')
    parser.add_argument('--check_grads', action='store_true')
    parser.add_argument('--max_it', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    eta_method = args.eta_method
    check_grads = args.check_grads
    max_it = args.max_it
    
    
    x0 = torch.tensor([8.0, 0.0, 30.0]).to(device)
    x0.requires_grad_()

    t_space = torch.linspace(0, 100, 1000).to(device)
    tol = 10**-5

    m_basis_fns = 2
    n_dim = 3

    
    if eta_method == 'random':
        eta0 = torch.tensor(np.random.normal(0, 0.001, (n_dim,m_basis_fns)), dtype=torch.float).to(device)
    elif eta_method == 'zeros':
        eta0 = torch.tensor(np.zeros((n_dim,m_basis_fns)), dtype=torch.float).to(device)
    elif eta_method == 'actual':
        eta0 = torch.tensor(np.array([[0, 0.003], [0, 0], [0.005, 0]]), dtype=torch.float).to(device)
    else:
        raise ValueError('You are trying to set eta in a way that is not supported. Check eta_method.')
    
    
    with torch.no_grad():
        true_eta = torch.tensor([[0, 0.003], [0, 0], [0.005, 0]], dtype=torch.float)
        true_l63 = lambda t, x : L63_torch_modified(t, x, true_eta)
        true_soln = odeint(true_l63, x0, t_space)
        true_diff = torch.diff(true_soln, axis=0)


    optlor = OptimizeLorenz(x0, t_space, 2, eta0=eta0).to(device)


    optimizer = torch.optim.Adam([optlor.eta], lr=args.lr)
    loss = torch.nn.MSELoss().to(device)
    # loss = DiffLoss().to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True)

    print('eta_0 = \n{}\n'.format(optlor.eta.detach().numpy()))

    loss_vec = []
    eta_log = []
    
    np.set_printoptions(precision=4)
    for it in range(max_it):
    
        optimizer.zero_grad()
    
        # pred_soln = odeint(optlor, x0, t_space).to(device)
        pred_diff = make_predictions(t_space, true_soln, optlor).to(device)
        
        loss_curr = loss(pred_diff, true_diff)
        loss_curr.retain_grad()
        loss_curr.backward()

        if check_grads:
            if it > 0:
                for i in range(optlor.eta.shape[0]):
                    for j in range(optlor.eta.shape[1]):
                        print('Backprop Derivative for {},{} = {:.2f}'.format(i, j, optlor.eta.grad[i,j]))
                        print('Checked Derivative for {},{} = {:.2f}'.format(i, j, grad_check[i,j]))
                        print()
                print('\n')

        print('Iterarion {}'.format(it+1))
        print('eta = \n{}'.format(optlor.eta.detach().numpy()))
        eta_log.append(optlor.eta.detach().numpy())
        print('loss = {:.3f}\n\n'.format(loss_curr))

        if np.linalg.norm(optlor.eta.grad.detach().numpy()) < tol:
            if not (eta_method == 'actual' and it < 1):
                break
        
        loss_vec.append(loss_curr.detach().numpy())

        print(optlor.eta.grad)
        # print(pred_diff)
        # print(true_diff)

        if check_grads:

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
        scheduler.step(loss_curr)

    fig, ax = plt.subplots(1,1)
    ax.plot(range(len(loss_vec)), loss_vec)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Loss as a Function of Iterations')
    ax.set_xlim((0, max_it))
    ax.set_ylim((0, max(loss_vec) * 1.2))
    plt.grid()
    plt.show()

    
    # fig, ax = plt.subplots(3,1)
    # ax[0].set_title('Movement in Eta Space')
    # for i in range(3):
    #     eta_0s = [eta_log[j][i,0] for j in range(len(eta_log))]
    #     eta_1s = [eta_log[j][i,1] for j in range(len(eta_log))]
    #     print(eta_0s)
    #     print(eta_1s)
    #     ax[i].plot(eta_0s, eta_1s)
    #     ax[i].set_xlabel('$\eta_{}^1$'.format(i+1))
    #     ax[i].set_ylabel('$\eta_{}^2$'.format(i+1))
    #     ax[i].set_xlim((-0.05, 0.05))
    #     ax[i].set_ylim((-0.05, 0.05))
    #     ax[i].grid()
    # plt.show()
