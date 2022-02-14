# dynamical-systems-learning

Finds coefficient values of hidden terms in a dynamical system given the shape of the hidden terms.

## Dependencies

- numpy
- scipy
- matplotlib
- torch
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## Structure of the Code

[torch_learn_params.py](https://github.com/carterkoehler/dynamical-systems-learning/blob/main/torch_learn_params.py): Contains main functionality, including the creation of the torch model and optimization loop.

[torch_params_utils.py](https://github.com/carterkoehler/dynamical-systems-learning/blob/main/torch_params_utils.py): Contains modules used by the main function, including the torch module that serves as an argument to `odeint`.

[notebooks/](https://github.com/carterkoehler/dynamical-systems-learning/tree/main/notebooks): Contains jupyter notebooks with elementary tests of the code and visualizations of the results. Primarily serves as scratch space for development.

## Current Problem

We are concerned with a perturbed Lorenz system, where the perturbations come in the form of functions added onto the RHS of the equations, scaled by small parameters eta. We wish to know if we can use standard optimization techniques, as implemented by torch, to learn the values of these parameters, given that we know what shape they are (i.e. x-squared or xy).

## Methodology

## Running the Code

The main file ([torch_learn_params.py](https://github.com/carterkoehler/dynamical-systems-learning/blob/main/torch_learn_params.py)) runs in Python 3.6 or later. Ensure all dependencies are installed correctly beforehand.

For the sake of testing different conditions, the python script can be run with command-line arguments 

- `--lr` specifies the learning rate
- `--max_it` specifies the number of iterations before terminating
- `--check_grads` if True, outputs an approximation of the gradients of each parameter at the current value alongside the gradients used in the last update
- `--eta_method` specifies how the initial eta should be chosen
 - "random" samples the etas randomly from a Gaussian distribution with low variance centered on 0.
 - "zeros" initializes them all to be 0
 - "true" sets them to the true values of eta. This one is intended to be a check on our methods
