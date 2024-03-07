import torch
import numpy as np
from torch import optim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def ackley(x, y):
    return -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2))) - torch.exp(0.5 * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y))) + torch.exp(torch.tensor([1.0])) + 20


'''
Plotting various key objective functions
'''


x = torch.linspace(-30, 30, 300)
y = torch.linspace(-30, 30, 300)
x, y = torch.meshgrid(x, y, indexing = 'xy')

def plot_function(objective_function, x, y):
    z = objective_function(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(x, y, z, cmap = 'viridis')
    return fig

plot_function(rosenbrock, x, y).savefig("Rosenbrock.png")
plot_function(beale, x, y).savefig("Beale.png")
plot_function(ackley, x, y).savefig("Ackley.png")


# Re-plotting Ackley function over a smaller range
x = torch.linspace(-3, 3, 100)
y = torch.linspace(-3, 3, 100)
x, y = torch.meshgrid(x, y, indexing = 'xy')

plot_function(ackley, x, y).savefig('Ackley3.png')


'''
Performing gradient descent to optimize passed objective function, accepts various optimizers
'''
def optimizer_values(objective_function, optimizer_function, learning_rate):
    
    num_iterations = 30000

    x = torch.tensor([10.0], requires_grad = True)
    y = torch.tensor([10.0], requires_grad = True)

    optimizer = optimizer_function([x, y], lr = learning_rate)

    objective_values = []
    
    for iteration in range(num_iterations):
        # Zeroes out previous gradients
        optimizer.zero_grad()
        # Calculates value of objective function
        objective = objective_function(x, y)
        # Computes gradient of objective function
        objective.backward()
        # Updates x, y
        optimizer.step()
        # Appends value of objective function to array
        objective_values.append(objective.item())
        
    
    print(x, y, objective_values[-1])
    return objective_values


# Optimize various objective functions with Stochastic Gradient Descent 
rosenbrock_sgd = optimizer_values(rosenbrock, torch.optim.SGD, 0.00001)
ackley_sgd = optimizer_values(ackley, torch.optim.SGD, 0.00001)
beale_sgd = optimizer_values(beale, torch.optim.SGD, 0.0000001)


'''
# Optimizing various objective functions with Adam optimizer
'''


rosenbrock_adam = optimizer_values(rosenbrock, torch.optim.Adam, 0.001)
ackley_adam = optimizer_values(ackley, torch.optim.Adam, 0.001)
beale_adam = optimizer_values(beale, torch.optim.Adam, 0.001)


'''
Plotting the convergence of SGD and Adam Optimizers for various functions
'''


def plot_optimized(sgd_values, sgd_rate, adam_values, adam_rate, function_name):
    
    # Create plot labels and titles
    sgd_label = f'SGD (lr={sgd_rate})'
    adam_label = f'Adam (lr={adam_rate})'
    plot_title = f'Optimization of {function_name} Function'
    save_title = f'{function_name}_convergence'
    
    # Create plot, assign labels, and save with title
    plt.figure(figsize=(10, 6))
    plt.plot(sgd_values, label=sgd_label)
    plt.plot(adam_values, label=adam_label)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Value')
    plt.title(plot_title)
    plt.legend()
    plt.savefig(save_title)
    
plot_optimized(rosenbrock_sgd, 0.00001, rosenbrock_adam, 0.001, 'Rosenbrock')
plot_optimized(ackley_sgd, 0.00001, ackley_adam, 0.001, 'Ackley')
plot_optimized(beale_sgd, 0.0000001, beale_adam, 0.001, 'Beale')
