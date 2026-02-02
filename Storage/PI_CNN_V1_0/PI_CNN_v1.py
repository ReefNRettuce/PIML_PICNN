"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
"""I've copy and pasted this from the deep xde website. I think once I build the UNET I can 
Actually build this with a PDE. """

"""
This is composed as a series of modules
The first module is that it is a UNET architecture
The Second module is that it is a Physics Informed UNET
The third module is that it uses matlab generated synthetic training data
The fourth module is the bicubic spline interpolator
So I'm taking the example code from the XDE website and putting that into 
this part. Now. Then I need to add the data from the matlab script and replace
the datagenerator in the exmaple as the data. After that I need to replace the model. 

"""


import deepxde as dde
import numpy as np
import torch

#GPU accelerator 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# General parameters
n = 2
precision_train = 5
precision_test = 60
hard_constraint = True
weights = 100  # if hard_constraint == False
iterations = 5000
parameters = [1e-5, 3, 150, "sin"] #these need to change i probably need relu

# Define sine function
sin = dde.backend.sin #probably eras this 

learning_rate, num_dense_layers, num_dense_nodes, activation = parameters

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)

    if dde.backend.backend_name == "jax":
        y = y[0]
        dy_xx = dy_xx[0]
        dy_yy = dy_yy[0]

    f = k0**2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
    return -dy_xx - dy_yy - k0**2 * y - f

def func(x):
    return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])


def transform(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
    return res * y


def boundary(_, on_boundary):
    return on_boundary

#the geometry is no longer a square/rectangle this can be erased
geom = dde.geometry.Rectangle([0, 0], [1, 1])
k0 = 2 * np.pi * n
wave_len = 1 / n


#I think this can stay
hx_train = wave_len / precision_train
nx_train = int(1 / hx_train)

hx_test = wave_len / precision_test
nx_test = int(1 / hx_test)


#WTF does this do remove this 
if hard_constraint == True:
    bc = []
else:
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

#remove this 
data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain=nx_train**2,
    num_boundary=4 * nx_train,
    solution=func,
    num_test=nx_test**2,
)


#also remove this
net = dde.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
)

#keep this 
net.to(device)

if hard_constraint == True:
    net.apply_output_transform(transform)


#I don't understand what these do at all.
model = dde.Model(data, net)

if hard_constraint == True:
    model.compile("adam", lr=learning_rate, metrics=["l2 relative error"])
else:
    loss_weights = [1, weights]
    model.compile(
        "adam",
        lr=learning_rate,
        metrics=["l2 relative error"],
        loss_weights=loss_weights,
    )

#i think these are explanatory and good to have. 
losshistory, train_state = model.train(iterations=iterations)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)