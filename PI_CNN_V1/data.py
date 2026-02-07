import torch
import torch.nn as nn
import numpy as np

from pathlib import Path

#do really simple data generation to be replaced later by K wave or MUST Toolbox
size = 32
num_blobs = 5

def generate_guassian_blob(size = size, num_blobs = num_blobs):

    #first generate an empty gaussian block field 
    field = np.zeros()

    for _ in range(num_blobs):

        # why random center
        # 
        c_x = np.random.uniform(low=5, high=size-5)
        c_y = np.random.uniform(low=5, high=size-5)

        #random width 
        sigma = np.random.uniform(low=2, high=8)

        #random amplitude 
        amplitude = np.random.uniform(low=1,high=5)

        y, x = np.ogrid[0:size, 0:size]

        numerator = ((x-c_x)**2 + (y-c_y)**2)
        denominator = 2*sigma**2
        gaussian_blob = amplitude* np.exp(-(numerator/denominator))

        field = field + gaussian_blob
    
    num_f = field - field.min()
    den_f = field.max() - field.max() + 1e-8
    field = num_f/den_f

    return field

Path("./data_set").mkdir(exist_ok=True)

for i in range(500):

    ground_truth = generate_guassian_blob()

    #sample sparsely 
    #to simulate sampling sparsely we need to input the 32 by 32 sensor array
    # and then randomely choose a sensor

    #simulated snesors 
    sensor_number = 16
    sensor_indices = np.random.choice(32*32,16,replace=False)

    mask = np.zeros(32*32)
    mask.flat[sensor_indices] = 1.0

    measurements = ground_truth.flat[sensor_indices]

    #save as npz
    np.savez(f'./data/gaussian_{i:04d}.npz',
             measurements=measurements,
             mask=mask,
             ground_truth=ground_truth)






