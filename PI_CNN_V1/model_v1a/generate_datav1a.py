# generate_gaussian_data.py
import numpy as np
from pathlib import Path

def generate_gaussian_field(size=32, num_blobs=5):
    """Generate 2D field with overlapping Gaussian blobs"""
    field = np.zeros((size, size))
    
    for _ in range(num_blobs):
        # Random center
        cx = np.random.uniform(5, size-5)
        cy = np.random.uniform(5, size-5)
        
        # Random width (2-8 pixels)
        sigma = np.random.uniform(2, 8)
        
        # Random amplitude
        amp = np.random.uniform(0.3, 1.0)
        
        # Create Gaussian
        y, x = np.ogrid[0:size, 0:size]
        gaussian = amp * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
        field += gaussian
    
    # Normalize to [0, 1]
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field

# Generate dataset
Path('./data').mkdir(exist_ok=True)

for i in range(500):
    # Ground truth
    ground_truth = generate_gaussian_field()
    
    # Sparse sampling (16 random points)
    num_sensors = 16
    sensor_indices = np.random.choice(32*32, num_sensors, replace=False)
    
    # Create mask
    mask = np.zeros((32, 32))
    mask.flat[sensor_indices] = 1.0
    
    # Extract measurements
    measurements = ground_truth.flat[sensor_indices]
    
    # Save
    np.savez(f'./data/gaussian_{i:04d}.npz',
             measurements=measurements,
             mask=mask,
             ground_truth=ground_truth)
    
    if i % 100 == 0:
        print(f'Generated {i}/500 scenarios')

print('Dataset complete!')