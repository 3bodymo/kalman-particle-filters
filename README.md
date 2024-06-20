# Kalman and Particle Filters for Ball Throwing Simulation

This project demonstrates the implementation of Kalman and Particle Filters to estimate the trajectory of a projectile. The filters are applied to noisy observations of the projectile's path to provide more accurate state estimates.

## Project Structure

```
filters/
    - KalmanFilter.py: Implementation of the Kalman Filter
    - ParticleFilter.py: Implementation of the Particle Filter
utils/
    - helper.py: Utility functions for simulating projectile motion and noisy observations and visualizing particle movements

- Kalman_Filter.ipynb: Jupyter Notebook demonstrating Kalman Filter usage
- Particle_Filter.ipynb: Jupyter Notebook demonstrating Particle Filter usage
- Particles_Visualization.py: Visualizing the particle filter estimates
```

## Setup Instructions

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib

### Installation

```sh
pip install numpy matplotlib
```

## Usage

### Running the Kalman Filter

The `Kalman_Filter.ipynb` notebook provides an example of how to use the `KalmanFilter` class. Open the notebook and follow the instructions to see the Kalman Filter in action.

### Running the Particle Filter

The `Particle_Filter.ipynb` notebook provides an example of how to use the `ParticleFilter` class. Open the notebook and follow the instructions to see the Particle Filter in action.

Alternatively, you can run the `Particles_Visualization.py` script to visualizing particle movements:

```sh
python utils/Particles_Visualization.py
```

### Example Code

#### Simulating Projectile Motion

```python
from utils.helper import simulate_ball_throwing, simulate_noisy_observations

positions = simulate_ball_throwing(init_y=10, init_speed=20, launch_angle=45, dt=0.01, max_time=100)
noisy_observations = simulate_noisy_observations(positions, noise_std=0.9, drop_out_interval=4)
```

#### Using the Kalman Filter

```python
from filters.KalmanFilter import KalmanFilter
kf = KalmanFilter(dt=0.01, std_acceleration=0.1, std_observation=0.9)
for obs in noisy_observations:
    kf.predict()
    kf.update(obs)
    print(kf.u)  # Print the state estimate
```

#### Using the Particle Filter

```python
from filters.ParticleFilter import ParticleFilter
pf = ParticleFilter(num_particles=1000, dt=0.01, std_acceleration=0.1, std_observation=0.9, init_state=[0, 10, 20*np.cos(np.radians(45)), 20*np.sin(np.radians(45))])
for obs in noisy_observations:
    pf.predict()
    pf.update(obs)
    pf.resample()
    print(pf.estimate())  # Print the state estimate
```

## Visualizing the Results

The `visualize_particles` function in `utils/helper.py` provides a way to visualize the performance of the Particle Filter. The function shows the actual, observed, and estimated paths of the projectiles.

```python
from utils.helper import visualize_particles

visualize_particles(pf1, pf2, positions1, positions2, observations1, observations2, noisy_observations1, noisy_observations2, DT)
```

## Author

This project was created by Abdullah Abdelrazek.
