import numpy as np
from filters.ParticleFilter import ParticleFilter
from utils.helper import *

NOISE_STD = 0.9
INIT_Y1 = 10
INIT_Y2 = 20
INIT_SPEED1 = 20
INIT_SPEED2 = 15
LAUNCH_ANGLE1 = 45
LAUNCH_ANGLE2 = 30
DT = 0.01

positions1 = simulate_ball_throwing(init_y=INIT_Y1, init_speed=INIT_SPEED1, launch_angle=LAUNCH_ANGLE1, dt=DT, max_time=100)
positions2 = simulate_ball_throwing(init_y=INIT_Y2, init_speed=INIT_SPEED2, launch_angle=LAUNCH_ANGLE2, dt=DT, max_time=100)

noisy_observations1 = simulate_noisy_observations(positions1, noise_std=NOISE_STD, drop_out_interval=4)
noisy_observations2 = simulate_noisy_observations(positions2, noise_std=NOISE_STD, drop_out_interval=3)

std_acceleration = 0.01
std_observation = NOISE_STD
num_particles = 10000

init_state1 = np.array([0, INIT_Y1, INIT_SPEED1 * np.cos(np.radians(LAUNCH_ANGLE1)), INIT_SPEED1 * np.sin(np.radians(LAUNCH_ANGLE1))])
init_state2 = np.array([0, INIT_Y2, INIT_SPEED2 * np.cos(np.radians(LAUNCH_ANGLE2)), INIT_SPEED2 * np.sin(np.radians(LAUNCH_ANGLE2))])

pf1 = ParticleFilter(num_particles, DT, std_acceleration, std_observation, init_state1)
pf2 = ParticleFilter(num_particles, DT, std_acceleration, std_observation, init_state2)

observations1 = [obs if not np.isnan(obs).any() else None for obs in noisy_observations1]
observations2 = [obs if not np.isnan(obs).any() else None for obs in noisy_observations2]

visualize_particles(pf1, pf2, positions1, positions2, observations1, observations2, noisy_observations1, noisy_observations2, DT)