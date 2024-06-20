import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

GRAVITY = 9.81

def simulate_ball_throwing(init_y, init_speed, launch_angle, dt, max_time):
    """
    Simulates the trajectory of a projectile thrown with an initial speed and angle from a given height.
    
    Parameters:
    - init_y: Initial height
    - init_speed: Launch speed
    - launch_angle: Launch angle (degrees)
    - dt: Time step for simulation
    - max_time: Total time for the simulation
    
    Returns:
    - positions: A numpy array of (x, y) positions of the projectile
    """
    angle_rad = np.radians(launch_angle)    # Convert angle to radians
    vx = init_speed * np.cos(angle_rad)     # Initial horizontal velocity
    vy = init_speed * np.sin(angle_rad)     # Initial vertical velocity
    
    # Generate time points from 0 to max_time with intervals of dt
    time_points = np.arange(0, max_time, dt)
    positions = []

    for t in time_points:
        x = vx * t  # Horizontal position
        y = init_y + vy * t - 0.5 * GRAVITY * t**2  # Vertical position
        
        if y < 0:   # Stop if the ball hits the ground (y = 0)
            break
        positions.append((x, y))
    
    return np.array(positions)


def simulate_noisy_observations(true_positions, noise_std, drop_out_interval=0):
    """
    Simulates observations of positions with added noise and optional dropout intervals.
    
    Parameters:
    - true_positions: Array of Actual Paths (shape: [n, 2] for n positions)
    - noise_std: Standard deviation of the observation noise
    - drop_out_interval: Interval index to introduce None
    
    Returns:
    - noisy_positions: Array of noisy positions with optional NaNs for dropout intervals
    """
    # Add Gaussian noise to the actual path
    noisy_positions = true_positions + np.random.normal(0, noise_std, true_positions.shape)
    
    if drop_out_interval > 0:
        # Determine the range of indices for the dropout interval
        interval_length = len(noisy_positions) // 5             # Split the positions into 5 equal parts
        start_idx = (drop_out_interval - 1) * interval_length   # Start index of the dropout interval
        end_idx = drop_out_interval * interval_length           # End index of the dropout interval
        
        # Add None in the specified interval
        noisy_positions[start_idx:end_idx] = None
    
    return noisy_positions


def visualize_particles(pf1, pf2, positions1, positions2, observations1, observations2, noisy_observations1, noisy_observations2, DT):
    """
    Visualizes the motion of particles.
    """
    max_length = max(len(observations1), len(observations2))
    fig, ax = plt.subplots()
    ax.set_xlim(0, max(positions1[:, 0].max(), positions2[:, 0].max()) + 10)
    ax.set_ylim(0, max(positions1[:, 1].max(), positions2[:, 1].max()) + 10)
    ax.plot(positions1[:, 0], positions1[:, 1], 'b-', label='Actual Path - Ball 1')
    ax.plot(noisy_observations1[:, 0], noisy_observations1[:, 1], 'rx', label='Observed Positions - Ball 1')
    estimated_path1, = ax.plot([], [], 'bo', label='Estimated Positions - Ball 1')
    ax.plot(positions2[:, 0], positions2[:, 1], 'g-', label='Actual Path - Ball 2')
    ax.plot(noisy_observations2[:, 0], noisy_observations2[:, 1], 'mx', label='Observed Positions - Ball 2')
    estimated_path2, = ax.plot([], [], 'go', label='Estimated Positions - Ball 2')
    particles1, = ax.plot([], [], 'b.', markersize=2)
    particles2, = ax.plot([], [], 'g.', markersize=2)

    def init():
        estimated_path1.set_data([], [])
        particles1.set_data([], [])
        estimated_path2.set_data([], [])
        particles2.set_data([], [])
        return estimated_path1, particles1, estimated_path2, particles2

    def animate(i):
        pf1.predict()
        pf2.predict()

        if i < len(observations1) and observations1[i] is not None:
            pf1.update(observations1[i])
        if i < len(observations2) and observations2[i] is not None:
            pf2.update(observations2[i])

        pf1.resample()
        pf2.resample()

        estimate1 = pf1.estimate()
        estimate2 = pf2.estimate()

        estimated_path1.set_data(estimate1[0], estimate1[1])
        particles1.set_data(pf1.particles[0, :], pf1.particles[1, :])

        estimated_path2.set_data(estimate2[0], estimate2[1])
        particles2.set_data(pf2.particles[0, :], pf2.particles[1, :])

        return estimated_path1, particles1, estimated_path2, particles2

    anima = animation.FuncAnimation(fig, animate, init_func=init, frames=max_length, interval=DT*1000, blit=True)

    plt.legend()
    plt.show()