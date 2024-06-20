import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, dt, std_acceleration, std_observation, init_state):
        """
        Initialize the Particle Filter with the given parameters.
        
        Parameters:
        - num_particles: Number of particles
        - dt: Time step
        - std_acceleration: Standard deviation of the acceleration noise
        - std_observation: Standard deviation of the observation noise
        - init_state: Initial state for the particles (shape: [4, num_particles])
        """
        self.num_particles = num_particles
        self.dt = dt
        self.std_acceleration = std_acceleration
        self.std_observation = std_observation
        self.GRAVITY = 9.81

        # Initialize particles around the initial state
        self.particles = np.tile(init_state, (num_particles, 1)).T + np.random.randn(4, num_particles) * std_observation

        self.weights = np.ones(num_particles) / num_particles

    def predict(self):
        """
        Predict the next state for each particle.
        """
        # Create acceleration noise
        noise = np.random.randn(4, self.num_particles) * self.std_acceleration

        # Apply the state transition model
        self.particles[0] += self.dt * self.particles[2]  # x position update
        self.particles[1] += self.dt * self.particles[3]  # y position update => - 0.5 * 9.81 * self.dt**2
        # self.particles[2] += 0  # x velocity (no acceleration in x direction)
        self.particles[3] -=  self.GRAVITY * self.dt  # y velocity update (acceleration due to gravity)

        self.particles += noise

    def update(self, observation):
        """
        Update the weights of each particle based on the new observation.
        
        Parameters:
        - observation: The new observation for updating the state
        """
        observation_noise = np.random.randn(2, self.num_particles) * self.std_observation
        pred_observation = self.particles[:2, :] + observation_noise

        if observation is not None:
            distances = np.linalg.norm(pred_observation - observation.reshape(2, 1), axis=0)
            self.weights *= np.exp(-distances**2 / (2 * self.std_observation**2))
            self.weights /= sum(self.weights)

    def resample(self):
        """
        Resample particles with replacement with probability proportional to their weight.
        """
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.
        indices = np.searchsorted(cumulative_sum, np.random.rand(self.num_particles))
        self.particles = self.particles[:, indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        """
        Estimate the state based on the weighted particles.
        
        Returns:
        - state_estimate: The estimated state
        """
        return np.average(self.particles, weights=self.weights, axis=1)

    def run_filter(self, observations):
        """
        Run the Particle Filter on a series of observations.
        
        Parameters:
        - observations: List of observations over time
        
        Returns:
        - estimates: List of state estimates over time
        """
        estimates = []
        for observation in observations:
            self.predict()
            self.update(observation)
            self.resample()
            estimates.append(self.estimate())
        return np.array(estimates)
