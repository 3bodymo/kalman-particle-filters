import numpy as np

class KalmanFilter:
    def __init__(self, dt, std_acceleration, std_observation):
        """
        Initialize the Kalman Filter with the given parameters.
        
        Parameters:
        - dt: Time step
        - std_acceleration: Standard deviation of the acceleration noise
        - std_observation: Standard deviation of the observation noise
        """
        self.dt = dt
        self.std_acceleration = std_acceleration
        self.std_measurement = std_observation

        # State transition matrix
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Control matrix
        self.B = np.array([[0, 0, 0],
                           [0, 0.5*dt**2, 0],
                           [0, 0, 0],
                           [0, dt, 0]])
        
        # Control variables
        self.a = np.array([[0],
                           [-9.81], # -g
                           [0]])
        
        # Observation matrix
        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        # Process/State noise covariance
        self.Q = np.eye(4) * std_acceleration**2

        # Measurement noise covariance
        self.R = np.eye(2) * std_observation**2

        # Initial covariance estimate
        self.P = np.eye(4)

        # State Vector (initial state)
        self.u = np.zeros((4, 1))

    def predict(self):
        """
        Predict the next state and covariance based on the current state.
        """        
        # State prediction using state transition and control input
        self.u = np.dot(self.A, self.u) + np.dot(self.B, self.a)
        
        # Covariance prediction
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, observation):
        """
        Update the state and covariance with a new measurement.
        
        Parameters:
        - measurement: The new measurement for updating the state
        """
        if not np.isnan(observation).any():
            # Kalman gain computation
            S = np.dot(np.dot(self.C, self.P), self.C.T) + self.R
            K = np.dot(np.dot(self.P, self.C.T), np.linalg.inv(S))
        
            # Observation model
            y = observation - np.dot(self.C, self.u)
            
            # State update
            self.u = self.u + np.dot(K, y)
            
            # Covariance update
            I = np.eye(self.C.shape[1])
            self.P = np.dot((I - np.dot(K, self.C)), self.P)