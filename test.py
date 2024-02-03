import jax.numpy as jnp
from jax import jit, random
import matplotlib.pyplot as plt
from jax import grad

# Constants for the bathtub model using JAX numpy
A = 10  # Cross-sectional area of the bathtub
C = 0.1  # Cross-sectional area of the drain
g = 9.8  # Gravitational constant

class BathtubPlant:

    def __init__(self, initial_level, area, drain_area, g=9.8):
        # Ensure initial water level is non-negative
        self.water_level = initial_level
        self.A = area
        self.C = drain_area
        self.g = g
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C  

        # Used for resetting plant
        self.intial_height = initial_level
    
    def get_state(self):
        return self.water_level

    def update_state(self, control_signal, D):
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C  
        delta_B = control_signal + D - self.Q
        delta_H = delta_B / self.A
        self.water_level += delta_H
        return self.water_level
    
    def reset(self):
        self.water_level = self.intial_height
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C


# PID Controller using JAX
class PIDController:
    def __init__(self, kp, ki, kd, set_point):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point
        self.integral = 0
        self.prev_error = 0

    def update(self, current_value):
        """Update the PID controller using JAX."""
        # error = self.set_point - current_value
        # self.integral += error
        # derivative = error - self.prev_error
        # self.prev_error = error

        error = self.set_point - current_value
        P = self.kp * error
        self.integral += error
        derivative = (error - self.prev_error)
        self.prev_error = error

        # mse = self.mse_fun(current_value)

        return P + (self.ki * self.integral) + (self.kd * derivative)

        # return self.kp + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        self.integral = 0
        self.prev_error = 0


# Simulation parameters
num_epochs = 100
num_timesteps = 100
# initial_height = jnp.array(1.0)  # Starting water height
initial_height = 20  # Starting water height
# target_height = jnp.array(1.0)  # Target water height
target_height = 20
pid_params = {'kp': 0.1, 'ki': 5, 'kd': 3}  # PID parameters
key = random.PRNGKey(0)  # Random seed for JAX

bathtub = BathtubPlant(initial_height, A, C, g)

# Initialize PID controller
pid = PIDController(**pid_params, set_point=target_height)

# Simulation and Visualization
heights = []
errors = []

kp_values, ki_values, kd_values = [], [], []


for epoch in range(num_epochs):

    # a) Reset the plant to its initial state
    bathtub.reset()

    pid.reset()

    # a) Reset the plant to its initial state
    error_history_over_this_epoch = []

    for t in range(num_timesteps):
        key, subkey = random.split(key)
        D = random.uniform(subkey, (), minval=-0.01, maxval=0.01)  # Random disturbance/noise
        current_height = bathtub.water_level
        U = pid.update(current_value=current_height)
        bathtub.update_state(U, D)
        error = target_height - current_height

        error_history_over_this_epoch.append(error)
        heights.append(current_height)

    mse = jnp.mean(jnp.square(jnp.array(error_history_over_this_epoch)))
    print(mse)
    errors.append(mse)
    
    print('Errors over time', errors)

print('Final PID parameters KP: ', pid.kp, 'KI: ', pid.ki, 'KD: ', pid.kd)




# Convert results to NumPy for plotting
heights_np = jnp.array(heights)
errors_np = jnp.array(errors)


# Visualization
plt.figure(figsize=(18, 6))

# Plotting mean squared error
plt.subplot(1, 3, 1)
plt.plot(errors_np)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error Over Time')
plt.xlim(0, num_epochs)

# Plotting water height over time
plt.subplot(1, 3, 2)
plt.plot(heights_np)
plt.xlabel('Epochs')
plt.ylabel('Water Height')
plt.title('Water Height Over Time')
plt.xlim(0, num_epochs)

# Plotting PID parameters over time
plt.subplot(1, 3, 3)
plt.plot(kp_values, label='Kp')
plt.plot(ki_values, label='Ki')
plt.plot(kd_values, label='Kd')
plt.xlabel('Epochs')
plt.ylabel('PID Parameters')
plt.title('PID Parameters Over Time')
plt.xlim(0, num_epochs)
plt.legend()


plt.tight_layout()
plt.show()


