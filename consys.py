import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

class consys:
    def __init__(self, controller, plant):
        self.controller = controller
        self.plant = plant
        self.k_values = []
        self.mse_values = []

    def generate_noise(self):
        self.key, subkey = jax.random.split(self.key)
        return jax.random.uniform(subkey, minval=self.noise_range[0], maxval=self.noise_range[1])


    def control_pid(self):
        # Control the PID controller using the bathtub class
        while True:
            error = self.bathtub.get_error()
            control_signal = self.pid.calculate_control_signal(error)
            self.bathtub.apply_control_signal(control_signal)
            self.k_values.append(self.pid.k)
            self.mse_values.append(self.bathtub.mse)

    def plot_k_values(self):
        plt.plot(self.k_values)
        plt.xlabel('Iteration')
        plt.ylabel('K Value')
        plt.title('K Value Progression')
        plt.show()

    def plot_mse_values(self):
        plt.plot(self.mse_values)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('MSE Progression')
        plt.show()

# Create an instance of the consys class
system = consys()

# Control the PID controller and collect data
system.control_pid()

# Plot the K values progression
system.plot_k_values()

# Plot the MSE progression
system.plot_mse_values()
