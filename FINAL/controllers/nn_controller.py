from jax import random, value_and_grad, nn
import jax.numpy as jnp
from utils.pid_error_calculator import PIDErrorCalculator
import optax


class NNController:
    def __init__(self, sizes, key, activation_func):
        self.layer_sizes = sizes
        self.key = key
        self.activation_func = activation_func
        self.init_network_params(sizes, key)

    def init_network_params(self, sizes, key):
        params = []
        for i in range(len(sizes) - 1):
            d_in, d_out = sizes[i], sizes[i+1]
            key, subkey = random.split(key)
            W = random.normal(subkey, (d_in, d_out)) * jnp.sqrt(2.0 / d_in)
            b = jnp.zeros(d_out)
            params.append((W, b))
        self.network_params = params
        return params

    def nn_forward(self, params, x):
        """Forward pass through the network."""
        for w, b in params[:-1]:
            outputs = jnp.dot(x, w) + b
            # x = nn.relu(outputs)
            x = self.activation_func(outputs)

        final_w, final_b = params[-1]
        y = jnp.dot(x, final_w) + final_b
        return y

    def update(self, params, pid_errors):
        """Compute the control signal based on the current state and the target."""
        pid_errors = jnp.array(pid_errors).reshape(1, -1)
        u = self.nn_forward(params, pid_errors)
        return u

    def nn_loss(self, params, controller, plant, set_point, num_timesteps, epoch_key, pid_errors):
        total_loss = 0.0
        for i in range(num_timesteps):
            iter_key, subkey = random.split(epoch_key)
            D = random.uniform(subkey, (), minval=-0.01, maxval=0.01)  # Random disturbance/noise

            current_state = plant.get_state()
            pid_err = pid_errors.calculate_errors(current_state)

            control_signal = controller.update(params, pid_err)
            plant.update_state(control_signal, D)

            error = set_point - current_state
            total_loss += error**2

        mse = total_loss / num_timesteps
        return (jnp.squeeze(mse))

    # update step function using optax
    def step_update(self, params, opt_state, grads, optimizer):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    def step(self, opt_state, network_params, controller, plant, set_point, num_timesteps, epoch_key, optimizer, pid_errors):
        # Compute the gradient of the loss with respect to network parameters
        loss_value, grads = value_and_grad(self.nn_loss, argnums=0)(network_params, controller, plant, set_point, num_timesteps, epoch_key, pid_errors)

        # Apply the gradients to the network parameters
        network_params, opt_state = self.step_update(network_params, opt_state, grads, optimizer)
        return loss_value, network_params, opt_state

