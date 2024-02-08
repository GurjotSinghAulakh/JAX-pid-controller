import jax
import jax.numpy as jnp
from jax import random
import optax
from utils.pid_error_calculator import PIDErrorCalculator
from controllers.nn_controller import NNController
from controllers.pid_controller import PIDController
from utils.config import PID_KD, PID_KI, PID_KP

class CONSYS:
    def __init__(self, plant, controller, num_epochs, lr, num_timesteps):
        self.plant = plant
        self.controller = controller
        self.num_epochs = num_epochs
        self.lr = lr
        self.num_timesteps = num_timesteps
        if isinstance(self.controller, NNController):
            self.run_inner = self.inner_nn
        elif isinstance(self.controller, PIDController):
            self.run_inner = self.inner_pid

    def train(self):
        mse_loss = []
        kp_vals, ki_vals, kd_vals = [], [], []
        for epoch in range(self.num_epochs):
            rets = self.run_inner(epoch)
            if isinstance(rets, tuple):
                kp_vals.append(rets[1])
                ki_vals.append(rets[2])
                kd_vals.append(rets[3])
                rets = rets[0]

            mse_loss.append(rets)

            print(f"Epoch {epoch}, Loss: {rets}")
        if kp_vals:
            return (mse_loss, kp_vals, ki_vals, kd_vals)
        return mse_loss

    def inner_nn(self, epoch):
        optimizer = optax.adam(self.lr)
        if epoch == 0:
            network_params = self.controller.network_params
            opt_state = optimizer.init(network_params)
        else:
            network_params = self.network_params
            opt_state = self.opt_state
        self.plant.reset()

        epoch_key = random.PRNGKey(epoch)
        pid_errors = PIDErrorCalculator(set_point=self.plant.initial_value)
        loss, network_params, opt_state = self.controller.step(
            opt_state,
            network_params,
            self.controller,
            self.plant,
            self.plant.initial_value,
            self.num_timesteps,
            epoch_key,
            optimizer,
            pid_errors
        )
        self.network_params = network_params
        self.opt_state = opt_state
        return loss

    def inner_pid(self, epoch):
        if epoch == 0:
            self.controller.make_loss_function(
                key=random.PRNGKey(epoch),
                plant=self.plant,
            )
            self.kp = PID_KP
            self.ki = PID_KI
            self.kd = PID_KD

        grads = self.controller.grad_jit(
            self.kp,
            self.ki,
            self.kd,
        )
        self.kp -= self.lr * grads[0]
        self.ki -= self.lr * grads[1]
        self.kd -= self.lr * grads[2]
        mse = self.controller.loss_fn(
            self.kp,
            self.ki,
            self.kd,
        )
        return (mse, self.kp, self.ki, self.kd)
