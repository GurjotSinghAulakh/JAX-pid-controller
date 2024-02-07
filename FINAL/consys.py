import jax
import jax.numpy as jnp
from jax import random
import optax
from utils.pid_error_calculator import PIDErrorCalculator
from controllers.nn_controller import NNController
from controllers.pid_controller import PIDController

class CONSYS:
    def __init__(self, plant, controller, num_epochs, lr, pid_point, num_timesteps):
        self.plant = plant
        self.controller = controller
        self.num_epochs = num_epochs
        self.lr = lr
        self.pid_point = pid_point
        self.num_timesteps = num_timesteps
        if isinstance(self.controller, NNController):
            self.run_inner = self.inner_nn
        elif isinstance(self.controller, PIDController):
            self.run_inner = self.inner_pid

    def train(self):
        mse_loss = []
        # network_params = self.controller.init_network_params(
        #     self.controller.layer_sizes,
        #     self.controller.key
        # )
        for epoch in range(self.num_epochs):
            loss = self.run_inner(epoch)
            # self.plant.reset()

            # epoch_key = random.PRNGKey(epoch)
            # optimizer = optax.adam(self.lr)
            # pid_errors = PIDErrorCalculator(set_point=self.pid_point)
            # opt_state = optimizer.init(network_params)
            # loss, network_params, opt_state = self.controller.step(
            #     opt_state,
            #     network_params,
            #     self.controller,
            #     self.plant,
            #     self.plant.initial_value,
            #     self.num_timesteps,
            #     epoch_key,
            #     optimizer,
            #     pid_errors
            # )
            mse_loss.append(loss)

            print(f"Epoch {epoch}, Loss: {loss}")
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
        pid_errors = PIDErrorCalculator(set_point=self.pid_point)
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
            self.kp = 0.5
            self.ki = 1.0
            self.kd = 1.0

        grads = self.controller.grad_jit(
            self.kp,
            self.ki,
            self.kd,
        )
        self.kp -= self.lr * grads[0]
        self.ki -= self.lr * grads[1]
        self.kd -= self.lr * grads[2]
        mse = self.controller.loss(
            self.kp,
            self.ki,
            self.kd,
        )
        print(mse)
