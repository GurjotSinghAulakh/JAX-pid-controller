import jax
from jax import random


class PIDController:
    def __init__(self, kp, ki, kd, set_point, timesteps):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point
        self.integral = 0
        self.prev_error = 0
        self.num_timesteps = timesteps

    def update(self, current_value):
        """Update the PID controller using JAX."""
        error = self.set_point - current_value
        P = self.kp * error
        self.integral += error
        derivative = (error - self.prev_error)
        self.prev_error = error

        return P + (self.ki * self.integral) + (self.kd * derivative)

    def make_loss_function(self, key, controller=None, plant=None):
        def pid_loss(kp, ki, kd, set_point, initial_height, num_timesteps, key):
            pid = PIDController(kp, ki, kd, set_point, num_timesteps)
            # pid = PIDController(kp, ki, kd, set_point)
            plant.reset()
            # bathtub = BathtubPlant(initial_height, A, C, g)
            total_error = 0.0

            for _ in range(num_timesteps):
                key, subkey = random.split(key)
                D = random.uniform(subkey, (), minval=-0.01, maxval=0.01)  # Random disturbance/noise
                current_height = plant.get_state()
                U = pid.update(current_height)
                plant.update_state(U, D)
                error = set_point - current_height
                total_error += error**2

            mse = total_error / num_timesteps
            return mse

        def make_loss_function(set_point, initial_height, num_timesteps, key):
            def loss_fn(kp, ki, kd):
                return pid_loss(kp, ki, kd, set_point, initial_height, num_timesteps, key)
            return loss_fn

        loss_fn = make_loss_function(self.set_point, 0.0, self.num_timesteps, key)

        grad_loss_fn = jax.grad(loss_fn, argnums=[0, 1, 2])
        grad_jit = jax.jit(grad_loss_fn)
        self.loss_fn = loss_fn
        self.grad_jit = grad_jit


    def reset(self):
        self.integral = 0
        self.prev_error = 0