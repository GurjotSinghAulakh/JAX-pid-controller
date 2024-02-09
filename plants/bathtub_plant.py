import jax.numpy as jnp

class BathtubPlant:
    def __init__(self, initial_level, area, drain_area, g=9.8):
        # Ensure initial water level is non-negative
        self.water_level = initial_level
        self.A = area
        self.C = drain_area
        self.g = g
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C
        self.initial_value = initial_level  # Used for resetting plant
        self.epsilon = 1e-7

    def get_state(self):
        return self.water_level

    def update_state(self, control_signal, D, dt=1.0):
        self.water_level = jnp.maximum(self.water_level, self.epsilon)  # Ensure water level is non-negative or too small
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C
        delta_B = (control_signal + D - self.Q) * dt
        delta_H = delta_B / self.A
        self.water_level += delta_H
        return self.water_level

    def reset(self):
        self.water_level = self.initial_value
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C
