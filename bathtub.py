import jax.numpy as jnp
from jax import random

class BathtubPlant:
    def __init__(self, initial_water_level, area, drain_area, g=9.8):
        self.water_level = initial_water_level
        self.A = area
        self.C = drain_area
        self.g = g  # acceleration due to gravity (m/s^2)

    def get_water_level(self):
        return self.water_level

    def update_state(self, control_signal, dt, key):
        V = jnp.sqrt(2 * self.g * self.water_level)
        Q = V * self.C  
        D = random.uniform(key, minval=-0.1, maxval=0.1)  
        self.H += (control_signal - Q - D) * dt / self.A
        self.H = jnp.maximum(self.H, 0) 
        self.water_level -= self.H * dt
        return self.get_state()
