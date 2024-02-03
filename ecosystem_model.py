import jax.numpy as jnp
from jax import random

class EcosystemModel:
    def __init__(self, initial_population, carrying_capacity, birth_rate, death_rate):
        """
        Initialize the Ecosystem model parameters.
        :param initial_population: Initial population of the species.
        :param carrying_capacity: The maximum sustainable population.
        :param birth_rate: The birth rate of the species.
        :param death_rate: The natural death rate of the species.
        """
        self.population = initial_population
        self.carrying_capacity = carrying_capacity
        self.birth_rate = birth_rate
        self.death_rate = death_rate

    def get_current_state(self):
        """
        Return the current population.
        """
        return self.population

    def apply_control(self, control_signal, dt, key):
        """
        Apply the control signal to update the population based on the birth rate.
        :param control_signal: The adjustment to the birth rate.
        :param dt: Time step.
        :param key: Random key for generating noise.
        """
        # Update birth rate based on control signal
        self.birth_rate += control_signal

        # Calculate the change in population
        growth = self.birth_rate * self.population * (1 - self.population / self.carrying_capacity)
        death = self.death_rate * self.population
        noise = random.uniform(key, minval=-0.01, maxval=0.01) * self.population

        # Update the population with the growth, death, and noise
        self.population += (growth - death + noise) * dt
        self.population = max(self.population, 0)  # Ensure population doesn't go negative
