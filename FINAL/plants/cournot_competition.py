import jax.numpy as jnp

class CournotCompetition:
    def __init__(self, initial_production, p_max, rival_production, marginal_cost):
        self.q1 = jnp.maximum(initial_production, 0)
        self.p_max = p_max
        self.q2 = rival_production
        self.initial_value = initial_production
        self.initial_p_max = p_max
        self.initial_rival_production = rival_production
        self.marginal_cost = marginal_cost
        self.epsilon = 1e-7

    def get_state(self):
        total_production = self.q1 + self.q2
        print(total_production)
        price = self.p_max - total_production
        profit = self.q1 * price - (self.q1 * self.marginal_cost)
        return profit

    def update_state(self, control_signal, D):
        self.q2 = self.q2 + D
        self.q1 += control_signal

    def reset(self):
        self.q1 = jnp.maximum(self.initial_value, 0)
        self.p_max = self.initial_p_max
        self.q2 = self.initial_rival_production
