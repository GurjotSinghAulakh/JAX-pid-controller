class ChemicalReactionPlant:
    def __init__(self, initial_concentration, reaction_rate_constant):
        self.product_concentration = max(initial_concentration, 0)  # Ensure non-negative concentration
        self.k = reaction_rate_constant
        self.initial_value = initial_concentration

    def get_state(self):
        return self.product_concentration

    def update_state(self, reactant_feed, dt):
        rate_of_reaction = self.k * reactant_feed
        self.product_concentration += rate_of_reaction * dt - self.product_concentration * dt
        return self.product_concentration

    def reset(self):
        self.product_concentration = 0