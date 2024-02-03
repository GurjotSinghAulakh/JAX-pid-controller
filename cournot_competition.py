class CournotCompetition:
    def __init__(self, initial_production, p_max, rival_production):
        """Initialize the Cournot Competition model parameters."""
        self.q1 = max(initial_production, 0)  # Ensure non-negative production
        self.p_max = p_max
        self.q2 = rival_production

    def get_current_state(self):
        """Calculate and return the current profit for producer 1."""
        total_production = self.q1 + self.q2
        price = self.p_max - total_production
        profit = self.q1 * price
        return profit

    def apply_control(self, control_signal, dt):
        """Apply the control signal to update the production quantity."""
        self.q1 += control_signal * dt 