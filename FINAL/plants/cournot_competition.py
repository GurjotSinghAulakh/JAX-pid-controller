class CournotCompetition:
    def __init__(self, initial_production, p_max, rival_production):
        self.q1 = max(initial_production, 0)
        self.p_max = p_max
        self.q2 = rival_production
        self.initial_value = initial_production
        self.initial_p_max = p_max
        self.initial_rival_production = rival_production

    def get_state(self):
        total_production = self.q1 + self.q2
        price = self.p_max - total_production
        profit = self.q1 * price
        return profit

    def update_state(self, control_signal, dt=1.0):
        self.q1 += control_signal * dt

    def reset(self):
        self.q1 = max(self.initial_value, 0)
        self.p_max = self.initial_p_max
        self.q2 = self.initial_rival_production
