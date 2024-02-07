class PIDErrorCalculator:
    def __init__(self, set_point):
        self.set_point = set_point
        self.integral_error = 0
        self.previous_error = 0

    def reset(self):
        self.integral_error = 0
        self.previous_error = 0

    def calculate_errors(self, current_value):
        error = self.set_point - current_value
        self.integral_error += error
        derivative_error = error - self.previous_error
        self.previous_error = error

        return [error, self.integral_error, derivative_error]
