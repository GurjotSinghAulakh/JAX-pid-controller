class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0
        self.prev_error = 0

    def calculate(self, error, dt):
        # Calculate the proportional term
        p = self.kp * error

        # Calculate the integral term
        self.error_sum += error
        i = self.ki * self.error_sum

        # Calculate the derivative term
        d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        # Calculate the PID output
        output = p + i + d

        return output
