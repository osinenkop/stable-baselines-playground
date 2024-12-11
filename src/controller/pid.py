class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Desired setpoint (e.g., upright position)

        self.integral = 0.0
        self.previous_error = 0.0

    def compute(self, measurement, dt):
        # Compute the error
        error = self.setpoint - measurement

        # Proportional term
        p = self.kp * error

        # Integral term
        self.integral += error * dt
        i = self.ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / dt
        d = self.kd * derivative

        # Update previous error
        self.previous_error = error

        # Compute the control action
        control_action = p + i + d
        return control_action
