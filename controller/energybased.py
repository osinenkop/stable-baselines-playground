import numpy as np

class EnergyBasedController:
    def __init__(self, mass=1.0, length=1.0, gravity=9.81, dt=0.05):
        self.mass = mass  # Mass of the pendulum
        self.length = length  # Length of the pendulum
        self.gravity = gravity  # Gravitational acceleration
        self.dt = dt # Fundamental step size of the plant to be controlled

        # Integral term for upright stabilization
        self.integral_error = 0.0

    def compute(self, angle, cos_angle, angular_velocity):
        proportional_gain = 24.0
        damping_coefficient = 27.0
        integral_gain = 25.0

        control_action = 0.0

        if cos_angle < -0.8:  # Energy injection for upswing
            control_action = 2.0 * np.sign(angular_velocity)
            print(f"angle: {angle:.2f}, angular_velocity: {angular_velocity:.2f}, control_action: {np.clip(control_action, -2.0, 2.0):.2f} SWINGUP")
        elif cos_angle > 0.8:  # Stabilization near upright
           # Update the integral error
            self.integral_error += angle * self.dt

            # Compute control action with proportional, integral, and damping terms
            control_action = (
                - proportional_gain * angle
                - damping_coefficient * angular_velocity
                - integral_gain * self.integral_error
            )
            print(f"angle: {angle:.2f}, angular_velocity: {angular_velocity:.2f}, control_action: {np.clip(control_action, -2.0, 2.0):.2f} HOLD")

        return control_action
