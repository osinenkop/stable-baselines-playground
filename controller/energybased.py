import numpy as np

class EnergyBasedController:
    def __init__(self, mass=1.0, length=1.0, gravity=9.81):
        self.mass = mass  # Mass of the pendulum
        self.length = length  # Length of the pendulum
        self.gravity = gravity  # Gravitational acceleration

    def compute(self, cos_theta, angular_velocity):
        # Calculate the potential energy: m * g * l * (1 - cos(theta))
        potential_energy = self.mass * self.gravity * self.length * (1 - cos_theta)
        
        # Calculate the kinetic energy: 0.5 * m * l^2 * angular_velocity^2
        kinetic_energy = 0.5 * self.mass * (self.length ** 2) * (angular_velocity ** 2)
        
        # Total energy
        total_energy = potential_energy + kinetic_energy
        
        # Desired energy: energy at the upright position
        desired_energy = self.mass * self.gravity * self.length
        
        # Energy difference
        energy_deficit = desired_energy - total_energy

        # Control action initialization
        control_action = 0.0

        # Apply torque to increase energy if there is an energy deficit
        if cos_theta < -0.2:  # If the pendulum is below the horizontal
            control_action = 2.0 * np.sign(angular_velocity)  # Apply torque to increase energy
        elif cos_theta > 0.6:  # If the pendulum is close to upright
            # Apply a stronger proportional term for stabilization
            proportional_gain = 24.0  # Proportional gain for holding upright
            damping_coefficient = 30.0  # Damping coefficient to reduce oscillations
            control_action = -proportional_gain * (1 - cos_theta) - damping_coefficient * angular_velocity
        else:
            # Smoothly transition between energy injection and stabilization
            control_action = -0.3 * angular_velocity  # Moderate damping

        # Debugging: Print the control action for tuning
        # print(f"cos_theta: {cos_theta:.2f}, angular_velocity: {angular_velocity:.2f}, control_action: {control_action:.2f}")

        return control_action