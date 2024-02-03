import matplotlib.pyplot as plt
from consys import ConSys
from pid_controller import PIDController
from bathtub import BathtubPlant

def main():
    pid = PIDController(kp=1.0, ki=0.0, kd=0.1)

    area = 1.0  # Cross-sectional area of the bathtub
    drain_area = 0.01  # Cross-sectional area of the drain
    initial_water_level = 10.0  # Initial water height in the bathtub
    gravitational_constant = 9.8  # Gravitational constant

    # Creating an instance of BathtubPlant
    bathtub = BathtubPlant(initial_water_level, area, drain_area, gravitational_constant)

    # Adding a bathtub to the system
    system = ConSys(controller=pid, plant=bathtub)

    setpoint = 10  # Desired water level
    total_time = 100  # Total time for the simulation

    water_levels, mse_values, kp_values, ki_values, kd_values = system.run(setpoint, total_time)

    # Plotting the results
    plt.figure(figsize=(18, 6))

    # Plot 1: Water Level
    plt.subplot(1, 3, 1)
    plt.plot(water_levels, label='Water Level')
    plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Time')
    plt.ylabel('Water Level')
    plt.title('Bathtub Water Level Control')
    plt.legend()

    # Plot 2: PID Parameters
    plt.subplot(1, 3, 2)
    plt.plot(kp_values, label='Kp')
    plt.plot(ki_values, label='Ki')
    plt.plot(kd_values, label='Kd')
    plt.xlabel('Time')
    plt.ylabel('PID Parameters')
    plt.title('PID Parameter Values Over Time')
    plt.legend()

    # Plot 3: MSE
    plt.subplot(1, 3, 3)
    plt.plot(mse_values, label='MSE', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Progression (MSE)')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
