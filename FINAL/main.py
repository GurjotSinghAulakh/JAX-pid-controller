from jax import random
from jax.nn import relu, tanh, sigmoid
import matplotlib.pyplot as plt
import optax
from utils.config import * 
from consys import CONSYS
from utils.pid_error_calculator import PIDErrorCalculator
from controllers.nn_controller import NNController
from controllers.pid_controller import PIDController
from plants.cournot_competition import CournotCompetition
from plants.bathtub_plant import BathtubPlant
from plants.chemical_reaction_plant import ChemicalReactionPlant


def main():
    key = random.PRNGKey(0)  # Random seed for JAX

    plant = BathtubPlant(initial_level=INITIAL_WATER_LEVEL, area=A, drain_area=C, g=g)
    # plant = CournotCompetition(initial_production=1.0, p_max=5.0, rival_production=1.0)
    # plant = ChemicalReactionPlant(initial_concentration=0.0, reaction_rate_constant=0.1)
    # controller = NNController(sizes=SIZES, key=key, activation_func=sigmoid)
    controller = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD, timesteps=NUM_TIMESTEPS, set_point=1.0)
    consys = CONSYS(
        plant=plant,
        controller=controller,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        pid_point=5.0,
        num_timesteps=NUM_TIMESTEPS
    )
    mse_loss = consys.train()

    plt.plot(range(NUM_EPOCHS), mse_loss, label='MSE', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Value')
    plt.title('MSE Over Epochs')
    plt.legend()

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
