from controllers.nn_controller import NNController
from controllers.pid_controller import PIDController
from plants.cournot_competition import CournotCompetition
from plants.bathtub_plant import BathtubPlant
from plants.chemical_reaction_plant import ChemicalReactionPlant
from jax import random
from jax.nn import relu, tanh, sigmoid


NUM_EPOCHS = 20  # Number of training epochs
NUM_TIMESTEPS = 50  # Number of timesteps per epoch
LEARNING_RATE = 0.01  # Learning rate for the optimizer
SIZES = [3, 15, 1]  # Neural network sizes (input layer, hidden layer, output layer)
NOISE_RANGE = 0.01

# Random seed for reproducibility
SEED = 0

# For PIDController
PID_SET_POINT = 1.0  # Target set point for PID controller
PID_KP = 1.0  # Proportional gain
PID_KI = 0.1  # Integral gain
PID_KD = 0.01  # Derivative gain

# For NNController
KEY = random.PRNGKey(SEED)
ACTIVATION_FUNC = tanh
INIT_RANGE_MIN = 0.0
INIT_RANGE_MAX = 1.0

# For BathtubPlant
INITIAL_WATER_LEVEL = 1.0
A = 10  # Cross-sectional area of the bathtub
C = 0.1  # Cross-sectional area of the drain
G = 9.8  # Gravitational constant

# For CournotCompetition
INITIAL_PRODUCTION = 1.0
P_MAX = 5.0
RIVAL_PRODUCTION = 1.0
MARGINAL_COST = 0.01

# For ChemicalReactionPlant
INITIAL_CONCENTRATION = 0.0
REACTION_RATE_CONSTANT = 0.1


# PLANT = BathtubPlant(initial_level=INITIAL_WATER_LEVEL, area=A, drain_area=C, g=G)
# PLANT = CournotCompetition(initial_production=INITIAL_PRODUCTION, p_max=P_MAX, rival_production=RIVAL_PRODUCTION, marginal_cost=MARGINAL_COST)
PLANT = ChemicalReactionPlant(initial_concentration=INITIAL_CONCENTRATION, reaction_rate_constant=REACTION_RATE_CONSTANT)
CONTROLLER = NNController(sizes=SIZES, key=KEY, activation_func=ACTIVATION_FUNC, init_range_min=INIT_RANGE_MIN, init_range_max=INIT_RANGE_MAX, noise_range=NOISE_RANGE)
# CONTROLLER = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD, timesteps=NUM_TIMESTEPS, set_point=PID_SET_POINT, noise_range=NOISE_RANGE)
