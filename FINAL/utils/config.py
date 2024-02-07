

A = 10  # Cross-sectional area of the bathtub
C = 0.1  # Cross-sectional area of the drain
g = 9.8  # Gravitational constant
NUM_EPOCHS = 100  # Number of training epochs
NUM_TIMESTEPS = 10  # Number of timesteps per epoch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
SIZES = [3, 20, 1]  # Neural network sizes (input layer, hidden layer, output layer)

# Random seed for reproducibility
SEED = 0

# PID controller setting´
PID_SET_POINT = 1.0  # Target set point for PID controller
PID_KP = 1.0  # Proportional gain
PID_KI = 0.1  # Integral gain
PID_KD = 0.01  # Derivative gain

# Plant settings 

# For BathtubPlant
INITIAL_WATER_LEVEL = 1.0

# For CournotCompetition
INITIAL_PRODUCTION = 1.0
P_MAX = 5.0
RIVAL_PRODUCTION = 1.0

# For ChemicalReactionPlant
INITIAL_CONCENTRATION = 0.0
REACTION_RATE_CONSTANT = 0.1
