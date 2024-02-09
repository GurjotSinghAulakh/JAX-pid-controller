# General Purpose JAX-based Controller Project

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project implements a general-purpose PID controller using Python’s JAX package for automatic differentiation. It includes both traditional and AI-driven (neural network-based) approaches to PID control, capable of simulating a wide variety of controllable systems and applying PID controllers to control tasks. The system uses JAX to compute gradients and update controller parameters via gradient descent.

## Installation

### Requirements

- Python 3.8+
- JAX
- NumPy
- Matplotlib (for visualization)

### Setup

Clone the repository and install dependencies:

```git clone <repository-url>```

```cd <project-directory>```

```pip install -r requirements.txt```

## Usage
To run the controller, simply execute the following command:
```python3 main.py```

You can customize simulation parameters by editing the config.py file. This allows for easy adjustments to the plant model, controller type, and other simulation settings directly from the configuration file, without needing to modify the main script.

## Configuration
Configuration parameters can be set in config.py, including the plant to simulate, controller type, neural network architecture, learning rate, and simulation parameters like the number of epochs and timesteps per epoch.

## File Structure
`main.py`: The entry point of the application.

`config.py:` Configuration parameters for simulation runs.

`consys.py:` Defines the system that includes both controller and plant.

`pid_controller.py, nn_controller.py:` Implementations of the PID and neural network-based controllers.

`pid_error_calculator.py:` Utility for calculating PID errors.

`bathtub_plant.py, cournot_competition.py, chemical_reaction_plant.py:` Plant implementations.

Other utility scripts for the project’s functionality.

## Visualizations
Visualizations for the progression of learning and changes to PID parameters (for the standard PID controller) are generated using Matplotlib and can be viewed after running simulations.
