import matplotlib.pyplot as plt
import optax
from utils.config import * 
from consys import CONSYS


def main():
    plant = PLANT
    controller = CONTROLLER
    consys = CONSYS(
        plant=plant,
        controller=controller,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
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
