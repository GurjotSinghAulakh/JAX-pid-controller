import matplotlib.pyplot as plt
from utils.config import PLANT, CONTROLLER, NUM_EPOCHS, LEARNING_RATE, NUM_TIMESTEPS 
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
    rets = consys.train()
    if isinstance(rets, tuple):
        mse_loss, kp_vals, ki_vals, kd_vals = rets

        # Plot for MSE Loss
        plt.figure(figsize=(10, 5))  # Optional: Adjust figure size
        plt.plot(range(NUM_EPOCHS), mse_loss, label='MSE', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('MSE Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot for KP, KI, KD gains
        plt.figure(figsize=(10, 5))  # Optional: Adjust figure size
        plt.plot(range(NUM_EPOCHS), kp_vals, label='KP', color='red')
        plt.plot(range(NUM_EPOCHS), ki_vals, label='KI', color='green')
        plt.plot(range(NUM_EPOCHS), kd_vals, label='KD', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('PID Gains Value')
        plt.title('PID Gains (KP, KI, KD) Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
        return

    plt.plot(range(NUM_EPOCHS), rets, label='MSE', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Value')
    plt.title('MSE Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
