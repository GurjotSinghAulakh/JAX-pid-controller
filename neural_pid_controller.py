import jax
import jax.numpy as jnp
from jax import grad, jit, random

class NeuralPIDController:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.params = self.initialize_params(layer_sizes)
        self.learning_rate = learning_rate

    def initialize_params(self, layer_sizes):
        params = []
        key = random.PRNGKey(0)
        for i in range(len(layer_sizes) - 1):
            key, subkey = random.split(key)
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            params.append({
                'weights': random.normal(subkey, (output_size, input_size)) * 0.01,
                'biases': jnp.zeros(output_size)
            })
        return params

    def forward(self, params, input):
        activations = input
        for layer in params[:-1]:
            activations = jax.nn.relu(jnp.dot(layer['weights'], activations) + layer['biases'])
        output = jnp.dot(params[-1]['weights'], activations) + params[-1]['biases']
        return output

    def loss_function(self, params, input, target):
        predictions = self.forward(params, input)
        return jnp.mean((predictions - target)**2)

    def train_step(self, params, input, target):
        grads = grad(self.loss_function)(params, input, target)
        return self.update_parameters(params, grads)

    def update_parameters(self, params, gradients, lr=0.01):
        return [(param - lr * grad) for param, grad in zip(params, gradients)]
