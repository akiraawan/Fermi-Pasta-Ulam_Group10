import tensorflow as tf
import numpy as np

# Constants and settings
learning_rate = 0.001
epochs = 1000
a = 10  # Truncation of the real line for the loss calculation
alpha = 0.1  # Alpha parameter from the equation

# Define a custom layer for the characteristic transformation
class CharacteristicLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CharacteristicLayer, self).__init__()
        self.snn = self.add_weight(shape=(), initializer="random_normal", trainable=True)

    def call(self, inputs):
        t, x = inputs
        return x - self.snn * t

# Define the neural network architecture
class FpuWaveNet(tf.keras.Model):
    def __init__(self):
        super(FpuWaveNet, self).__init__()
        self.characteristic_layer = CharacteristicLayer()
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        z = self.characteristic_layer(inputs)
        z = tf.expand_dims(z, -1)  # Ensure z has the right shape for Dense layers
        return self.dense_layers(z)

# Initialize models for U and V
U_nn = FpuWaveNet()
V_nn = FpuWaveNet()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# # Define the loss functions
# def loss_function(U_model, V_model, t_points, x_points):
#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(x_points)
#         U_outputs = U_model([t_points, x_points])
#         V_outputs = V_model([t_points, x_points])
#         U_x = tape.gradient(U_outputs, x_points)
#         U_xx = tape.gradient(U_x, x_points)
#         F_V = U_xx + alpha * tape.gradient(U_x * U_x, x_points)
    
#     # Loss for the PDE residuals and boundary conditions
#     loss_PDE = tf.reduce_mean(tf.square(V_outputs - F_V))
#     loss_boundary = tf.reduce_mean(tf.square(U_outputs)) + tf.reduce_mean(tf.square(V_outputs))

#     return loss_PDE + loss_boundary

# def loss_function(U_model, V_model, t_points, x_points, a):
#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(x_points)
#         U_outputs = U_model([t_points, x_points])
#         V_outputs = V_model([t_points, x_points])
#         U_x = tape.gradient(U_outputs, x_points)
#         U_xx = tape.gradient(U_x, x_points)
#         F_V = U_xx + alpha * tape.gradient(U_x * U_x, x_points)

#         # Add Neumann Boundary Condition
#         U_x_a = tape.gradient(U_outputs, x_points)[-1]  # Gradient at x = a
#         U_x_neg_a = tape.gradient(U_outputs, x_points)[0]  # Gradient at x = -a

#     # Loss for the PDE residuals
#     loss_GSE = tf.reduce_mean(tf.square(V_outputs - F_V))
    
#     # Boundary losses
#     loss_limit = tf.square(U_outputs[-1] - U_outputs[0])  # Simple version for demonstration

#     # Neumann Boundary Condition Loss
#     loss_BC = tf.square(U_x_a) + tf.square(U_x_neg_a)

#     # Preventing translation
#     loss_trans = tf.square(U_outputs[0] - (U_outputs[-1] + U_outputs[0]) / 2)

#     # Total loss
#     total_loss = loss_GSE + loss_limit + loss_BC + loss_trans
#     return total_loss

def loss_function(U_model, V_model, t_points, x_points, a):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_points)
        U_outputs = U_model([t_points, x_points])
        V_outputs = V_model([t_points, x_points])
        U_x = tape.gradient(U_outputs, x_points)
        U_xx = tape.gradient(U_x, x_points)
        F_V = U_xx + alpha * tape.gradient(U_x * U_x, x_points)

        # Add Neumann Boundary Condition derivatives
        U_x_a = tape.gradient(U_outputs, x_points)[-1]  # Gradient at x = a
        U_x_neg_a = tape.gradient(U_outputs, x_points)[0]  # Gradient at x = -a

    # Loss for the PDE residuals approximated by an integral
    loss_GSE = tf.reduce_mean(tf.square(V_outputs - F_V))

    # Boundary condition losses at -a and a
    U_minus_a = U_outputs[0]      # U at x = -a
    U_plus_a = U_outputs[-1]      # U at x = a
    target_minus_a = U_outputs[1] # Assume some target or reference value at the next point
    target_plus_a = U_outputs[-2] # Assume some target or reference value at the previous point

    # Loss_Limit at the boundaries
    loss_Limit1 = tf.square(U_minus_a - target_minus_a)
    loss_Limit2 = tf.square(U_plus_a - target_plus_a)
    loss_Limit = loss_Limit1 + loss_Limit2

    # Neumann Boundary Condition Loss
    loss_BC = tf.square(U_x_a) + tf.square(U_x_neg_a)

    # Preventing translation of the wave solution
    mean_U = tf.reduce_mean(U_outputs)  # Mean across spatial domain as a simple approximation
    loss_Trans = tf.square(U_outputs[0] - mean_U) + tf.square(U_outputs[-1] - mean_U)

    # Combine all losses
    total_loss = loss_GSE + loss_Limit + loss_BC + loss_Trans
    return total_loss


# Training loop
def train(model_U, model_V, epochs):
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            t_points = tf.random.uniform((100, 1), minval=-a, maxval=a)
            x_points = tf.random.uniform((100, 1), minval=-a, maxval=a)
            loss = loss_function(model_U, model_V, t_points, x_points, a)
        gradients = tape.gradient(loss, model_U.trainable_variables + model_V.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_U.trainable_variables + model_V.trainable_variables))
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Start training
train(U_nn, V_nn, epochs)
