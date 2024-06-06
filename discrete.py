import tensorflow as tf
import numpy as np

class CharacteristicLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CharacteristicLayer, self).__init__()
        self.snn = self.add_weight(shape=(), initializer="random_normal", trainable=True)

    def call(self, inputs):
        # Assuming last dimension has two elements: t and indices
        t = inputs[..., 0]
        indices = inputs[..., 1]
        t = tf.cast(t, tf.float32)
        indices = tf.cast(indices, tf.float32)
        return indices - self.snn * t

class FpuWaveNet(tf.keras.Model):
    def __init__(self):
        super(FpuWaveNet, self).__init__()
        self.characteristic_layer = CharacteristicLayer()
        self.q_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(1)
        ])
        self.p_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        z = self.characteristic_layer(inputs)
        z = tf.expand_dims(z, -1)  # Ensure z has the right shape for Dense layers
        q = self.q_layers(z)
        p = self.p_layers(z)
        return q, p

def update_boundaries(tensor):
    return tensor

def loss_function(model, t_points, indices, alpha):
    t_grid, idx_grid = tf.meshgrid(t_points, indices, indexing='ij')
    inputs = tf.stack([t_grid, idx_grid], axis=-1)
    q, p = model(inputs)

    q_m1 = tf.roll(q, shift=1, axis=1)
    q_p1 = tf.roll(q, shift=-1, axis=1)
    dp_dt = q_p1 + q_m1 - 2 * q + alpha * (tf.pow(q_p1 - q, 3) - tf.pow(q - q_m1, 3))
    dq_dt = p

    # Update boundary conditions
    dp_dt = update_boundaries(dp_dt)
    dq_dt = update_boundaries(dq_dt)

    residual_q = tf.reduce_mean(tf.square(dq_dt - p))
    residual_p = tf.reduce_mean(tf.square(dp_dt - tf.roll(p, shift=-1, axis=1)))

    return residual_q + residual_p

def train(model, epochs, N, t_range, alpha):
    t_points = tf.linspace(float(t_range[0]), float(t_range[1]), 200)
    indices = tf.range(-N, N + 1, dtype=tf.float32)
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = loss_function(model, t_points, indices, alpha)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Initialize and train the model
model = FpuWaveNet()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
N = 10  # Number of particles on each side of the lattice
alpha = 0.1  # Specific to FPU problem dynamics
t_range = (-10, 10)  # Simulation time range

train(model, 1000, N, t_range, alpha)


####################
### For plotting ###
####################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf

# Assuming model is defined and loaded correctly elsewhere
# Constants
a = 10  # Spatial domain endpoint
N = 10  # Number of particles
time_duration = 1000  # Time for simulation in arbitrary units
time_steps = 100  # Number of time steps in the animation

# Generate space and time points for prediction
indices = np.linspace(-N, N, 2 * N + 1)  # particle indices, discretized model
time_values = np.linspace(0, time_duration, time_steps)  # Time domain for the animation

# Prepare input for predictions
test_inputs = np.array([[t, idx] for t in time_values for idx in indices])

# Predict using the trained model
q_predictions, p_predictions = model(test_inputs.astype(np.float32))
q_predictions = q_predictions.numpy().reshape((time_steps, -1))  # Reshape q predictions
p_predictions = p_predictions.numpy().reshape((time_steps, -1))  # Reshape p predictions

# Create the animation for 'q' using dots
fig, ax = plt.subplots()
line, = ax.plot(indices, q_predictions[0, :], 'o', color='b', label='q(t)', lw=2)  # Use 'o' for dots

def update(frame):
    # Update the data with dots
    line.set_ydata(q_predictions[frame, :])
    return line,

ani = FuncAnimation(fig, update, frames=time_steps, blit=True)
ax.set_xlim([indices.min(), indices.max()])
ax.set_ylim([q_predictions.min(), q_predictions.max()])
ax.set_xlabel('Particle Index')
ax.set_ylabel('q(t)')
ax.set_title('Discrete Traveling Wave Solution Over Time')

# Save the animation
ani.save('discrete_traveling_wave.mp4', writer='ffmpeg', fps=15)

plt.show()
