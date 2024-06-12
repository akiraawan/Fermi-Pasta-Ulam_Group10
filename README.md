# FPUT Travelling Wave Neural Network 

This project implements a TensorFlow-based physics-informed neural network (PINN) model, `FpuWaveNet`, designed to find traveling wave solutions to the FPUT non-linear lattice system. The model utilizes custom learning rate schedules, a characteristic layer for wave speed adjustment, and specialized loss functions to optimize network performance. More information about the model, in addition to theory on the FPUT system and travelling wave, can be found in our project report. Examples of or findings can be found in the 'weakly non-linear' and 'strongly non-linear' folders containing subfolders named with the wave speed of the corresponding travelling wave solution.

The architecture is heavily inspired by the one presented in 'Traveling Wave Solutions of Partial Differential Equations via Neural Networks' (arXiv:2101.08520) by Cho et al.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install the necessary dependencies using:

```bash
pip install tensorflow numpy matplotlib
```

## Usage
- Ensure all dependencies are installed.
- Run the script to train the 'FpuWaveNet' model.
- Monitor the loss and speed history for training progress.
- Run and download the functions to animate the model's findings.
- We recommend storing the model weights via model.save_weights(path_to_file) and model.load_weights(path_to_file) for reproducibility. 

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Code Overview

### Learning Rate Schedule

The 'LearningRateSchedule' class defines a custom learning rate schedule that decays the learning rate over time.

```python
import tensorflow as tf

class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        return self.initial_learning_rate * (self.decay_rate ** (step / self.decay_steps))
```

### Characteristic Layer
The 'CharacteristicLayer' class defines a custom layer that adjusts the wave speed based on a learnable parameter snn. It performes a transformation using the travelling wave ansatz in order to force travelling wave solutions.

The initializer can be changed to allow travelling waves with various wave speeds.

```python
class CharacteristicLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CharacteristicLayer, self).__init__()
        self.snn = self.add_weight(shape=(), initializer=tf.random_uniform_initializer(minval=-1, maxval=1), trainable=True)

    def call(self, inputs):
        t = inputs[..., 0]
        indices = inputs[..., 1]
        t = tf.cast(t, tf.float32)
        indices = tf.cast(indices, tf.float32)
        return indices - self.snn * t
```

### FPU WaveNet Model
The 'FpuWaveNet' class defines the main model, comprising the characteristic layer and two separate dense networks for predicting q and p values. Here, q is the dispalcement of the particles from equilibrium and p is their momentum. 

```python
class FpuWaveNet(tf.keras.Model):
    def __init__(self):
        super(FpuWaveNet, self).__init__()
        self.characteristic_layer = CharacteristicLayer()
        self.q_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        self.p_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='tanh', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        z = self.characteristic_layer(inputs)
        z = tf.expand_dims(z, -1)
        q = self.q_layers(z)
        p = self.p_layers(z)
        return q, p
```

### Loss Function
The 'loss_function' computes the loss for training the model, considering residuals and boundary conditions. 
- Loss_GE is the standard L2 loss of the governance function, which determines the behaviour of the system; it ensures we get solutions to the FPUT system. 
- Loss_BC imposes the Neumann boudnary conditions which rewards the system for finding solutions with derivatives that decay towards the edges of the simulated boundary (this aligns with physical intuition as it prevents sharp changes). 
- Loss_Limit penalises the system if the particles towards the edges are not at rest. This is valid since the FPUT system has the particles at the 2 ends fixed, giving them both zero displacement and momentum. 
- Loss_Trans has the effect of reducing a family of travelling waves to a single principle candidate by centering it at the origin of the spatial domian. The motivation for this condition comes from the fact that a translation of a travelling wave is still a travelling wave (this is a phase shift).  

```python
def loss_function(model, t_points, indices, alpha):
    t_grid, idx_grid = tf.meshgrid(t_points, indices, indexing='ij')
    inputs = tf.stack([t_grid, idx_grid], axis=-1)
    q, p = model(inputs)
    N = indices.shape[0] // 2

    q_m1 = tf.roll(q, shift=1, axis=1)
    q_p1 = tf.roll(q, shift=-1, axis=1)
    dp_dt = q_p1 + q_m1 - 2 * q + alpha * (tf.pow(q_p1 - q, 2) - tf.pow(q - q_m1, 2))
    dq_dt = p

    residual_q = tf.reduce_mean(tf.square(dq_dt - p), axis=None)
    residual_p = tf.reduce_mean(tf.square(dp_dt - tf.roll(p, shift=-1, axis=1)), axis=None)
    Loss_GE = residual_q + residual_p

    Loss_BC = (tf.square(q[0, -N + 1] - q[0, -N]) +
               tf.square(q[0, N] - q[0, N - 1]) +
               tf.square(p[0, -N + 1] - p[0, -N]) +
               tf.square(p[0, N] - p[0, N - 1]))

    Loss_Limit = (tf.square(q[0, -N]) + tf.square(q[0, N]) +
                  tf.square(p[0, -N]) + tf.square(p[0, N]))

    Loss_Trans = tf.square(q[0, 0])

    return Loss_GE + Loss_BC + Loss_Limit + Loss_Trans
```

### Training Function
The 'train' function trains the model using separate optimizers for wave speed and network weights, recording loss and speed history. As shown, the learning rate decays by a factor of 0.9 every 1000 epochs.

```python
def train(model, epochs, N, t_range, alpha):
    indices = tf.range(-N, N + 1, dtype=tf.float32)

    lr_schedule_s = LearningRateSchedule(0.00001, 1000, 0.9)
    lr_schedule_qp = LearningRateSchedule(0.00001, 1000, 0.9)
    optimizer_s = tf.keras.optimizers.Adam(learning_rate=lr_schedule_s)
    optimizer_qp = tf.keras.optimizers.Adam(learning_rate=lr_schedule_qp)

    loss_history = []
    speed_history = []

    for epoch in range(epochs):
        t_points = tf.random.uniform(shape=(500,), minval=t_range[0], maxval=t_range[1], dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss = loss_function(model, t_points, indices, alpha)
        gradients = tape.gradient(loss, model.trainable_variables)

        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        grad_s = [grad for grad, var in zip(gradients, model.trainable_variables) if 'characteristic_layer' in var.name]
        var_s = [var for var in model.trainable_variables if 'characteristic_layer' in var.name]
        optimizer_s.apply_gradients(zip(grad_s, var_s))

        grad_qp = [grad for grad, var in zip(gradients, model.trainable_variables) if 'characteristic_layer' not in var.name]
        var_qp = [var for var in model.trainable_variables if 'characteristic_layer' not in var.name]
        optimizer_qp.apply_gradients(zip(grad_qp, var_qp))

        if epoch % 10 == 0:
            loss_value = loss.numpy()
            loss_history.append(loss_value)
            speed_history.append(model.characteristic_layer.snn.numpy())
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss_value}')

    return loss_history, speed_history
```

### Model Training
Train the model by specifying the number of particles, non-linear coefficient, time range, and epochs. We found 2000 epochs to result in the desired loss range of 10e-6. Traditionally, the FPUT system is used for small values of alpha << 1 for weakly non-linear effects.

```python
model = FpuWaveNet()
N = 32
alpha = 0.1
t_range = (-200, 200)
n_epochs = 1000

loss_history, speed_history = train(model, n_epochs, N, t_range, alpha)
```

Functions to plot the loss history, wave speed estimation convergence, and create animations of the travlling waves are provided.
