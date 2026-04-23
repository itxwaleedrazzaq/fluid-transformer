# Flexible Unified Information Dynamics (FLUID)
---
The repository contains the code of the FLUID transformer developed at Networked Intelligent Control (NIC) Lab at the University of Science and Technology of China. 

## FLUID Model (Plug & Play) Usage Example

```python
import tensorflow as tf
from FLUID import FLUID

inputs = tf.keras.Input(shape=(1, 1))

x = FLUID(
    d_model=64,                 # Dimension of the model of LAN
    num_heads=16,               # Number of attention heads of LAN
    num_layers=1,               # Number of stacked encoder/decoder layers
    ff_dim=32,                  # Dimension of the feed-forward network
    delta_t= 0.01,              # Time-step for the Liquid Attention
    euler_steps=5,              # Number of Euler steps for Liquid Attention
    topk=8,                     # Number of top-k attention interactions
    expansion_rate=2,           # Expansion factor for feed-forward layers
    use_sink_gate=True,         # Enable sink gate mechanism
    use_pairwise=False,         # disable top-k sparsity if True
    enable_hc=True,             # Enable hyper-connections if True, Otherwise -> Residual connections
    dynamic_hc=True,            # Enable Liquid hyper-connections if True, Otherwise -> Static
    dropout=0.0,                # Dropout rate
    max_len=1000,               # Maximum sequence length of positional encoder
    return_attention=False,     # Return attention weights if True
)(inputs)

x = tf.keras.layers.Activation('sigmoid')(x)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

```

## Experiments


### 1. Irreguler Time-series.

Training and evaluation of Spiral and E-MNIST.

* Code for Spiral available in: `Irregular_exp/Spiral_exp/`
* Code for E-MNIST available in: `Irregular_exp/mnist_exp/`

### 2. Long-Range Modeling (LRM).

Training and evaluation of long-range modeling.

* Code available in: `LRM_exps/`

### 3. Lane-Keeping of Autonomous Vehicle.

Training and evaluation of Autonomous-Vehicle lane-keeping.

* Code for Udacity available in: `AVs_exps/Udacity_exp/`
* Code for CarRacing available in: `AVs_exps/CarRacing_exp/`

### 4. Learning Physical Dynamics.

Learning of Physical dynamics modeling.

* Code available in: `rul_exps/`

### 5. Key Hyperparameter Measure.

Impact of key hyperparameters experiment.

* Code available in: `Hyperparam_exp/`


### 5. RunTime

Run-time & memory experiment.

* Code available in: `run_time_exp/`


