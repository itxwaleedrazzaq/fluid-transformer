# Flexible Unified Information Dynamics (FLUID)
---
The repository contains the code of the FLUID transformer developed at Networked Intelligent Control (NIC) Lab at University of Science and Technology of China. 
## FLUID Model Usage Example
```python
import tensorflow as tf
from FLUID import FLUID

inputs = tf.keras.Input(shape=(1, 1))
# Apply the FLUID layer (custom Transformer-like block)
x = FLUID(
    d_model=64,                 # Dimension of the model of LAN
    num_heads=16,               # Number of attention heads of LAN
    num_layers=1,               # Number of stacked encoder/decoder layers
    ff_dim=32,                  # Dimension of the feed-forward network
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
```


    metrics=["mae"]
)
