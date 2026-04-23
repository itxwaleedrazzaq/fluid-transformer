import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="FLUID_Transformer", name="HyperConnection")
class HyperConnection(tf.keras.layers.Layer):
    def __init__(self, d_model, expansion_rate, layer_id, dynamic_hc, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.expansion_rate = expansion_rate
        self.layer_id = layer_id
        self.dynamic_hc = dynamic_hc

        # ----- static beta -----
        self.static_beta = self.add_weight(
            name="static_beta",
            shape=(self.expansion_rate,),
            initializer="ones",
            trainable=True
        )

        # ----- static alpha -----
        init_alpha0 = tf.zeros((self.expansion_rate, 1))
        init_alpha0 = tf.tensor_scatter_nd_update(
            init_alpha0,
            [[self.layer_id % self.expansion_rate, 0]],
            [1.0]
        )
        eye = tf.eye(self.expansion_rate)
        init_alpha = tf.concat([init_alpha0, eye], axis=1)

        self.static_alpha = self.add_weight(
            name="static_alpha",
            shape=(self.expansion_rate, self.expansion_rate + 1),
            initializer=tf.keras.initializers.Constant(init_alpha.numpy()),
            trainable=True
        )

        if self.dynamic_hc:
            self.dynamic_hc_alpha_fn = self.add_weight(
                name="dynamic_alpha_fn",
                shape=(self.d_model, self.expansion_rate + 1),
                initializer="zeros",
                trainable=True
            )
            self.dynamic_hc_alpha_scale = self.add_weight(
                name="dynamic_alpha_scale",
                shape=(1,),
                initializer=tf.keras.initializers.Constant(0.01),
                trainable=True
            )
            self.dynamic_hc_beta_fn = self.add_weight(
                name="dynamic_beta_fn",
                shape=(self.d_model,),
                initializer="zeros",
                trainable=True
            )
            self.dynamic_hc_beta_scale = self.add_weight(
                name="dynamic_beta_scale",
                shape=(1,),
                initializer=tf.keras.initializers.Constant(0.01),
                trainable=True
            )
            self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inputs):
        x, x_o = inputs 
        x_exp = tf.tile(tf.expand_dims(x, axis=2), [1, 1, self.expansion_rate, 1]) #copying input

        if self.dynamic_hc:
            norm_x = self.layer_norm(x_o)
            wc = tf.tanh(tf.matmul(norm_x, self.dynamic_hc_alpha_fn)) * self.dynamic_hc_alpha_scale  # (B,L,expansion_rate+1)
            alpha = wc[:, :, None, :] + self.static_alpha[None, None, :, :] # (B,L,expansion_rate,expansion_rate+1)
        else:
            alpha = self.static_alpha[None, None, :, :]  # (1,1,expansion_rate,expansion_rate+1)

        if self.dynamic_hc:
            dc = tf.tanh(tf.matmul(norm_x, tf.reshape(self.dynamic_hc_beta_fn, [self.d_model, 1]))) * self.dynamic_hc_beta_scale
            beta = dc + self.static_beta[None, None, :]
        else:
            beta = self.static_beta[None, None, :]  # (1,1,expansion_rate)
            
        mix_x = tf.matmul(alpha[:, :, :, :self.expansion_rate], x_exp)
        beta_sum = tf.reduce_sum(beta, axis=-1, keepdims=True)  # (B, L, 1)
        new_x = x_o * beta_sum + tf.reduce_sum(mix_x, axis=2)

        return new_x
