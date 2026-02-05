import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="PCHC_LANFormer", name="LAN")
class LAN(tf.keras.layers.Layer):

    def __init__(
        self,
        d_model : int,
        num_heads : int,
        topk : int = 8,
        euler_steps : int = 2,
        use_sink_gate : bool = True,
        activation = None,
        tau_epsilon : float = 1e-6,
        delta_t : float = 0.01,
        dropout : float = 0.0,
        use_bias : bool = True,
        use_pairwise : bool = False,
        return_attention : bool = False,
        return_sequences : bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)


        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.topk = int(topk)
        self.euler_steps = int(euler_steps)
        self.delta_t = float(delta_t)
        self.use_sink_gate = use_sink_gate
        self.tau_epsilon = tau_epsilon
        self.dropout = dropout
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.use_pairwise = use_pairwise
        self.return_attention = return_attention
        self.return_sequences = return_sequences

        #linear projection
        self.q_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name='q_proj')
        self.k_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name='k_proj')
        self.v_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name='v_proj')
        self.out_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name='out_proj')

        #recurrent-gating
        self.gate_in = tf.keras.layers.LSTM(self.d_model, use_bias=use_bias, return_sequences=True, name='gate_in')
        self.gate_out = tf.keras.layers.Dense(1, use_bias=use_bias, name='gate_out')

        #sink gate
        self.sink_gate = tf.keras.layers.Dense(self.d_model, activation='sigmoid', use_bias=True, name="gate")

        self.attn_dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        super().build(input_shape)

    def split_heads(self, x):
        '''
        Reshape input tensor into multiple attention heads.

        Args:
          x: [B, T, d_model]
        Returns:
          [B, num_heads, T, depth]
        '''
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        x = tf.reshape(x, (B, T, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        '''
        Reverse split_heads: combine multi-head output.

        Args:
          x: [B, num_heads, T, depth]
        Returns:
          [B, T, d_model]
        '''
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        return tf.reshape(x, (B, T, self.d_model))

    def pairwise_concat(self, q, k):
        '''
        Concatenate each query with each key across sequence positions.

        Args:
          q: [B, H, Tq, D]
          k: [B, H, Tk, D]
        Returns:
          [B, H, Tq, Tk, 2D]
        '''
        q_exp = tf.expand_dims(q, axis=3)
        k_exp = tf.expand_dims(k, axis=2)
        q_tile = tf.tile(q_exp, [1, 1, 1, tf.shape(k)[2], 1])
        k_tile = tf.tile(k_exp, [1, 1, tf.shape(q)[2], 1, 1])
        return tf.concat([q_tile, k_tile], axis=-1)

    
    def sparse_topk_pairwise(self, q, k, K=None):
        '''
        Compute pairwise dot products between q and k, and select the top-K keys per query.
        Args:
            q: Query tensor of shape [B, H, Tq, D]
            k: Key tensor of shape [B, H, Tk, D]
            K: Optional override for top-k value
        Returns:
            topk_pairs: Concatenated [q, selected_k] pairs, shape [B, H, Tq, K, 2D]
            topk_idx: Indices of selected keys, shape [B, H, Tq, K]
        '''
        if K is None:
            K = self.topk
        scores = tf.einsum("bhqd,bhkd->bhqk", q, k)
        Tk = tf.shape(k)[2]
        K_eff = tf.minimum(K, Tk)
        _, topk_idx = tf.math.top_k(scores, k=K_eff)
        k_gathered = tf.gather(k, topk_idx, batch_dims=2, axis=2)
        q_tiled = tf.tile(tf.expand_dims(q, axis=3), [1, 1, 1, K_eff, 1])
        topk_pairs = tf.concat([q_tiled, k_gathered], axis=-1)
        return topk_pairs, topk_idx
    
    def compute_phi_tau(self, q, k):
        '''
        Compute phi (target-content gate), tau (time constant gate) and for each query-key pair.

        Args:
          q, k: [B, H, T, D]
        Returns:
          phi: [B, H, Tq, Tk]
          tau: [B, H, Tq, Tk]
        '''
        if self.use_pairwise:
            pair = self.pairwise_concat(q, k)
            idx = None
        else:
            pair, idx = self.sparse_topk_pairwise(q, k)
        B, H, Tq, Tk, F = tf.shape(pair)[0], tf.shape(pair)[1], tf.shape(pair)[2], tf.shape(pair)[3], tf.shape(pair)[4]

        pair = tf.reshape(pair, (B * H * Tq, Tk, F))

        x = self.gate_in(pair)
        gate_raw = self.gate_out(x)
        gate_raw = tf.reshape(gate_raw, (-1, H, Tq, Tk))
        phi = tf.nn.relu(gate_raw) # Compute phi
        tau = tf.nn.softplus(gate_raw) + self.tau_epsilon  # Compute tau

        return phi, tau, idx

    def call(self, inputs, mask=None, training=None):
        '''
        Forward pass for LAN.

        Args:
          inputs: can be
            - single tensor (self-attention)
            - tuple of 2 tensors ((q, k), v)
            - tuple of 3 tensors (q, k, v)
          mask: optional attention mask
          training: flag for dropout
        Returns:
          Attention output (and optionally weights)
        '''
        if isinstance(mask, (list, tuple)):
            mask = mask[0] if len(mask) > 0 else None
            if mask is None or (hasattr(mask, 'shape') and mask.shape == ()):
                mask = None

        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            x, v_in = inputs
            q_in = k_in = x
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            q_in, k_in, v_in = inputs
        else:
            q_in = k_in = v_in = inputs

        # Linear projections
        q = self.q_dense(q_in)
        k = self.k_dense(k_in)
        v = self.v_dense(v_in)

        # Split heads
        qh = self.split_heads(q)
        kh = self.split_heads(k)
        vh = self.split_heads(v)

        # Compute phi, tau
        phi, tau, topk_idx = self.compute_phi_tau(qh, kh)
        dt = tf.cast(self.delta_t, phi.dtype)

        # Euler Stability condition: dt <= 1 / sup(tau)
        tau_max = tf.reduce_max(tau)
        dt_max = 1.0 / (tau_max + 1e-12)
        dt = tf.minimum(dt, dt_max)

        # Euler integration for attention logits
        a = tf.zeros_like(phi)
        for _ in range(self.euler_steps):
            increment = dt * (-tau * a + phi)
            a = a + increment
        attn_logits = a
        # Apply mask
        if mask is not None:
            mask = tf.cast(mask, attn_logits.dtype)
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)  # [batch, 1, 1, seq_len]
            very_neg = tf.constant(-1e9, dtype=attn_logits.dtype)
            attn_logits = attn_logits + (1.0 - mask) * very_neg

        
        # Attention weights
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        # Integrated output
        if self.use_pairwise:
            output = tf.matmul(attn_weights, vh)
        else:
            vh_topk = tf.gather(vh, topk_idx, batch_dims=2, axis=2)
            output = tf.reduce_sum(attn_weights[..., tf.newaxis] * vh_topk,axis=3)

        # Combine heads and final projection
        combined = self.combine_heads(output)
        # Sink gate modulation
        if self.use_sink_gate:
            sink_gate_values = self.sink_gate(q_in)  
            combined = combined * sink_gate_values

        out = self.activation(self.out_dense(combined))

        result = out if self.return_sequences else out[:, -1, :]
        if self.return_attention:
            return result, attn_weights
        return result


    def get_config(self):
        '''Return layer configuration for serialization.'''
        cfg = super().get_config()
        cfg.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'tau_epsilon': self.tau_epsilon,
            'topk': self.topk,
            'euler_steps': self.euler_steps,
            'delta_t': self.delta_t,
            'use_sink_gate': self.use_sink_gate,
            'dropout': self.dropout,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'use_pairwise': self.use_pairwise,
            'return_attention': self.return_attention,
            'return_sequences': self.return_sequences
        })
        return cfg
