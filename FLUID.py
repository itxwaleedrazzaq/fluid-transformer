import tensorflow as tf
import numpy as np  
from liquid_attention import LAN
from hyperconnections import HyperConnection


@tf.keras.utils.register_keras_serializable(package="FLUID_Transformer", name="FLUID")
class FLUID(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model, num_heads, ff_dim,
        topk : int = 8,
        expansion_rate : int = 4,
        use_sink_gate  : bool = True,
        use_pairwise : bool = False,
        enable_hc  : bool = True,
        dynamic_hc : bool = True,
        num_layers : int = 1,  # support multiple encoder,decoder layers for better generalization
        dropout : float = 0.0,
        max_len : int = 1000,  # For positional encoding
        return_attention : bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.hc = enable_hc
        self.return_attention = return_attention

        # Input projection (necessary for time-series if input features != d_model)
        self.embedding = tf.keras.layers.Dense(d_model)

        # Positional encoding (fixed for time-series order awareness)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Multi-layer encoder and decoder
        self.encoders = [
            Encoder(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                topk = topk,
                use_sink_gate=use_sink_gate,
                use_pairwise=use_pairwise,
                expansion_rate=expansion_rate,
                enable_hc=enable_hc,
                dynamic_hc=dynamic_hc,
                dropout=dropout,
                return_attention=return_attention,
            )
            for _ in range(num_layers)
        ]

        self.decoders = [
            Decoder(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                topk = topk,
                use_sink_gate=use_sink_gate,
                use_pairwise=use_pairwise,
                expansion_rate=expansion_rate,
                enable_hc=enable_hc,
                dynamic_hc=dynamic_hc,
                dropout=dropout,
                return_attention=return_attention,
            )
            for _ in range(num_layers)
        ]


    def call(self, x, training=None):
        # Project input to d_model if needed
        x = self.embedding(x)
        # Add positional encoding for time-series generalization
        x = self.pos_encoder(x)
        # Multi-layer encoder
        enc_out = x
        # gather attention weights
        enc_weights = []
        dec_weights = []
        for encoder in self.encoders:
            enc_out, enc_weight = encoder(enc_out, training=training)
            enc_weights.append(enc_weight)

        # Multi-layer decoder (using input sequence for both; adjust if needed for autoregressive)
        dec_out = x
        for decoder in self.decoders:
            dec_out, dec_weight = decoder(dec_out, enc_out, training=training)
            dec_weights.append(dec_weight)
        if self.return_attention:
            return dec_out, {'encoder_attention': enc_weights, 'decoder_attention': dec_weights}
        else:
            return dec_out


@tf.keras.utils.register_keras_serializable(package="FLUID_Transformer", name="PositionalEncoding")
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

        # Precompute positional encodings using numpy
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, ...]  # Add batch dimension

        # Store as non-trainable weight to avoid scope issues
        self.pe = self.add_weight(
            name='pe',
            shape=(1, max_len, d_model),
            initializer=tf.constant_initializer(pe),
            trainable=False
        )

    def call(self, x):
        # Add positional encoding to inputs
        x = x + self.pe[:, :tf.shape(x)[1], :]
        return x

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'd_model': self.d_model,
            'max_len': self.max_len
        })
        return config

@tf.keras.utils.register_keras_serializable(package="FLUID_Transformer", name="Encoder")
class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, topk=8, enable_hc=True, 
                 dynamic_hc=True, use_sink_gate=True, use_pairwise=False, expansion_rate=4,dropout=0.1,return_attention=False):
        super().__init__()

        self.hc = enable_hc
        self.return_attention = return_attention

        # Self-attention 
        self.attn = LAN(d_model=d_model,
                         num_heads=num_heads,
                         topk= topk,
                         use_sink_gate=use_sink_gate, 
                         use_pairwise=use_pairwise,
                         return_sequences=True,
                         return_attention=return_attention)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.enc_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.hyper_residual1 = HyperConnection(
            d_model=d_model, 
            expansion_rate=expansion_rate, 
            dynamic_hc=dynamic_hc,
            layer_id=1
        )

        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model),
        ])
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self.enc_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.hyper_residual2 = HyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            dynamic_hc=dynamic_hc,
            layer_id=2
        )

    def call(self, x, training=None):

        # Self-attention 
        if self.return_attention:
            attn_out, attn_weights = self.attn(x)
        else:
            attn_out = self.attn(x)
            attn_weights = None
        attn_out = self.drop1(attn_out, training=training)
        if self.hc:
            x = self.hyper_residual1([x, attn_out])
        else:
            x = x + attn_out 
        x = self.enc_norm1(x)

        # Feed-forward network
        ffn_out = self.ffn(x)
        ffn_out = self.drop2(ffn_out, training=training)
        if self.hc:
            x = self.hyper_residual2([x, ffn_out])
        else:
            x = x + ffn_out
        x = self.enc_norm2(x)

        return x,attn_weights

@tf.keras.utils.register_keras_serializable(package="FLUID_Transformer", name="Decoder")
class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, topk=8, enable_hc=True, 
                 dynamic_hc=True, use_sink_gate=True, use_pairwise=False, expansion_rate=4,dropout=0.1, return_attention=False):
        super().__init__()

        self.hc = enable_hc
        self.return_attention = return_attention

        #self-attention 
        self.self_attn = LAN(d_model=d_model,
                         num_heads=num_heads, 
                         topk= topk,
                         use_sink_gate=use_sink_gate,
                         use_pairwise=use_pairwise,
                         return_sequences=True,
                         return_attention=return_attention)
        self.dec_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.hyper_residual1 = HyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            dynamic_hc=dynamic_hc,
            layer_id=1
        )

        #cross-attention
        self.cross_attn = LAN(d_model=d_model,
                         num_heads=num_heads, 
                         topk= topk,
                         use_sink_gate=use_sink_gate,
                         use_pairwise=use_pairwise,
                         return_sequences=True,
                         return_attention=return_attention)
        self.dec_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self.hyper_residual2 = HyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            dynamic_hc=dynamic_hc,
            layer_id=2
        )

        #feed-forward network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model),
        ])
        self.drop3 = tf.keras.layers.Dropout(dropout)
        self.dec_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.hyper_residual3 = HyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            dynamic_hc=dynamic_hc,
            layer_id=3
        )

    def call(self, x, enc_out, training=None):

        # self-attention
        if self.return_attention:
            sa, sa_weights = self.self_attn(x)
        else:
            sa = self.self_attn(x)
            sa_weights = None

        sa = self.drop1(sa, training=training)
        if self.hc:
            x = self.hyper_residual1([x, sa])
        else:
            x = x + sa
        x = self.dec_norm1(x)

        # Cross-attention: decoder attending over encoder output
        if self.return_attention:
            ca, ca_weights = self.cross_attn([x, enc_out, enc_out])
        else:
            ca = self.cross_attn([x, enc_out, enc_out])
            ca_weights = None
        ca = self.drop2(ca, training=training)
        if self.hc:
            x = self.hyper_residual2([x, ca])
        else:
            x = x + ca  
        x = self.dec_norm2(x)

        # Feed-forward network 
        ffn_out = self.ffn(x)
        ffn_out = self.drop3(ffn_out, training=training)
        if self.hc:
            x = self.hyper_residual3([x, ffn_out])
        else:
            x = x + ffn_out
        x = self.dec_norm3(x)
        dec_weights = {
            "self_attention": sa_weights,
            "cross_attention": ca_weights
        }

        return x, dec_weights
