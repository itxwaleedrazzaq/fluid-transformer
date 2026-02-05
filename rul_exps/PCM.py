import tensorflow as tf
from tensorflow.keras.losses import get as get_loss
from degradation_physics import dDdt


@tf.keras.utils.register_keras_serializable(package="FLUID_Transformer", name="PCM")
class PCM(tf.keras.Model):
    """
    Physics-Constraint Model (PCM)

    This model combines data-driven learning with physics-based constraints.
    A base neural network is trained using both supervised data loss and a
    physics-informed loss derived from degradation dynamics.

    Key characteristics:
    - Custom training and validation loops via `train_step` and `test_step`
    - Optional dynamic weighting between data and physics losses
    - Internal state variable updated according to a physics model
    """

    def __init__(
        self,
        model_fn,
        input_shape=(16,),
        loss_fn="mse",
        metrics_fn="mae",
        dynamic_weights=True,
        lmbda=0.5,
        model_name="PCNN",
    ):
        """
        Parameters
        ----------
        model_fn : callable
            Function that returns a compiled Keras model given an input shape.
        input_shape : tuple
            Shape of the input feature vector.
        loss_fn : str or callable
            Loss function used for supervised learning.
        metrics_fn : str or callable
            Metric specification (stored but not explicitly used).
        dynamic_weights : bool
            Whether to dynamically weight data and physics losses.
        lmbda : float
            Static physics loss weight when dynamic weighting is disabled.
        model_name : str
            Name of the model.
        """
        super().__init__(name=model_name)

        # Base neural network
        self._input_shape = input_shape  # rename to avoid conflict
        self.base_model = model_fn(input_shape)

        # Loss and metric definitions
        self.loss_fn = get_loss(loss_fn)
        self.metric_fn = get_loss(metrics_fn)

        # Loss weighting configuration
        self.dynamic_weights = dynamic_weights
        self.lmbda = lmbda

        # Internal physics state (non-trainable)
        self.initial_state = tf.constant(
            [0.0, 0.0, 1e-6, 0.0, 0.0, 0.0], dtype=tf.float32
        )
        self.state_variable = tf.Variable(
            self.initial_state, trainable=False, dtype=tf.float32
        )

        # Metric trackers for monitoring and callbacks
        self.data_loss_tracker = tf.keras.metrics.Mean(name="data_loss")
        self.physics_loss_tracker = tf.keras.metrics.Mean(name="physics_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")

    def compile(self, optimizer=None, **kwargs):
        """Compile the model with the given optimizer."""
        super().compile(optimizer=optimizer, **kwargs)

    def call(self, inputs, training=None):
        """Forward pass through the base model."""
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        """
        Custom training step.

        Expected data format:
            x = (X_features, t, T)
            y = (y_true, (Load, RPM))
        """
        x, (y_true, (Load_batch, RPM_batch)) = data
        X_batch, t_batch, T_batch = x

        # Ensure consistent dtype
        X_batch = tf.cast(X_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)
        T_batch = tf.cast(T_batch, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        Load_batch = tf.cast(Load_batch, tf.float32)
        RPM_batch = tf.cast(RPM_batch, tf.float32)

        batch_size = tf.shape(X_batch)[0]
        n_features = self._input_shape[-1] - 2  # last two inputs are t and T

        # Physics-only input: zeroed features with time and temperature
        zeros = tf.zeros((batch_size, n_features), dtype=tf.float32)
        X_physics = tf.concat([zeros, t_batch, T_batch], axis=1)

        with tf.GradientTape() as tape:
            # Data-driven prediction and loss
            y_pred = self.base_model(X_batch, training=True)
            data_loss = self.loss_fn(y_true, y_pred)

            # Physics-informed derivative: dD/dt
            with tf.GradientTape() as tape2:
                tape2.watch(X_physics)
                D_physics = self.base_model(X_physics, training=True)
            dD_dt_pred = tape2.gradient(D_physics, X_physics)[:, -2]

            # Physics model state update
            state_update = dDdt(
                self.state_variable, Load_batch, RPM_batch, T_batch
            )
            state_update_mean = tf.reduce_mean(state_update, axis=1)
            state_update_mean = tf.squeeze(state_update_mean)

            # Physics loss (D component only)
            physics_loss = self.loss_fn(
                self.state_variable[-1], dD_dt_pred
            )

            # Update internal state
            self.state_variable.assign(state_update_mean)

            # Dynamic or static loss weighting
            if self.dynamic_weights:
                phyx_weight = (
                    tf.math.reduce_std(self.state_variable)
                    / tf.math.reduce_std(X_batch)
                )
                data_weight = 1.0 - phyx_weight
                weights_raw = tf.stack([phyx_weight, data_weight])
                weights = tf.nn.softmax(weights_raw)
                phyx_weight = weights[0]
                data_weight = weights[1]
            else:
                phyx_weight = 1.0
                data_weight = 1.0

            total_loss = (
                data_weight * data_loss
                + phyx_weight * physics_loss
            )

        # Backpropagation
        grads = tape.gradient(
            total_loss, self.base_model.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(grads, self.base_model.trainable_variables)
        )

        # Update metrics
        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.mae_tracker.update_state(y_true, y_pred)

        return {
            "loss": self.total_loss_tracker.result(),
            "data_loss": self.data_loss_tracker.result(),
            "physics_loss": self.physics_loss_tracker.result(),
        }

    def test_step(self, data):
        """
        Custom validation step.

        Assumes validation data includes physics-related inputs.
        """
        x, (y_true, (Load_batch, RPM_batch)) = data
        X_batch, t_batch, T_batch = x

        # Ensure consistent dtype
        X_batch = tf.cast(X_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)
        T_batch = tf.cast(T_batch, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        Load_batch = tf.cast(Load_batch, tf.float32)
        RPM_batch = tf.cast(RPM_batch, tf.float32)

        batch_size = tf.shape(X_batch)[0]
        n_features = self._input_shape[-1] - 2

        zeros = tf.zeros((batch_size, n_features), dtype=tf.float32)
        X_physics = tf.concat([zeros, t_batch, T_batch], axis=1)

        # Data-driven prediction
        y_pred = self.base_model(X_batch, training=False)
        data_loss = self.loss_fn(y_true, y_pred)

        # Physics-informed derivative
        with tf.GradientTape() as tape2:
            tape2.watch(X_physics)
            D_physics = self.base_model(X_physics, training=False)
        dD_dt_pred = tape2.gradient(D_physics, X_physics)[:, -2]

        # Physics loss
        physics_loss = self.loss_fn(
            self.state_variable[-1], dD_dt_pred
        )

        # Loss weighting
        if self.dynamic_weights:
            phyx_weight = (
                tf.math.reduce_std(self.state_variable)
                / tf.math.reduce_std(X_batch)
            )
            data_weight = 1.0 - phyx_weight
            weights_raw = tf.stack([phyx_weight, data_weight])
            weights = tf.nn.softmax(weights_raw)
            phyx_weight = weights[0]
            data_weight = weights[1]
        else:
            phyx_weight = self.lmbda
            data_weight = 1.0

        total_loss = (
            data_weight * data_loss
            + phyx_weight * physics_loss
        )

        # Update validation metrics
        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.mae_tracker.update_state(y_true, y_pred)

        return {
            "loss": self.total_loss_tracker.result(),
            "mae": self.mae_tracker.result(),
        }

    def Score(self, y_true, y_pred):
        """
        Custom asymmetric scoring function.

        Penalizes late predictions more strongly than early predictions.
        Intended for evaluation, not training.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        error = y_pred - y_true

        mask_early = error < 0
        mask_late = error >= 0

        score_early = tf.reduce_sum(
            tf.exp(-error[mask_early] / 13.0) - 1.0
        )
        score_late = tf.reduce_sum(
            tf.exp(error[mask_late] / 10.0) - 1.0
        )

        return score_early + score_late

    def summary(self):
        """Display a summary of the base model."""
        dummy_input = tf.zeros((1, self._input_shape[-1]), dtype=tf.float32)
        self.base_model(dummy_input)
        return self.base_model.summary()

    def predict(self, X, verbose=1):
        """Run inference using the base model."""
        X = tf.cast(X, tf.float32)
        return self.base_model.predict(X,verbose=verbose)
    
    def get_config(self):
        """Return a dictionary of arguments needed to recreate the model."""
        config = super().get_config()
        config.update({
            "input_shape": self._input_shape,
            "loss_fn": self.loss_fn.__class__.__name__,
            "metrics_fn": self.metric_fn.__class__.__name__,
            "dynamic_weights": self.dynamic_weights,
            "lmbda": self.lmbda,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # You may need to pass a model_fn manually here because it's not serializable
        model_fn = custom_objects.get("model_fn") if custom_objects else None
        return cls(model_fn=model_fn, **config)
