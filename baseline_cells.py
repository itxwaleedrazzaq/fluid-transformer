'''
The code is taken from https://github.com/mlech26l/ode-lstms for testing purposes.
'''
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
import numpy as np
from typing import Optional
from einops import rearrange
import math


class CTRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, method, num_unfolds=None, tau=1, **kwargs):
        self.fixed_step_methods = {
            "euler": self.euler,
            "heun": self.heun,
            "rk4": self.rk4,
        }
        allowed_methods = ["euler", "heun", "rk4", "dopri5"]
        if not method in allowed_methods:
            raise ValueError(
                "Unknown ODE solver '{}', expected one of '{}'".format(
                    method, allowed_methods
                )
            )
        if method in self.fixed_step_methods.keys() and num_unfolds is None:
            raise ValueError(
                "Fixed-step ODE solver requires argument 'num_unfolds' to be specified!"
            )
        self.units = units
        self.state_size = units
        self.num_unfolds = num_unfolds
        self.method = method
        self.tau = tau
        super(CTRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units), initializer="glorot_uniform", name="kernel"
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer="orthogonal",
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units), initializer=tf.keras.initializers.Zeros(), name="bias"
        )
        self.scale = self.add_weight(
            shape=(self.units),
            initializer=tf.keras.initializers.Constant(1.0),
            name="scale",
        )
        if self.method == "dopri5":
            # Only load tfp packge if it is really needed
            import tensorflow_probability as tfp

            # We don't need the most precise solver to speed up training
            self.solver = tfp.math.ode.DormandPrince(
                rtol=0.01,
                atol=1e-04,
                first_step_size=0.01,
                safety_factor=0.8,
                min_step_size_factor=0.1,
                max_step_size_factor=10.0,
                max_num_steps=None,
                make_adjoint_solver_fn=None,
                validate_args=False,
                name="dormand_prince",
            )
        self.built = True

    def call(self, inputs, states):
        hidden_state = states[0]
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        if self.method == "dopri5":
            # Only load tfp packge if it is really needed
            import tensorflow_probability as tfp

            if not type(elapsed) == float:
                batch_dim = tf.shape(elapsed)[0]
                elapsed = tf.reshape(elapsed, [batch_dim])

                idx = tf.argsort(elapsed)
                solution_times = tf.gather(elapsed, idx)
            else:
                solution_times = elapsed
            hidden_state = states[0]
            res = self.solver.solve(
                ode_fn=self.dfdt_wrapped,
                initial_time=0,
                initial_state=hidden_state,
                solution_times=solution_times,  # tfp.math.ode.ChosenBySolver(elapsed),
                constants={"input": inputs},
            )
            if not type(elapsed) == float:
                i2 = tf.stack([idx, tf.range(batch_dim)], axis=1)
                hidden_state = tf.gather_nd(res.states, i2)
            else:
                hidden_state = res.states[-1]
        else:
            delta_t = elapsed / self.num_unfolds
            method = self.fixed_step_methods[self.method]
            for i in range(self.num_unfolds):
                hidden_state = method(inputs, hidden_state, delta_t)
        return hidden_state, [hidden_state]

    def dfdt_wrapped(self, t, y, **constants):
        inputs = constants["input"]
        hidden_state = y
        return self.dfdt(inputs, hidden_state)

    def dfdt(self, inputs, hidden_state):
        h_in = tf.matmul(inputs, self.kernel)
        h_rec = tf.matmul(hidden_state, self.recurrent_kernel)
        dh_in = self.scale * tf.nn.tanh(h_in + h_rec + self.bias)
        if self.tau > 0:
            dh = dh_in - hidden_state * self.tau
        else:
            dh = dh_in
        return dh

    def euler(self, inputs, hidden_state, delta_t):
        dy = self.dfdt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def heun(self, inputs, hidden_state, delta_t):
        k1 = self.dfdt(inputs, hidden_state)
        k2 = self.dfdt(inputs, hidden_state + delta_t * k1)
        return hidden_state + delta_t * 0.5 * (k1 + k2)

    def rk4(self, inputs, hidden_state, delta_t):
        k1 = self.dfdt(inputs, hidden_state)
        k2 = self.dfdt(inputs, hidden_state + k1 * delta_t * 0.5)
        k3 = self.dfdt(inputs, hidden_state + k2 * delta_t * 0.5)
        k4 = self.dfdt(inputs, hidden_state + k3 * delta_t)

        return hidden_state + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class LSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units)
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        super(LSTMCell, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_shape = (input_shape[0][-1] + input_shape[1][-1],)

        self.input_kernel = self.add_weight(
            shape=(input_shape[-1], 4 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )

        self.built = True

    def call(self, inputs, states):
        cell_state, output_state = states
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            inputs = tf.concat([inputs[0], inputs[1]], axis=-1)

        z = (
            tf.matmul(inputs, self.input_kernel)
            + tf.matmul(output_state, self.recurrent_kernel)
            + self.bias
        )
        i, ig, fg, og = tf.split(z, 4, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg + 1.0)
        output_gate = tf.nn.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = tf.nn.tanh(new_cell) * output_gate

        return output_state, [new_cell, output_state]


class ODELSTM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units)
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        self.ctrnn = CTRNNCell(self.units, num_unfolds=4, method="euler")
        super(ODELSTM, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.ctrnn.build([self.units])
        self.input_kernel = self.add_weight(
            shape=(input_dim, 4 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )

        self.built = True

    def call(self, inputs, states):
        cell_state, ode_state = states
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        z = (
            tf.matmul(inputs, self.input_kernel)
            + tf.matmul(ode_state, self.recurrent_kernel)
            + self.bias
        )
        i, ig, fg, og = tf.split(z, 4, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg + 3.0)
        output_gate = tf.nn.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        ode_input = tf.nn.tanh(new_cell) * output_gate

        # Implementation choice on how to parametrize ODE component
        ode_output, new_ode_state = self.ctrnn.call([ode_input, elapsed], [ode_state])
        # ode_output, new_ode_state = self.ctrnn.call([ode_input, elapsed], [ode_input])

        return ode_output, [new_cell, new_ode_state[0]]


class CTGRU(tf.keras.layers.Layer):
    # https://arxiv.org/abs/1710.04110
    def __init__(self, units, M=8, **kwargs):
        self.units = units
        self.M = M
        self.state_size = units * self.M

        # Pre-computed tau table (as recommended in paper)
        self.ln_tau_table = np.empty(self.M)
        self.tau_table = np.empty(self.M)
        tau = 1.0
        for i in range(self.M):
            self.ln_tau_table[i] = np.log(tau)
            self.tau_table[i] = tau
            tau = tau * (10.0 ** 0.5)

        super(CTGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.retrieval_layer = tf.keras.layers.Dense(
            self.units * self.M, activation=None
        )
        self.detect_layer = tf.keras.layers.Dense(self.units, activation="tanh")
        self.update_layer = tf.keras.layers.Dense(self.units * self.M, activation=None)
        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        batch_dim = tf.shape(inputs)[0]

        # States is actually 2D
        h_hat = tf.reshape(states[0], [batch_dim, self.units, self.M])
        h = tf.reduce_sum(h_hat, axis=2)
        states = None  # Set state to None, to avoid misuses (bugs) in the code below

        # Retrieval
        fused_input = tf.concat([inputs, h], axis=-1)
        ln_tau_r = self.retrieval_layer(fused_input)
        ln_tau_r = tf.reshape(ln_tau_r, shape=[batch_dim, self.units, self.M])
        sf_input_r = -tf.square(ln_tau_r - self.ln_tau_table)
        rki = tf.nn.softmax(logits=sf_input_r, axis=2)

        q_input = tf.reduce_sum(rki * h_hat, axis=2)
        reset_value = tf.concat([inputs, q_input], axis=1)
        qk = self.detect_layer(reset_value)
        qk = tf.reshape(qk, [batch_dim, self.units, 1])  # in order to broadcast

        ln_tau_s = self.update_layer(fused_input)
        ln_tau_s = tf.reshape(ln_tau_s, shape=[batch_dim, self.units, self.M])
        sf_input_s = -tf.square(ln_tau_s - self.ln_tau_table)
        ski = tf.nn.softmax(logits=sf_input_s, axis=2)

        # Now the elapsed time enters the state update
        base_term = (1 - ski) * h_hat + ski * qk
        exp_term = tf.exp(-elapsed / self.tau_table)
        exp_term = tf.reshape(exp_term, [batch_dim, 1, self.M])
        h_hat_next = base_term * exp_term

        # Compute new state
        h_next = tf.reduce_sum(h_hat_next, axis=2)
        h_hat_next_flat = tf.reshape(h_hat_next, shape=[batch_dim, self.units * self.M])
        return h_next, [h_hat_next_flat]


class VanillaRNN(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units

        super(VanillaRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self._layer = tf.keras.layers.Dense(self.units, activation="tanh")
        self._out_layer = tf.keras.layers.Dense(self.units, activation=None)
        self._tau = self.add_weight(
            "tau",
            shape=(self.units),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.1),
        )
        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        fused_input = tf.concat([inputs, states[0]], axis=-1)
        new_states = self._out_layer(self._layer(fused_input)) - elapsed * self._tau

        return new_states, [new_states]


class BidirectionalRNN(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units, units)

        self.ctrnn = CTRNNCell(self.units, num_unfolds=4, method="euler")
        self.lstm = LSTMCell(units=self.units)

        super(BidirectionalRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        self._out_layer = tf.keras.layers.Dense(self.units, activation=None)
        fused_dim = ((input_dim + self.units,), (1,))
        self.lstm.build(fused_dim)
        self.ctrnn.build(fused_dim)
        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        lstm_state = [states[0], states[1]]
        lstm_input = [tf.concat([inputs, states[2]], axis=-1), elapsed]
        ctrnn_state = [states[2]]
        ctrnn_input = [tf.concat([inputs, states[1]], axis=-1), elapsed]

        lstm_out, new_lstm_states = self.lstm.call(lstm_input, lstm_state)
        ctrnn_out, new_ctrnn_state = self.ctrnn.call(ctrnn_input, ctrnn_state)

        fused_output = lstm_out + ctrnn_out
        return (
            fused_output,
            [new_lstm_states[0], new_lstm_states[1], new_ctrnn_state[0]],
        )


class GRUD(tf.keras.layers.Layer):
    # Implemented according to
    # https://www.nature.com/articles/s41598-018-24271-9.pdf
    # without the masking

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(GRUD, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self._reset_gate = tf.keras.layers.Dense(
            self.units, activation="sigmoid", kernel_initializer="glorot_uniform"
        )
        self._detect_signal = tf.keras.layers.Dense(
            self.units, activation="tanh", kernel_initializer="glorot_uniform"
        )
        self._update_gate = tf.keras.layers.Dense(
            self.units, activation="sigmoid", kernel_initializer="glorot_uniform"
        )
        self._d_gate = tf.keras.layers.Dense(
            self.units, activation="relu", kernel_initializer="glorot_uniform"
        )

        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        dt = self._d_gate(elapsed)
        gamma = tf.exp(-dt)
        h_hat = states[0] * gamma

        fused_input = tf.concat([inputs, h_hat], axis=-1)
        rt = self._reset_gate(fused_input)
        zt = self._update_gate(fused_input)

        reset_value = tf.concat([inputs, rt * h_hat], axis=-1)
        h_tilde = self._detect_signal(reset_value)

        # Compute new state
        ht = zt * h_hat + (1.0 - zt) * h_tilde

        return ht, [ht]


class PhasedLSTM(tf.keras.layers.Layer):
    # Implemented according to
    # https://papers.nips.cc/paper/6310-phased-lstm-accelerating-recurrent-network-training-for-long-or-event-based-sequences.pdf

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units)
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        super(PhasedLSTM, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.input_kernel = self.add_weight(
            shape=(input_dim, 4 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )
        self.tau = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Zeros(), name="tau"
        )
        self.ron = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Zeros(), name="ron"
        )
        self.s = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Zeros(), name="s"
        )

        self.built = True

    def call(self, inputs, states):
        cell_state, hidden_state = states
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        # Leaky constant taken fromt he paper
        alpha = 0.001
        # Make sure these values are positive
        tau = tf.nn.softplus(self.tau)
        s = tf.nn.softplus(self.s)
        ron = tf.nn.softplus(self.ron)

        phit = tf.math.mod(elapsed - s, tau) / tau
        kt = tf.where(
            tf.less(phit, 0.5 * ron),
            2 * phit * ron,
            tf.where(tf.less(phit, ron), 2.0 - 2 * phit / ron, alpha * phit),
        )

        z = (
            tf.matmul(inputs, self.input_kernel)
            + tf.matmul(hidden_state, self.recurrent_kernel)
            + self.bias
        )
        i, ig, fg, og = tf.split(z, 4, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg + 1.0)
        output_gate = tf.nn.sigmoid(og)

        c_tilde = cell_state * forget_gate + input_activation * input_gate
        c = kt * c_tilde + (1.0 - kt) * cell_state

        h_tilde = tf.nn.tanh(c_tilde) * output_gate
        h = kt * h_tilde + (1.0 - kt) * hidden_state

        return h, [c, h]


class GRUODE(tf.keras.layers.Layer):
    # Implemented according to
    # https://arxiv.org/pdf/1905.12374.pdf
    # without the Bayesian stuff

    def __init__(self, units, num_unfolds=4, **kwargs):
        self.units = units
        self.num_unfolds = num_unfolds
        self.state_size = units
        super(GRUODE, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        self._reset_gate = tf.keras.layers.Dense(
            self.units,
            activation="sigmoid",
            bias_initializer=tf.constant_initializer(1),
        )
        self._detect_signal = tf.keras.layers.Dense(self.units, activation="tanh")
        self._update_gate = tf.keras.layers.Dense(self.units, activation="sigmoid")

        self.built = True

    def _dh_dt(self, inputs, states):
        fused_input = tf.concat([inputs, states], axis=-1)
        rt = self._reset_gate(fused_input)
        zt = self._update_gate(fused_input)

        reset_value = tf.concat([inputs, rt * states], axis=-1)
        gt = self._detect_signal(reset_value)

        # Compute new state
        dhdt = (1.0 - zt) * (gt - states)
        return dhdt

    def euler(self, inputs, hidden_state, delta_t):
        dy = self._dh_dt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        delta_t = elapsed / self.num_unfolds
        hidden_state = states[0]
        for i in range(self.num_unfolds):
            hidden_state = self.euler(inputs, hidden_state, delta_t)
        return hidden_state, [hidden_state]

        return ht, [ht]


class HawkLSTMCell(tf.keras.layers.Layer):
    # https://papers.nips.cc/paper/7252-the-neural-hawkes-process-a-neurally-self-modulating-multivariate-point-process.pdf
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units, units)  # state is a tripple
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        super(HawkLSTMCell, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        self.input_kernel = self.add_weight(
            shape=(input_dim, 7 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 7 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(7 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )

        self.built = True

    def call(self, inputs, states):
        c, c_bar, h = states
        k = inputs[0]  # Is the input
        delta_t = inputs[1]  # is the elapsed time

        z = (
            tf.matmul(k, self.input_kernel)
            + tf.matmul(h, self.recurrent_kernel)
            + self.bias
        )
        i, ig, fg, og, ig_bar, fg_bar, d = tf.split(z, 7, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        input_gate_bar = tf.nn.sigmoid(ig_bar)
        forget_gate = tf.nn.sigmoid(fg)
        forget_gate_bar = tf.nn.sigmoid(fg_bar)
        output_gate = tf.nn.sigmoid(og)
        delta_gate = tf.nn.softplus(d)

        new_c = c * forget_gate + input_activation * input_gate
        new_c_bar = c_bar * forget_gate_bar + input_activation * input_gate_bar

        c_t = new_c_bar + (new_c - new_c_bar) * tf.exp(-delta_gate * delta_t)
        output_state = tf.nn.tanh(c_t) * output_gate

        return output_state, [new_c, new_c_bar, output_state]

#implemented based on the ODE-Transformer paper: https://arxiv.org/abs/2310.05573

class ODEformer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads=4, ff_dim=None, n_steps=3, step_size=0.25, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or 4 * hidden_dim
        self.n_steps = n_steps
        self.step_size = step_size
        self.dropout_rate = dropout

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim != self.hidden_dim:
            # add projection to match dimensions
            self.input_proj = tf.keras.layers.Dense(self.hidden_dim)
        else:
            self.input_proj = tf.identity

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.hidden_dim // self.num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        y = self.input_proj(inputs) if callable(self.input_proj) else self.input_proj(inputs)
        for _ in range(self.n_steps):
            attn_out = self.mha(y, y, y, training=training)
            y1 = self.norm1(y + self.dropout(attn_out, training=training))
            ffn_out = self.ffn(y1, training=training)
            dy = self.norm2(y1 + self.dropout(ffn_out, training=training))
            y = y + self.step_size * dy
        return y


#implemented based on https://ojs.aaai.org/index.php/AAAI/article/view/16875

class CTA(tf.keras.layers.Layer):
    def __init__(self, hidden_size, seq_len=None, ode_layers=[32, 32], **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Input and recurrent components
        self.input_projection = tf.keras.layers.Dense(hidden_size)
        self.rnn_cell = tf.keras.layers.GRUCell(hidden_size)

        # ODE function (neural dynamics)
        self.ode_func = tf.keras.Sequential()
        for units in ode_layers:
            self.ode_func.add(tf.keras.layers.Dense(units, activation="tanh"))
        self.ode_func.add(tf.keras.layers.Dense(hidden_size))

        # Attention components
        self.wq = tf.keras.layers.Dense(hidden_size)
        self.wk = tf.keras.layers.Dense(hidden_size)
        self.wv = tf.keras.layers.Dense(hidden_size)
        self.query = self.add_weight(
            name="query", shape=(1, hidden_size), initializer="random_normal"
        )

        # Learnable time step scale (positive via softplus)
        self.time_scale = self.add_weight(
            name="time_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # shapes (may be symbolic)
        batch_size = tf.shape(inputs)[0]
        seq_len = self.seq_len or tf.shape(inputs)[1]
        hidden_size = self.hidden_size

        # compute a scalar dt (learnable, positive)
        dt = tf.nn.softplus(self.time_scale)  # tensor scalar

        # Initialize states from first timestep
        z = self.input_projection(inputs[:, 0, :])          # (batch, hidden)
        c = tf.zeros((batch_size, hidden_size), dtype=z.dtype)
        a = tf.zeros((batch_size, 1), dtype=z.dtype)

        # fixed learnable query projected
        q = self.wq(self.query)                             # (1, hidden)
        q = tf.tile(q, [batch_size, 1])                    # (batch, hidden)

        # convert seq_len to int32 for range arithmetic in cond
        seq_len_i = tf.cast(seq_len, tf.int32)

        def loop_body(n, z, c, a):
            # ODE dynamics via Euler integration (single step)
            z_dot = self.ode_func(z)                       # (batch, hidden)

            # attention-like scalar from query/key (per batch)
            k = self.wk(z)                                 # (batch, hidden)
            v = self.wv(z)                                 # (batch, hidden)
            score = tf.reduce_sum(q * k, axis=-1, keepdims=True)  # (batch, 1)

            # Use sigmoid so alpha is in (0,1); softmax over single key is always 1.
            alpha = tf.sigmoid(score / tf.sqrt(tf.cast(hidden_size, z.dtype)))  # (batch,1)

            c_dot = alpha * v                              # (batch, hidden) broadcast alpha
            a_dot = alpha                                  # (batch, 1)

            # Euler integration over interval dt
            z_hat = z + dt * z_dot
            c_new = c + dt * c_dot
            a_new = a + dt * a_dot

            # GRU cell update using next observation
            # Note: GRUCell returns (output, new_state) where new_state has shape (batch, hidden)
            new_input = self.input_projection(inputs[:, n, :])
            z_new, z_state = self.rnn_cell(new_input, [z_hat])
            # z_new (output) and z_state[0] (state) are usually the same for GRUCell; use output.
            return n + 1, z_new, c_new, a_new

        # while loop initial counter = 1 (we already consumed t=0)
        init_n = tf.constant(1, dtype=tf.int32)

        # shape invariants including counter (scalar), z (batch x hidden), c (batch x hidden), a (batch x 1)
        _, z, c, a = tf.while_loop(
            cond=lambda n, *_: n < seq_len_i,
            body=loop_body,
            loop_vars=[init_n, z, c, a],
            shape_invariants=[
                tf.TensorShape([]),                       # n
                tf.TensorShape([None, hidden_size]),      # z
                tf.TensorShape([None, hidden_size]),      # c
                tf.TensorShape([None, 1]),                # a
            ],
        )

        attended = z + tf.math.divide_no_nan(c, a)
        return attended

#implemented based on https://arxiv.org/abs/2101.10318
class mTAN(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128, num_heads=8, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = dropout_rate

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_dense = tf.keras.layers.Dense(hidden_dim, use_bias=False)
        self.k_dense = tf.keras.layers.Dense(hidden_dim, use_bias=False)
        self.v_dense = tf.keras.layers.Dense(hidden_dim, use_bias=False)

        # Time encoding g(Δt)
        self.time_dense = tf.keras.layers.Dense(hidden_dim, activation='tanh', use_bias=True)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.out_dense = tf.keras.layers.Dense(hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Will be created in build() if needed
        self.input_proj = None

    def build(self, input_shape):
        # input_shape = (B, T, D+1)
        input_dim = input_shape[-1] - 1  # exclude time channel

        # Automatic projection for residual connection
        if input_dim != self.hidden_dim:
            self.input_proj = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)

        super().build(input_shape)

    def call(self, inputs, mask=None, training=None):
        """
        inputs: (B, T, D+1)
        mask: (B, T), boolean
        """
        x = inputs[:, :, :-1]      # (B, T, D)
        t = inputs[:, :, -1:]      # (B, T, 1)

        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # === Relative time differences: δt_{i,j} ===
        # Relative time differences
        t_i = tf.expand_dims(t, axis=2)   # (B, T, 1, 1)
        t_j = tf.expand_dims(t, axis=1)   # (B, 1, T, 1)
        delta_t = (t_i - t_j)[..., 0]     # (B, T, T)

        # Add last dimension for Dense
        delta_t_exp = tf.expand_dims(delta_t, axis=-1)  # (B, T, T, 1)
        time_enc = self.time_dense(delta_t_exp)         # (B, T, T, H)

        # Time encoding g(Δt)

        # === Standard Q, K, V projections ===
        Q = self.q_dense(x)  # (B, T, H)
        K = self.k_dense(x)
        V = self.v_dense(x)

        # Reshape for multi-head attention
        Q = tf.reshape(Q, [B, T, self.num_heads, self.head_dim])
        K = tf.reshape(K, [B, T, self.num_heads, self.head_dim])
        V = tf.reshape(V, [B, T, self.num_heads, self.head_dim])

        Q = tf.transpose(Q, [0, 2, 1, 3])  # (B, h, T, d)
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        # Expand time encoding for heads
        time_enc = tf.reshape(time_enc, [B, T, T, self.num_heads, self.head_dim])
        time_enc = tf.transpose(time_enc, [0, 3, 1, 2, 4])  # (B, h, T, T, d)

        # K + g(Δt)
        K_time = K[:, :, tf.newaxis, :, :] + time_enc  # (B, h, T, T, d)

        # Compute attention scores via elementwise multiply & sum
        Q_exp = tf.expand_dims(Q, 3)  # (B, h, T, 1, d)
        attn_scores = tf.reduce_sum(Q_exp * K_time, axis=-1)  # (B, h, T, T)
        attn_scores *= self.scale

        # === Masking ===
        if mask is not None:
            # Save original for pooling later
            orig_mask = mask

            mask = tf.cast(mask, tf.float32)
            mask = tf.reshape(mask, (B, 1, 1, T))  # broadcast to (B, h, T, T)
            attn_scores += (1.0 - mask) * -1e9
        else:
            orig_mask = None

        # Softmax
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        # Weighted sum
        attn_out = tf.matmul(attn_weights, V)  # (B, h, T, d)

        # Restore shape
        attn_out = tf.transpose(attn_out, [0, 2, 1, 3])  # (B, T, h, d)
        attn_out = tf.reshape(attn_out, [B, T, self.hidden_dim])

        # Output projection
        out = self.out_dense(attn_out)

        # === Residual connection with automatic projection ===
        residual = x
        if self.input_proj is not None:
            residual = self.input_proj(residual)

        out = self.norm(residual + out)

        # === Pooled representation ===
        if orig_mask is not None:
            m = tf.cast(orig_mask, out.dtype)
            m = tf.expand_dims(m, -1)
            summed = tf.reduce_sum(out * m, axis=1)
            context = summed / (tf.reduce_sum(m, axis=1) + 1e-8)
        else:
            context = tf.reduce_mean(out, axis=1)

        return context

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config

#implemented based on https://arxiv.org/abs/2402.10635

class ContiFormer(tf.keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        n_steps: int = 4,
        total_time: float = 1.0,
        learn_time: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or max(4 * dim, dim)
        self.n_steps = int(n_steps)
        self.total_time = float(total_time)
        self.learn_time = bool(learn_time)
        self.dropout_rate = float(dropout)

        # Layers used by the dynamics function f(y, t)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.dim // self.num_heads)
        self.norm_attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_ff = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation="relu"),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.dim),
        ])
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # Time projection: project scalar time -> vector (to condition attention/MLP)
        self.time_proj = tf.keras.layers.Dense(self.dim, activation=None)

    def build(self, input_shape):
        # create trainable raw deltas if requested
        if self.learn_time:
            # initialize raw_deltas so that after softplus and normalization they are approx uniform
            init_val = tf.constant(1.0 / float(self.n_steps), dtype=tf.float32)
            # raw_deltas shape: (n_steps,)
            self._raw_deltas = self.add_weight(
                name="raw_time_deltas",
                shape=(self.n_steps,),
                initializer=tf.keras.initializers.Constant(init_val.numpy() if hasattr(init_val, "numpy") else 1.0 / self.n_steps),
                trainable=True,
                dtype=tf.float32,
            )
        else:
            self._raw_deltas = None
        super().build(input_shape)

    def get_time_deltas(self):
        """Return positive time deltas that sum to total_time.

        If learn_time is enabled, transform raw parameters with softplus and normalize.
        Otherwise return uniform deltas.
        """
        if self.learn_time:
            # ensure strictly positive deltas
            pos = tf.nn.softplus(self._raw_deltas)
            # normalize to total_time
            normalized = pos / tf.reduce_sum(pos + 1e-12) * self.total_time
            return normalized
        else:
            return tf.fill((self.n_steps,), tf.constant(self.total_time / float(self.n_steps), dtype=tf.float32))

    def compute_time_embeddings(self):
        """Compute a (n_steps, dim) time embedding matrix.

        We compute cumulative times t_k = cumsum(deltas) and project t_k through a small
        dense layer to get per-step time conditioning vectors.
        """
        deltas = self.get_time_deltas()  # (n_steps,)
        times = tf.cumsum(deltas)  # (n_steps,)
        # project times to dim -- produce (n_steps, dim)
        times_expanded = tf.expand_dims(times, axis=-1)  # (n_steps, 1)
        time_embs = self.time_proj(times_expanded)  # (n_steps, dim)
        return time_embs, deltas

    def dynamics(self, y, t_emb, attention_mask=None, training=None):
        """Dynamics function f(y, t): uses attention + small MLP to produce derivative-like output.

        y: (batch, seq_len, dim)
        t_emb: (batch, seq_len, dim) or broadcastable
        """
        # condition by adding time embedding
        z = y + t_emb
        # self-attention
        attn_out = self.mha(query=z, value=z, key=z, attention_mask=attention_mask, training=training)
        attn_out = self.dropout(attn_out, training=training)
        # residual + norm
        attn_res = self.norm_attn(y + attn_out)
        # feed-forward
        ff_out = self.ffn(attn_res, training=training)
        ff_res = self.norm_ff(attn_res + ff_out)
        # derivative candidate
        return ff_res

    def call(self, inputs, attention_mask=None, training=None):
        """Run Euler integration internally and return final states.

        inputs: (batch, seq_len, dim)
        attention_mask: optional mask compatible with tf.keras.layers.MultiHeadAttention
        """
        x = tf.convert_to_tensor(inputs)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # compute time embeddings and deltas
        time_embs, deltas = self.compute_time_embeddings()  # time_embs: (n_steps, dim), deltas: (n_steps,)

        # expand time embeddings to (n_steps, batch, seq_len, dim) by broadcasting
        # We'll add per-step t_emb to y at each integration step.
        # Expand to (n_steps, 1, 1, dim) so it can broadcast with (batch, seq_len, dim)
        t_embs = tf.reshape(time_embs, (self.n_steps, 1, 1, self.dim))

        # initial state y0 = inputs
        y = tf.identity(x)

        # Euler integration loop
        for k in range(self.n_steps):
            # dt_k is scalar
            dt_k = deltas[k]
            # step-specific time embedding broadcasted to (batch, seq_len, dim)
            t_emb_k = tf.broadcast_to(t_embs[k], (batch_size, seq_len, self.dim))
            # compute derivative
            f_val = self.dynamics(y, t_emb_k, attention_mask=attention_mask, training=training)
            # Euler update: y <- y + dt * f
            # ensure dt broadcastable and cast consistent
            dt_k_cast = tf.cast(dt_k, y.dtype)
            y = y + dt_k_cast * f_val

        # optional final normalization
        out = self.final_norm(y)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "n_steps": self.n_steps,
            "total_time": self.total_time,
            "learn_time": self.learn_time,
            "dropout": self.dropout_rate,
        })
        return config



#Linear Attention taken from https://github.com/dongyups/rectified-linear-attention-tf2-keras/blob/main/RLTransformer.py 
# testing purposes
class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., rmsnorm=False):
        super(LinearAttention, self).__init__()
        innder_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = tf.keras.layers.Dense(innder_dim * 3, use_bias=False)

        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)

        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Dense(dim),
            tf.keras.layers.Dropout(dropout)
        ]) if project_out else tf.identity

    def call(self, inputs, **kwargs):
        b, n, _, h = *inputs.shape, self.heads
        qkv = tf.split(self.to_qkv(inputs), num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = tf.keras.activations.relu(dots)

        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(self.norm(out))
        return out


#implemented based on https://arxiv.org/abs/2009.14794
class PerformerAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        nb_features=256,
        redraw_projection=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.nb_features = nb_features
        self.redraw_projection = redraw_projection

        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        self.out_proj = tf.keras.layers.Dense(dim)

    def build(self, input_shape):
        # Create a trainable=False weight for the projection matrix
        self.projection_matrix = self.add_weight(
            shape=(self.num_heads, self.head_dim, self.nb_features),
            initializer=tf.random_normal_initializer(),
            trainable=False,
            name="projection_matrix"
        )
        super().build(input_shape)

    def redraw_projection_matrix(self):
        if self.redraw_projection:
            new_matrix = tf.random.normal(
                shape=(self.num_heads, self.head_dim, self.nb_features)
            )
            self.projection_matrix.assign(new_matrix)

    def kernel_feature_map(self, x):
        # x: (B, H, T, D)
        x = tf.einsum("bhtd,hdf->bhtf", x, self.projection_matrix)
        x = tf.exp(x - tf.reduce_max(x, axis=-1, keepdims=True))
        return x

    def call(self, x, training=None):
        if training:
            self.redraw_projection_matrix()

        B, T, _ = tf.unstack(tf.shape(x))
        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, 3, axis=-1)

        q = tf.reshape(q, (B, T, self.num_heads, self.head_dim))
        k = tf.reshape(k, (B, T, self.num_heads, self.head_dim))
        v = tf.reshape(v, (B, T, self.num_heads, self.head_dim))

        q = tf.transpose(q, (0, 2, 1, 3))
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))

        q_prime = self.kernel_feature_map(q)
        k_prime = self.kernel_feature_map(k)

        kv = tf.einsum("bhtf,bhtd->bhfd", k_prime, v)
        z = 1.0 / (
            tf.einsum("bhtf,bhf->bht", q_prime, tf.reduce_sum(k_prime, axis=2)) + 1e-6
        )

        out = tf.einsum("bhtf,bhfd,bht->bhtd", q_prime, kv, z)
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (B, T, self.dim))

        return self.out_proj(out)


# implemented based on https://papers.nips.cc/paper_files/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
class SSM(tf.keras.layers.Layer):
    def __init__(self, dim=64, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.dim = input_shape[-1]

        self.in_proj = tf.keras.layers.Dense(self.dim)

        self.log_A = self.add_weight(
            shape=(self.dim,),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True,
            name="log_A",
        )
        self.B = self.add_weight(
            shape=(self.dim,),
            initializer="random_normal",
            trainable=True,
            name="B",
        )
        self.C = self.add_weight(
            shape=(self.dim, self.dim),
            initializer="random_normal",
            trainable=True,
            name="C",
        )
        self.D = self.add_weight(
            shape=(self.dim,),
            initializer="zeros",
            trainable=True,
            name="D",
        )

        super().build(input_shape)

    def call(self, x):
        # x: (B, T, dim)
        B = tf.shape(x)[0]

        u = self.in_proj(x)  # (B, T, dim)
        A = -tf.exp(self.log_A)

        def step(h, u_t):
            return h * A + u_t * self.B

        h0 = tf.zeros((B, self.dim), dtype=x.dtype)

        h_seq = tf.scan(
            step,
            tf.transpose(u, (1, 0, 2)),
            initializer=h0,
        )

        h_seq = tf.transpose(h_seq, (1, 0, 2))  # (B, T, dim)

        y = tf.einsum("bts,sd->btd", h_seq, self.C)  # (B, T, dim)

        return y + tf.einsum("btd,d->btd", x, self.D)


# implemented based on https://github.com/srush/annotated-s4/blob/main/s4/s4.py for testing purposes

def make_DPLR_HiPPO(N):
    n = np.arange(N)
    alpha = 0.5
    P = np.sqrt(n + alpha)
    A = P[:, None] * P[None, :]
    A = np.tril(A) - np.diag(n)
    A = -A
    B = np.sqrt(2 * n + 1.0)

    # Cast to complex64
    A = A.astype(np.complex64)
    P = P.astype(np.complex64)
    B = B.astype(np.complex64)

    S = A + np.outer(P, P)
    S = -1j * S

    Lambda_imag, V = np.linalg.eigh(S)
    Lambda_real = np.zeros(N, dtype=np.float32)
    P = V.conj().T @ P
    B = V.conj().T @ B
    Lambda = Lambda_real + 1j * Lambda_imag
    return Lambda, P.real, B.real


class S4(tf.keras.layers.Layer):
    def __init__(self, d_model, state_size=64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.state_size = state_size

        Lambda, P, B = make_DPLR_HiPPO(state_size)

        Lambda_re_init = np.broadcast_to(Lambda.real, (d_model, state_size))
        Lambda_im_init = np.broadcast_to(Lambda.imag, (d_model, state_size))
        P_init = np.broadcast_to(P, (d_model, state_size))
        B_init = np.broadcast_to(B, (d_model, state_size))

        self.Lambda_re = self.add_weight(
            "Lambda_re", (d_model, state_size),
            initializer=tf.constant_initializer(Lambda_re_init), trainable=True
        )
        self.Lambda_im = self.add_weight(
            "Lambda_im", (d_model, state_size),
            initializer=tf.constant_initializer(Lambda_im_init), trainable=True
        )
        self.P = self.add_weight(
            "P", (d_model, state_size),
            initializer=tf.constant_initializer(P_init), trainable=True
        )
        self.B = self.add_weight(
            "B", (d_model, state_size),
            initializer=tf.constant_initializer(B_init), trainable=True
        )
        self.D = self.add_weight(
            "D", (d_model,),
            initializer=tf.constant_initializer(1.0), trainable=True
        )
        self.log_step = self.add_weight(
            "log_step", (d_model,),
            initializer=tf.random_uniform_initializer(np.log(0.001), np.log(0.1)),
            trainable=True
        )
        self.C_re = self.add_weight(
            "C_re", (d_model, state_size),
            initializer=tf.random_normal_initializer(stddev=np.sqrt(0.5)),
            trainable=True
        )
        self.C_im = self.add_weight(
            "C_im", (d_model, state_size),
            initializer=tf.random_normal_initializer(stddev=np.sqrt(0.5)),
            trainable=True
        )

    def call(self, u):
        B, L, H = tf.unstack(tf.shape(u))

        k = self.kernel(L, H)
        u = tf.transpose(u, [0, 2, 1])  # (B, H, L)

        pad_len = L
        u_pad = tf.pad(u, [[0, 0], [0, 0], [0, pad_len]])  # (B, H, 2L)
        k_pad = tf.pad(k, [[0, 0], [0, pad_len]])          # (H, 2L)
        u_fft = tf.signal.rfft(u_pad)
        k_fft = tf.signal.rfft(k_pad)

        out_fft = u_fft * k_fft[None, :, :]
        out = tf.signal.irfft(out_fft)
        out = out[:, :, :L]  # (B, H, L)

        # Skip connection
        D_in = self.D[:H]
        out = out + u * D_in[:, None]

        return tf.transpose(out, [0, 2, 1])

    def kernel(self, L, H):
        Lambda = tf.complex(
            tf.maximum(self.Lambda_re[:H], -1e-4),
            self.Lambda_im[:H]
        )
        P = tf.complex(self.P[:H], tf.zeros_like(self.P[:H]))
        B = tf.complex(self.B[:H], tf.zeros_like(self.B[:H]))
        C = tf.complex(self.C_re[:H], self.C_im[:H])

        step = tf.exp(self.log_step[:H])
        step = tf.cast(step, tf.complex64)

        Lf = tf.cast(L, tf.float32)
        freq = tf.range(Lf) / Lf
        freq = tf.cast(freq, tf.complex64)
        omega = tf.exp(tf.complex(0.0, -2.0 * np.pi) * freq)

        g = (2.0 / step[:, None]) * ((1.0 - omega) / (1.0 + omega))
        toeplitz = tf.cast(2.0 / (1.0 + omega), tf.complex64)

        def cauchy(v, g, Lambda):
            return tf.reduce_sum(
                v[:, :, None] / (g[:, None, :] - Lambda[:, :, None]),
                axis=1
            )

            
        k00 = cauchy(C * B, g, Lambda)
        k01 = cauchy(C * P, g, Lambda)
        k10 = cauchy(P * B, g, Lambda)
        k11 = cauchy(P * P, g, Lambda)

        at_roots = toeplitz[None, :] * (
            k00 - k01 * (1.0 / (1.0 + k11)) * k10
        )

        k = tf.signal.ifft(at_roots)
        k = tf.math.real(k)

        return k[:, :L]
