import os
# os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import math
import numpy as np
import tensorflow as tf
import numpy.random as npr

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from baseline_cells import SPDATransformer, ContiFormer, OTTransformer, ODEformer, CTA, CTRNNCell
from ncps.keras import CfC
from ncps.wirings import AutoNCP
from FLUID import FLUID

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
})


np.random.seed(42)
tf.random.set_seed(42)
npr.seed(42)

PLOT_DIR = "plots"
WEIGHTS_DIR  = 'model_weights'
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


class Spiral:
    def __init__(self,
                 n_spirals=500,
                 steps=300,
                 turns=3,
                 noise_std=0.05,
                 ntrain=350,
                 ntest=150,
                 nsample_obs=50):
        
        self.n_spirals = n_spirals
        self.steps = steps
        self.turns = turns
        self.noise_std = noise_std
        self.ntrain = ntrain
        self.ntest = ntest
        self.nsample_obs = nsample_obs

        self._generate_dataset()
        self.models = {}

    def _generate_noisy_spirals(self):
        all_t, all_xy = [], []
        for _ in range(self.n_spirals):
            t = np.linspace(0, 2 * np.pi * self.turns, self.steps)
            r = t / (2 * np.pi * self.turns)
            x = r * np.cos(t)
            y = r * np.sin(t)
            xy = np.stack([x, y], axis=-1)
            xy += np.random.normal(scale=self.noise_std, size=xy.shape)
            all_t.append(t[:, None])
            all_xy.append(xy)
        return (np.concatenate(all_t, axis=0).astype(np.float32),
                np.concatenate(all_xy, axis=0).astype(np.float32))

    def _generate_dataset(self):
        t_data, y_data = self._generate_noisy_spirals()
        t_data = t_data.reshape(self.n_spirals, self.steps, 1)
        y_data = y_data.reshape(self.n_spirals, self.steps, 2)

        t_clean = np.linspace(0, 2 * np.pi * self.turns, self.steps)
        r_clean = t_clean / (2 * np.pi * self.turns)
        x_clean = r_clean * np.cos(t_clean)
        y_clean = r_clean * np.sin(t_clean)
        clean_traj = np.stack([x_clean, y_clean], axis=-1)
        clean_data = np.tile(clean_traj, (self.n_spirals, 1, 1))

        self.train_noisy = y_data[:self.ntrain]
        self.train_clean = clean_data[:self.ntrain]
        self.test_noisy = y_data[self.ntrain:]
        self.test_clean = clean_data[self.ntrain:]
        self.t_clean = t_clean

        self.test_idx = sorted(npr.choice(self.steps, self.nsample_obs, replace=False).tolist())

    # ===== Model creation =====
    def _spda_transformer(self):
        inputs = tf.keras.Input(shape=(1, 1))
        x = SPDATransformer(embed_dim=64, num_heads=16, ff_dim=64)(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse', metrics=['mae'])
        return model

    def _ctrnn(self):
        inputs = tf.keras.Input(shape=(1, 1))
        x = tf.keras.layers.RNN(CTRNNCell(units=64,method='euler', num_unfolds=10),return_sequences=True)(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse', metrics=['mae'])
        return model

    def _odeformer(self):
        inputs = tf.keras.Input(shape=(1, 1))
        x = ODEformer(hidden_dim=64, num_heads=16, ff_dim=64)(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse', metrics=['mae'])
        return model

    def _cta(self):
        inputs = tf.keras.Input(shape=(1, 1))
        x = CTA(hidden_size=64)(inputs)
        output = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse', metrics=['mae'])
        return model

    def _contiformer(self):
        inputs = tf.keras.Input(shape=(1, 1))
        x = tf.keras.layers.Dense(64)(inputs)
        x = ContiFormer(dim=64, num_heads=16, ff_dim=64)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse', metrics=['mae'])
        return model

    def _lnn(self):
        inputs = tf.keras.Input(shape=(1, 1))
        output = CfC(AutoNCP(units=64, output_size=2, sparsity_level=0.1,))(inputs)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse', metrics=['mae'])
        return model

    def _ot_transformer(self):
        inputs = tf.keras.Input(shape=(1, 1))
        x = OTTransformer(key_dim=64, num_heads=16, ff_dim=64, num_steps=5)(inputs)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse', metrics=['mae'])
        return model

    def _fluid_transformer(self):
        inputs = tf.keras.Input(shape=(1, 1))
        x = FLUID(d_model=64, num_heads=16, euler_steps=10, num_layers=2, ff_dim=64, topk=32,
                  enable_hc=True, use_sink_gate=True, expansion_rate=4, dynamic_hc=True, max_len=5000)(inputs)
        x = tf.keras.layers.Flatten()(x)
        output = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse', metrics=['mae'])
        return model

    def create_models(self, model_list=None):
        builders = {
            # "CT-RNN" : self._ctrnn,
            # "LNN" : self._lnn,
            "SPDA-Transformer": self._spda_transformer,
            "ODEformer": self._odeformer,
            "ContiFormer": self._contiformer,
            "OTTransformer": self._ot_transformer,
            "FLUID": self._fluid_transformer,
        }
        
        for name in model_list:
            if name not in builders:
                raise ValueError(f"Unknown model: {name}")
            print(f"Building {name}...")
            self.models[name] = builders[name]()

    def _weight_path(self, model_name):
        return os.path.join(WEIGHTS_DIR, f"Spiral_{model_name}_weights.keras")

    def has_weights(self, model_name):
        return os.path.exists(self._weight_path(model_name))

    def load_weights(self, model_name):
        path = self._weight_path(model_name)
        if os.path.exists(path):
            self.models[model_name].load_weights(path)
            print(f"Loaded weights for {model_name}")
            return True
        return False

    def train(self, model_name, epochs=500, batch_size=64, verbose=1, force_retrain=False):
        model = self.models[model_name]

        if not force_retrain and self.has_weights(model_name):
            print(f"Skipping training, loading existing weights for {model_name}")
            self.load_weights(model_name)
            return

        print(f"Training {model_name}...")

        for epoch in range(epochs):
            train_idx = sorted(np.random.choice(self.steps, self.nsample_obs, replace=False).tolist())

            t_train = self.t_clean[train_idx]
            x_train = self.train_noisy[:, train_idx, :]

            t_flat = np.tile(t_train, (self.ntrain, 1)).flatten()[:, None, None]
            y_flat = x_train.reshape(-1, 2)

            perm = np.random.permutation(len(t_flat))
            t_flat = t_flat[perm]
            y_flat = y_flat[perm]

            model.fit(t_flat, y_flat, epochs=1, batch_size=batch_size, verbose=verbose, shuffle=False)
            if (epoch + 1) % 10 == 0:
                print(f"{model_name} - Epoch {epoch+1}/{epochs} completed")

        model.save(self._weight_path(model_name))
        print(f"Saved weights for {model_name}")

    # Evaluation
    def evaluate(self, model_name):
        model = self.models[model_name]
        t_full = self.t_clean[:, None, None]
        preds = model.predict(t_full, verbose=0)
        return np.repeat(preds[None, ...], self.ntest, axis=0)

    # Plotting
    def _plot_on_axis(self, ax, model_name, mean_pred):
        steps = self.steps
        idx_vis = 2

        mean_vis = mean_pred[idx_vis]
        true_vis = self.test_clean[idx_vis]
        obs_points_vis = self.test_noisy[idx_vis, self.test_idx, :]

        half = steps // 3.5
        interp_mask = np.arange(steps) < half
        extrap_mask = ~interp_mask

        ax.plot(true_vis[interp_mask, 0], true_vis[interp_mask, 1], color='green', linewidth=2.5)
        ax.plot(true_vis[extrap_mask, 0], true_vis[extrap_mask, 1], color='green', linestyle=':', linewidth=2.5)

        ax.scatter(obs_points_vis[:, 0], obs_points_vis[:, 1], color='black', s=30)

        ax.plot(mean_vis[interp_mask, 0], mean_vis[interp_mask, 1], color='blue', linewidth=2)
        ax.plot(mean_vis[extrap_mask, 0], mean_vis[extrap_mask, 1], color='red', linewidth=2)

        ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(model_name, fontweight="bold")

    def visualize(self, model_name, mean_pred):
        fig, ax = plt.subplots(figsize=(6, 6))
        self._plot_on_axis(ax, model_name, mean_pred)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"{model_name}.png")
        plt.savefig(path, dpi=300)
        # plt.show()

    def plot_all_models(self, models_to_plot=None):
        if models_to_plot is None:
            models_to_plot = list(self.models.keys())

        cols = 5
        rows = math.ceil(len(models_to_plot) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, name in enumerate(models_to_plot):
            preds = self.evaluate(name)
            self._plot_on_axis(axes[idx], name, preds)

        for ax in axes[len(models_to_plot):]:
            ax.remove()

        handles = [
            Line2D([0], [0], color='green', linewidth=2.5, label='Ground Truth'),
            Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=4, label='Data Points'),
            Line2D([0], [0], color='blue', linewidth=2, label='Interpolation'),
            Line2D([0], [0], color='red', linewidth=2, label='Extrapolation'),
        ]

        fig.legend(handles=handles, loc='upper center', ncol=4)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f"{PLOT_DIR}/all_models_spiral.png")
        # plt.show()


if __name__ == "__main__":
    model_list = [
        # "CT-RNN",
        # "LNN",
        "SPDA-Transformer",
        "ODEformer",
        "OTTransformer",
        "ContiFormer",
        "FLUID",
    ]

    spiral = Spiral()
    spiral.create_models(model_list)

    for name in model_list:
        spiral.train(name, epochs=500, batch_size=64, verbose=0)

        mean_pred = spiral.evaluate(name)

        half = int(spiral.steps / 3.5)
        interp_mask = np.arange(spiral.steps) < half
        extrap_mask = ~interp_mask

        y_true = spiral.test_clean
        
        spiral.visualize(name, mean_pred)

    spiral.plot_all_models(model_list)
