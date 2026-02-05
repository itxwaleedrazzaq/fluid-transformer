# main.py
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import pandas as pd
from utils.preprocess import process_features
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Dense, Reshape, Conv1D, LSTMCell, GRUCell,
    RNN, SimpleRNNCell, MultiHeadAttention, Flatten, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from ncps.tf import LTCCell, CfCCell
from ncps.wirings import FullyConnected
from baseline_cells import CTRNNCell, ODELSTM, PhasedLSTM, GRUODE, ODEformer, CTA, mTAN, ContiFormer, LinearAttention, PerformerAttention, SSM, S4
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from PCM import PCM
from FLUID import FLUID

np.random.seed(100)
tf.random.set_seed(100)

base_model_name = 'Degradation_Estimation'
feature_dir = 'tf_features_xjtu'
weights_dir = 'model_weights'
stat_dir = 'statistics'


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error = y_pred - y_true

    mask_early = error < 0  # early prediction
    mask_late = error >= 0  # late prediction

    score_early = tf.reduce_sum(tf.exp(-error[mask_early] / 13) - 1)
    score_late = tf.reduce_sum(tf.exp(error[mask_late] / 10) - 1)

    return score_early + score_late

bearings = ['Bearing1_1', 'Bearing1_2', 'Bearing1_3']

# Load and preprocess data
dfs = [pd.read_csv(f'{feature_dir}/{bearing}_features_with_labels.csv') for bearing in bearings]

# Process features
horizontal_data = [np.array(df['Horizontal'].apply(eval).tolist()) for df in dfs]
X_h = np.vstack([process_features(data) for data in horizontal_data])
vertical_data = [np.array(df['Vertical'].apply(eval).tolist()) for df in dfs]
X_v = np.vstack([process_features(data) for data in vertical_data])
vibration_features = np.concatenate((X_h, X_v), axis=-1)

# Get other features
t_data = np.concatenate([df['Time'].values.reshape(-1, 1) for df in dfs], axis=0)
# T_data = np.concatenate([(df['Temperature'].values + 273.15).reshape(-1, 1) for df in dfs], axis=0)
T_data = np.concatenate([np.full((df.shape[0], 1), 25 + 273.15) for df in dfs], axis=0)
y = np.concatenate([df['Degradation'].values.reshape(-1, 1) for df in dfs], axis=0)
RPM = np.concatenate([df['RPM'].values.reshape(-1, 1) for df in dfs], axis=0)
Load = np.concatenate([df['Load'].values.reshape(-1, 1) for df in dfs], axis=0)

# Combine features and normalize
X = np.concatenate([vibration_features, t_data, T_data], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val, t_train, t_val, T_train, T_val, Load_train, Load_val, RPM_train, RPM_val = train_test_split(
    X, y, t_data, T_data, Load, RPM, test_size=0.1, random_state=42
)

def create_dataset(X, y, t_data, T_data, Load, RPM, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((
        (X, t_data, T_data),  # Inputs
        (y, (Load, RPM))      # Targets and physics data
    ))
    return dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# Wiring for custom cells
wiring = FullyConnected(64)
# Model builder
def build_model(cell_type, input_shape=(64,), num_classes=1):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32, 3, padding='same', activation='relu')(x)
    x = Conv1D(64, 2, padding='same', activation='relu')(x)

    if cell_type == "RNNCell":
        x = RNN(SimpleRNNCell(64), return_sequences=False)(x)
    elif cell_type == "LSTMCell":
        x = RNN(LSTMCell(64), return_sequences=False)(x)
    elif cell_type == "GRUCell":
        x = RNN(GRUCell(64), return_sequences=False)(x)
    elif cell_type == "LTCCell":
        x = RNN(LTCCell(wiring), return_sequences=False)(x)
    elif cell_type == "CfCCell":
        x = RNN(CfCCell(64), return_sequences=False)(x)
    elif cell_type == "ODELSTM":
        x = RNN(ODELSTM(64), return_sequences=False)(x)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(64), return_sequences=False)(x)
    elif cell_type == "GRUODE":
        x = RNN(GRUODE(64), return_sequences=False)(x)
    elif cell_type == "CTRNNCell":
        x = RNN(CTRNNCell(64, num_unfolds=5, method='euler'), return_sequences=False)(x)
    elif cell_type == "SSM":
        x = SSM(dim=64)(x)
        x = Flatten()(x)
    elif cell_type == "S4":
        x = S4(d_model=64)(x)
        x = Flatten()(x)
    elif cell_type == "Perfomer":
        x = PerformerAttention(dim=64, num_heads=4)(x)
        x = Flatten()(x)
    elif cell_type == "Attention":
        x = Attention()([x, x])
        x = Flatten()(x)
    elif cell_type == "MultiHeadAttention":
        x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = Flatten()(x)
    elif cell_type == 'linear_attention':
        x = LinearAttention(dim=64, heads=8)(x)
        x = tf.keras.layers.Flatten()(x)
    elif cell_type == "odeformer":
        x = ODEformer(hidden_dim=64, num_heads=8, ff_dim=64)(x)
        x = Flatten()(x)
    elif cell_type == "CTA":
        x = CTA(hidden_size=64)(x)
    elif cell_type == "contiformer":
        x = ContiFormer(dim=64, num_heads=8,ff_dim=64)(x)
    elif cell_type == "mTAN":
        x = mTAN(hidden_dim=64, num_heads=4)(x)
        
    elif cell_type == "FLUID_residual":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, enable_hc=False, dynamic_hc=False )(x)
        x = Flatten()(x)
    elif cell_type in ["FLUID_dynamicHC", "FLUID_sink", "FLUID_DHC_expansion4", "FLUID_topk8",]:
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, topk=8, enable_hc=True, use_sink_gate=True, expansion_rate=4, dynamic_hc=True )(x)
        x = Flatten()(x)
    elif cell_type in ["FLUID_staticHC","FLUID_SHC_expansion4",]:
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, enable_hc=True, dynamic_hc=False )(x)
        x = Flatten()(x)
    elif cell_type == "FLUID_Nosink":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, enable_hc=True, dynamic_hc=True, use_sink_gate=False )(x)
        x = Flatten()(x)
    elif cell_type == "FLUID_DHC_expansion2":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, expansion_rate=2 )(x)
        x = Flatten()(x)
    elif cell_type == "FLUID_DHC_expansion8":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, expansion_rate=8 )(x)
        x = Flatten()(x)
    elif cell_type == "FLUID_SHC_expansion2":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, expansion_rate=2 , enable_hc=True, dynamic_hc=False )(x)
        x = Flatten()(x)
    elif cell_type == "FLUID_SHC_expansion8":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, expansion_rate=8 , enable_hc=True, dynamic_hc=False )(x)
        x = Flatten()(x)
    elif cell_type == "FLUID_topk2":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, topk=2 )(x)
        x = Flatten()(x)
    elif cell_type == "FLUID_topk4":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, topk=4 )(x)
        x = Flatten()(x)
    elif cell_type == "FLUID_pairwise":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, use_pairwise=True )(x)
        x = Flatten()(x)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")
    
    x = Dense(128, activation='relu')(x)
    out = Dense(num_classes, activation='linear')(x)
    return Model(inp, out)

# List of model types
model_types = [
    # "RNNCell","LSTMCell","GRUCell",
    # "GRUODE", "CTRNNCell", "PhasedLSTM",
    # "ODELSTM", "CfCCell", "LTCCell", 'SSM',
    # "MultiHeadAttention", "Attention", "linear_attention", "Perfomer",
    # "mTAN", "odeformer", "contiformer", "CTA",
    "FLUID_residual", "FLUID_dynamicHC",
    "FLUID_staticHC",  #either resiudal or hyperconnections
    "FLUID_Nosink", 'FLUID_sink',   #with/without sink gate
    "FLUID_DHC_expansion2", "FLUID_DHC_expansion4", "FLUID_DHC_expansion8",  #varying expansion rates
    "FLUID_SHC_expansion2", "FLUID_SHC_expansion4", "FLUID_SHC_expansion8",  #varying expansion rates
    "FLUID_topk2", "FLUID_topk4", "FLUID_topk8",  #varying_topk
    "FLUID_pairwise",

]



# Callbacks
def get_callbacks(model_name):
    return [
        ModelCheckpoint(
            f"{weights_dir}/{model_name}.weights.h5",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

# Run k-fold CV
k_folds = 5
results = {}

for cell_type in model_types:
    model_name = f"{base_model_name}_{cell_type}"
    print(f"\nTraining {model_name} with {k_folds}-fold CV...")

    fold_score = []

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y), 1):
        print(f"  Fold {fold}/{k_folds}")

        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        t_train, t_val = t_data[train_idx], t_data[val_idx]
        T_train, T_val = T_data[train_idx], T_data[val_idx]
        Load_train, Load_val = Load[train_idx], Load[val_idx]
        RPM_train, RPM_val = RPM[train_idx], RPM[val_idx]

        # Physics-aware datasets
        train_ds = create_dataset(X_train, y_train, t_train, T_train, Load_train, RPM_train, batch_size=64)
        val_ds = create_dataset(X_val, y_val, t_val, T_val, Load_val, RPM_val, batch_size=64)

        # Build model
        input_dim = X_scaled.shape[1]
        model = PCM(model_fn=lambda input_shape: build_model(cell_type, input_shape=(input_dim,)))

        model.compile(Adam(learning_rate=0.00067564))

        callbacks = get_callbacks(f"{model_name}_fold{fold}")

        # Train
        model.fit(train_ds, validation_data=val_ds, epochs=100, verbose=0,callbacks=callbacks)

        # Load best weights and evaluate
        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")
        _, val_score = model.evaluate(val_ds, verbose=0)

        fold_score.append(val_score)  # percentage

    # Store CV stats
    mean_acc = np.mean(fold_score)
    std_acc = np.std(fold_score)

    results[cell_type] = {"fold_score": fold_score,
                          "mean": mean_acc, "std": std_acc}

    print(f"{model_name} Fold Score: {fold_score}")
    print(f"{model_name} Mean Score: {mean_acc:.4f}, Std: {std_acc:.4f}")

# Final summary
print("\n=== Final Model Results ===")
for cell_type, data in results.items():
    print(f"{base_model_name}_{cell_type}: "
          f"Mean={data['mean']:.4f}, Std={data['std']:.4f}")
