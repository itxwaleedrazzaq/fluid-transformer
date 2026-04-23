# main_fixed.py
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Conv2D, Dropout, Lambda,
    RNN, SimpleRNNCell, LSTMCell, GRUCell, Flatten,
    Attention, MultiHeadAttention, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from utils_udacity import batch_generator, INPUT_SHAPE
from ncps.tf import LTCCell, CfCCell, CfC
from ncps.wirings import FullyConnected, AutoNCP
from baseline_cells import CTRNNCell, ODELSTM, PhasedLSTM, GRUODE, ODEformer,CTA, mTAN, ContiFormer, LinearAttention, PerformerAttention, SSM,S4, PDEAttention, OTTransformer, SPDATransformer
from FLUID import FLUID


# Paths and hyperparameters
base_model_name = 'Udacity_Simulator'
data_dir = 'data'
weights_dir = 'model_weights'
nb_epoch = 10
batch_size = 40
learning_rate = 1.0e-4
keep_prob = 0.5

# Load CSV
data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'))
X = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values

# Wiring for custom cells
# wiring = FullyConnected(32)
wiring = FullyConnected(32)
ncp_wiring = AutoNCP(units=64,output_size=5)

# Model builder
def build_model(cell_type, input_shape=INPUT_SHAPE, num_classes=1):
    inp = Input(shape=input_shape)
    x = Lambda(lambda x: x / 127.5 - 1.0)(inp)
    x = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Reshape((-1, x.shape[1]*x.shape[2]))(x)

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
    elif cell_type == "CfC-AutoNCP":
        x = CfC(ncp_wiring, return_sequences=False)(x)
    elif cell_type == "LTC-AutoNCP":
        x = RNN(LTCCell(ncp_wiring), return_sequences=False)(x)
    elif cell_type == "ODELSTM":
        x = RNN(ODELSTM(16), return_sequences=False)(x)
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
        x = PerformerAttention(dim=64, num_heads=16)(x)
        x = Flatten()(x)
    elif cell_type == "Attention":
        x = Attention()([x, x])
        x = Flatten()(x)
    elif cell_type == "MultiHeadAttention":
        x = MultiHeadAttention(num_heads=16, key_dim=64)(x, x)
        x = Flatten()(x)
    elif cell_type == "SPDATransformer":
        x = SPDATransformer(embed_dim=64, num_heads=16, ff_dim=64)(x)
        x = Flatten()(x)
    elif cell_type == 'linear_attention':
        x = LinearAttention(dim=16, heads=8)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "odeformer":
        x = ODEformer(hidden_dim=64, num_heads=16, ff_dim=64)(x)
        x = Flatten()(x)
    elif cell_type == "CTA":
        x = CTA(hidden_size=64)(x)
    elif cell_type == "mTAN":
        x = mTAN(hidden_dim=64, num_heads=16)(x)
    elif cell_type == "contiformer":
        x = tf.keras.layers.Dense(64)(x)         # project to expected dim
        x = ContiFormer(dim=64, num_heads=16, ff_dim=64)(x)
        x = tf.keras.layers.Flatten()(x)
    elif cell_type == "PDEAttention":
        x = PDEAttention(key_dim=64, num_heads=16, nt=5, dt=0.1, alpha=0.1)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "OTTransformer":
        x = OTTransformer(key_dim=64, num_heads=16,ff_dim=64, num_steps=5)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_residual":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, enable_hc=False, dynamic_hc=False )(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type in ["FLUID_dynamicHC", "FLUID_sink", "FLUID_DHC_expansion4", "FLUID_topk8"]:
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, topk=8, enable_hc=True, use_sink_gate=True, expansion_rate=4, dynamic_hc=True )(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type in ["FLUID_staticHC","FLUID_SHC_expansion4"]:
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, enable_hc=True, dynamic_hc=False )(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_Nosink":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, enable_hc=True, dynamic_hc=True, use_sink_gate=False )(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_DHC_expansion2":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, expansion_rate=2)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_DHC_expansion8":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, expansion_rate=8)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_SHC_expansion2":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, expansion_rate=2 , enable_hc=True, dynamic_hc=False)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_SHC_expansion8":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, expansion_rate=8 , enable_hc=True, dynamic_hc=False)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_topk2":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, topk=2)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_topk4":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, topk=4)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_pairwise":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=64, use_pairwise=True)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    x = Activation('elu')(x)
    out = Dense(num_classes, activation='linear')(x)
    return Model(inp, out)

# Callbacks
def get_callbacks(model_name):
    return [
        ModelCheckpoint(
            f"{weights_dir}/{model_name}.weights.h5",
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

# Model types
model_types = [
    # "RNNCell", "LSTMCell", "GRUCell",
    # "GRUODE", "CTRNNCell", "PhasedLSTM",
    # "ODELSTM", "LTCCell", 'LTC-AutoNCP',"CfCCell", 'CfC-AutoNCP','SSM', "S4",
    "SPDATransformer",
    # "linear_attention", "Perfomer",
    # "mTAN", "odeformer", "contiformer", "CTA", 
    'OTTransformer', 
    'PDEAttention',
    # "FLUID_residual", "FLUID_dynamicHC",
    # "FLUID_staticHC",  #either resiudal or hyperconnections
    # "FLUID_Nosink",   #with/without sink gate
    # "FLUID_DHC_expansion2", "FLUID_DHC_expansion8",  #varying expansion rates
    # "FLUID_SHC_expansion2", "FLUID_SHC_expansion8",  #varying expansion rates
    # "FLUID_topk2", "FLUID_topk4",  #varying_topk
    # "FLUID_pairwise",   
]


# K-fold CV
k_folds = 5
results = {}

for cell_type in model_types:
    model_name = f"{base_model_name}_{cell_type}"
    print(f"\nTraining {model_name} with {k_folds}-fold CV...")

    fold_mse = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"  Fold {fold}/{k_folds}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build model
        model = build_model(cell_type, input_shape=INPUT_SHAPE)
        model.compile(
            optimizer=AdamW(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        callbacks = get_callbacks(f"{model_name}_fold{fold}")

        # Train
        model.fit(
            batch_generator(data_dir, X_train, y_train, batch_size, True),
            steps_per_epoch=len(X_train)//batch_size,
            epochs=5,
            validation_data=batch_generator(data_dir, X_val, y_val, batch_size, False),
            validation_steps=len(X_val)//batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # Load best weights and evaluate
        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")
        mse, _ = model.evaluate(
            batch_generator(data_dir, X_val, y_val, batch_size, False),
            steps=len(X_val)//batch_size,
            verbose=0
        )
        fold_mse.append(mse)

    # Store CV stats
    mean_acc = np.mean(fold_mse)
    std_acc = np.std(fold_mse)
    results[cell_type] = {"fold_mse": fold_mse, "mean": mean_acc, "std": std_acc}

    print(f"{model_name} Fold MSE: {fold_mse}")
    print(f"{model_name} Mean MSE: {mean_acc:.4f}, Std: {std_acc:.4f}")

# Final summary
print("\n=== Final Model Results ===")
for cell_type, data in results.items():
    print(f"{base_model_name}_{cell_type}: Mean={data['mean']:.4f}, Std={data['std']:.4f}")
