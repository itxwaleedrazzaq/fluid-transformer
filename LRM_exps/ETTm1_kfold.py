import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input, Dense,RNN, SimpleRNNCell, LSTMCell, GRUCell, GlobalAveragePooling1D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from sklearn.preprocessing import StandardScaler
from ncps.tf import LTCCell, CfCCell, CfC
from ncps.wirings import FullyConnected, AutoNCP
from baseline_cells import CTRNNCell, ODELSTM, PhasedLSTM, GRUODE, ODEformer,CTA, mTAN, ContiFormer, LinearAttention, PerformerAttention, SSM,S4, PDEAttention, OTTransformer, SPDATransformer
from FLUID import FLUID


base_model_name = 'ETTm1'
weights_dir = 'model_weights'

# Jena Climate is sampled every 10 mins. 
# 96 steps = 16 hours. We predict 192 steps (32 hours) ahead.
LOOKBACK = 20      
HORIZON = 20
TARGET_COL = 'OT'  # Oil Temperature


def prepare_ettm1_data(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[:25000]
    features = df.drop(columns=['date'])
    n_features = features.shape[1]
    n = len(features)
    train_df = features[:int(n*0.8)]
    test_df = features[int(n*0.8):]

    scaler = StandardScaler()
    scaler.fit(train_df)
    train_s = scaler.transform(train_df)
    test_s = scaler.transform(test_df)

    def create_windows(data):
        X, y = [], []
        for i in range(len(data) - LOOKBACK - HORIZON):
            X.append(data[i:i+LOOKBACK, :])
            y.append(data[i+LOOKBACK:i+LOOKBACK+HORIZON, -1])
        return np.array(X), np.array(y)

    X_train, y_train = create_windows(train_s)
    X_test, y_test = create_windows(test_s)
    return X_train, y_train, X_test, y_test, scaler, n_features

X, y, X_test, y_test, scaler, n_feats = prepare_ettm1_data('ETTm1.csv')

wiring = FullyConnected(64)
ncp_wiring = AutoNCP(units=64,output_size=5)

# Model builder
def build_model(cell_type, input_shape=(LOOKBACK, n_feats), num_classes=HORIZON):
    inp = Input(shape=input_shape)
    if cell_type == "RNNCell":
        x = RNN(SimpleRNNCell(64), return_sequences=False)(inp)
    elif cell_type == "LSTMCell":
        x = RNN(LSTMCell(64), return_sequences=False)(inp)
    elif cell_type == "GRUCell":
        x = RNN(GRUCell(64), return_sequences=False)(inp)
    elif cell_type == "LTCCell":
        x = RNN(LTCCell(wiring), return_sequences=False)(inp)
    elif cell_type == "CfCCell":
        x = RNN(CfCCell(64), return_sequences=False)(inp)
    elif cell_type == "CfC-AutoNCP":
        x = CfC(ncp_wiring, return_sequences=False)(inp)
    elif cell_type == "LTC-AutoNCP":
        x = RNN(LTCCell(ncp_wiring), return_sequences=False)(inp)
    elif cell_type == "ODELSTM":
        x = RNN(ODELSTM(16), return_sequences=False)(inp)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(64), return_sequences=False)(inp)
    elif cell_type == "GRUODE":
        x = RNN(GRUODE(64), return_sequences=False)(inp)
    elif cell_type == "CTRNNCell":
        x = RNN(CTRNNCell(64, num_unfolds=5, method='euler'), return_sequences=False)(inp)
    elif cell_type == "SSM":
        x = SSM(dim=64)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "S4":
        x = S4(d_model=64)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "Perfomer":
        x = PerformerAttention(dim=64, num_heads=16)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "SPDATransformer":
        x = SPDATransformer(embed_dim=64, num_heads=16, ff_dim=64)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == 'linear_attention':
        x = LinearAttention(dim=16, heads=8)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "odeformer":
        x = ODEformer(hidden_dim=64, num_heads=16, ff_dim=64)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "CTA":
        x = CTA(hidden_size=64)(inp)
    elif cell_type == "mTAN":
        x = mTAN(hidden_dim=64, num_heads=16)(inp)
    elif cell_type == "contiformer":
        x = tf.keras.layers.Dense(64)(inp)         # project to expected dim
        x = ContiFormer(dim=64, num_heads=16, ff_dim=64)(x)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "PDEAttention":
        x = PDEAttention(key_dim=64, num_heads=16, nt=5, dt=0.1, alpha=0.1)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "OTTransformer":
        x = OTTransformer(key_dim=64, num_heads=16,ff_dim=64, num_steps=5)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_residual":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, enable_hc=False, dynamic_hc=False )(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type in ["FLUID_dynamicHC", "FLUID_sink", "FLUID_DHC_expansion4", "FLUID_topk8"]:
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, topk=8, enable_hc=True, use_sink_gate=True, expansion_rate=4, dynamic_hc=True )(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type in ["FLUID_staticHC","FLUID_SHC_expansion4",]:
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, enable_hc=True, dynamic_hc=False )(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_Nosink":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, enable_hc=True, dynamic_hc=True, use_sink_gate=False )(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_DHC_expansion2":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, expansion_rate=2)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_DHC_expansion8":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, expansion_rate=8)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_SHC_expansion2":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, expansion_rate=2 , enable_hc=True, dynamic_hc=False )(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_SHC_expansion8":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, expansion_rate=8 , enable_hc=True, dynamic_hc=False )(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_topk2":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, topk=2)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_topk4":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, topk=4)(inp)
        x = GlobalAveragePooling1D()(x)
    elif cell_type == "FLUID_pairwise":
        x = FLUID(d_model=64, num_heads=16, num_layers=1, ff_dim=32, use_pairwise=True)(inp)
        x = GlobalAveragePooling1D()(x)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    x = Activation('relu')(x)
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
    # "GRUODE", "CTRNNCell", "PhasedLSTM","ODELSTM", 
    # "LTCCell", 'LTC-AutoNCP',"CfCCell", 'CfC-AutoNCP','SSM', "S4", 
    # "SPDATransformer","linear_attention", "Perfomer",
    # "mTAN", "odeformer", "contiformer", "CTA", 'OTTransformer', 'PDEAttention',
    # "FLUID_residual",
    "FLUID_dynamicHC",
    "FLUID_staticHC",  #either resiudal or hyperconnections
    "FLUID_Nosink",   #with/without sink gate
    "FLUID_DHC_expansion2", "FLUID_DHC_expansion8",  #varying expansion rates
    "FLUID_SHC_expansion2", "FLUID_SHC_expansion8",  #varying expansion rates
    "FLUID_topk2", "FLUID_topk4",  #varying_topk
    "FLUID_pairwise",   
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
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        train_ds = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold))\
                                  .shuffle(buffer_size=500000).batch(128)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold)).batch(128)

        model = build_model(cell_type)
        model.compile(optimizer=AdamW(learning_rate=0.001),loss="mse",metrics=['mae'])
        model.fit(train_ds, validation_data=val_ds, epochs=40,
                  callbacks=get_callbacks(f"{model_name}_fold{fold}"), verbose=0)

        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")

        mse, _ = model.evaluate(X_test, y_test, verbose=0)
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
