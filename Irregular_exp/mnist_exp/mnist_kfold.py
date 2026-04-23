import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, LSTMCell, GRUCell,
    RNN, SimpleRNNCell, MultiHeadAttention, Flatten, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from ncps.tf import LTCCell, CfCCell, CfC
from ncps.wirings import FullyConnected, AutoNCP
from baseline_cells import CTRNNCell, ODELSTM, PhasedLSTM, GRUODE, ODEformer, CTA, mTAN, ContiFormer, LinearAttention, PerformerAttention, SSM, S4, OTTransformer, PDEAttention, SPDATransformer
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.datasets import mnist

from FLUID import FLUID

# Config
base_model_name = 'Event_based_MNIST'
weights_dir = 'model_weights'
os.makedirs(weights_dir, exist_ok=True)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# ---- Preprocessing functions ----
def binarize(img, threshold=128):
    return (img >= threshold).astype(np.int32)

def sequentialize(img):
    return img.flatten()

def event_based_compression(seq):
    events = []
    prev_val = seq[0]
    t = 1
    for i in range(1, len(seq)):
        if seq[i] == prev_val:
            t += 1
        else:
            events.append((prev_val, t))
            prev_val = seq[i]
            t = 1
    events.append((prev_val, t))
    return events

def normalize_time(events, max_len=256):
    total_time = sum(t for _, t in events)
    scale = max_len / total_time
    return [(val, t * scale) for val, t in events]

def pad_events(events, max_len=256):
    events = events[:max_len]
    if len(events) < max_len:
        events += [(0, 0)] * (max_len - len(events))
    return events

def preprocess_dataset(images, labels, max_len=256):
    event_data = []
    for img in images:
        binary_img = binarize(img)
        seq = sequentialize(binary_img)
        events = event_based_compression(seq)
        events = normalize_time(events, max_len)
        padded = pad_events(events, max_len)
        event_data.append(padded)
    return np.array(event_data, dtype=np.float32), labels

# Apply preprocessing
X_events, y = preprocess_dataset(X, y)

# Wiring
wiring = FullyConnected(64)
ncp_wiring = AutoNCP(units=64,output_size=10)

# ---- Model builder ----
def build_model(cell_type, input_shape=(256,2), num_classes=10):
    inp = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same', strides=5)(inp)
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same', strides=5)(x)

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
    elif cell_type == "LTC-AutoNCP":
        x = RNN(LTCCell(ncp_wiring), return_sequences=False)(x)
    elif cell_type == "CfC-AutoNCP":
        x = CfC(ncp_wiring, return_sequences=False)(x)
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
        x = PerformerAttention(dim=64, num_heads=16)(x)
        x = Flatten()(x)
    elif cell_type == "Attention":
        x = Attention()([x, x])
        x = Flatten()(x)
    elif cell_type == "MultiHeadAttention":
        x = MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        x = Flatten()(x)
    elif cell_type == 'linear_attention':
        x = LinearAttention(dim=64, heads=16)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "odeformer":
        x = ODEformer(hidden_dim=64, num_heads=8, ff_dim=128)(x)
        x = Flatten()(x)
    elif cell_type == "CTA":
        x = CTA(hidden_size=64)(x)
    elif cell_type == "mTAN":
        x = mTAN(hidden_dim=64,num_heads=16)(x)
    elif cell_type == "contiformer":
        x = ContiFormer(dim=64, num_heads=8, ff_dim=64)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "PDEAttention":
        x = PDEAttention(key_dim=16, num_heads=8, nt=5, dt=0.1, alpha=0.1)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "SPDATransformer":
        x = SPDATransformer(embed_dim=64, num_heads=16,ff_dim=32)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "OTTransformer":
        x = OTTransformer(key_dim=64, num_heads=16,ff_dim=32, num_steps=3)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    elif cell_type == "FLUID_residual":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, enable_hc=False, dynamic_hc=False )(x)
        x = Flatten()(x)
    elif cell_type in ["FLUID_dynamicHC", "FLUID_sink", "FLUID_DHC_expansion4", "FLUID_topk8", "FLUID_without_PCM",]:
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

    x = Dense(32, activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inp, out)

# ---- Model types ----
model_types = [
    # "RNNCell", "LSTMCell", "GRUCell",
    # "GRUODE", "CTRNNCell", "PhasedLSTM",
    # "ODELSTM", "LTCCell", 'LTC-AutoNCP',"CfCCell", 'CfC-AutoNCP','SSM', "S4",
    # "MultiHeadAttention", 
    # "Attention", 
    # "linear_attention", "Perfomer",
    # "mTAN", 
    # "odeformer", "contiformer", "CTA", 
    'OTTransformer', 'SPDATransformer',
    # 'PDEAttention',
    # "FLUID_residual", "FLUID_dynamicHC",
    # "FLUID_staticHC",  #either resiudal or hyperconnections
    # "FLUID_Nosink",   #with/without sink gate
    # "FLUID_DHC_expansion2", "FLUID_DHC_expansion8",  #varying expansion rates
    # "FLUID_SHC_expansion2", "FLUID_SHC_expansion8",  #varying expansion rates
    # "FLUID_topk2", "FLUID_topk4",  #varying_topk
    # "FLUID_pairwise",   
]


# ---- Callbacks ----
def get_callbacks(model_name):
    return [
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3,
                          min_lr=1e-10, mode='max'),
        ModelCheckpoint(
            f"{weights_dir}/{model_name}.weights.h5",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

# ---- K-Fold CV ----
k_folds = 5
results = {}

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

for cell_type in model_types:
    model_name = f"{base_model_name}_{cell_type}"
    print(f"\nTraining {model_name} with {k_folds}-fold CV...")

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_events, y), 1):
        print(f"  Fold {fold}/{k_folds}")

        # Split data
        X_train, X_val = X_events[train_idx], X_events[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(32)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

        # Build model
        model = build_model(cell_type, input_shape=(256,2), num_classes=10)
        model.compile(
            optimizer=AdamW(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = get_callbacks(f"{model_name}_fold{fold}")

        # Train
        model.fit(train_ds, validation_data=val_ds, epochs=30,callbacks=callbacks, verbose=0)

        # Load best weights and evaluate
        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")
        _, val_acc = model.evaluate(val_ds, verbose=0)

        fold_accuracies.append(val_acc * 100)

    # Store CV results
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    results[cell_type] = {"fold_accuracies": fold_accuracies,
                          "mean": mean_acc, "std": std_acc}

    print(f"{model_name} Fold Accuracies: {fold_accuracies}")
    print(f"{model_name} Mean Accuracy: {mean_acc:.2f}%, Std: {std_acc:.2f}%")

# ---- Final summary ----
print("\n=== Final Model Results ===")
for cell_type, data in results.items():
    print(f"{base_model_name}_{cell_type}: "
          f"Mean={data['mean']:.2f}%, Std={data['std']:.2f}%")
