import os
# os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
import numpy as np
import pandas as pd
from utils.preprocess import process_features
from sklearn.preprocessing import StandardScaler
from keras._tf_keras.keras import layers
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import AdamW
from FLUID import FLUID
from sklearn.model_selection import train_test_split
from utils.callbacks import LossLogger
from PCM import PCM

MODEL_NAME = 'RUL_FLUID'
FEATURE_DIR = 'tf_features_xjtu'
WEIGHTS_DIR = 'model_weights'
STATS_DIR = 'statistics'
bearings = ['Bearing1_1', 'Bearing1_2', 'Bearing1_3']


# Load and preprocess data
dfs = [pd.read_csv(f'{FEATURE_DIR}/{bearing}_features_with_labels.csv') for bearing in bearings]


horizontal_data = [np.array(df['Horizontal'].apply(eval).tolist()) for df in dfs]
X_h = np.vstack([process_features(data) for data in horizontal_data])
vertical_data = [np.array(df['Vertical'].apply(eval).tolist()) for df in dfs]
X_v = np.vstack([process_features(data) for data in vertical_data])
vibration_features = np.concatenate((X_h, X_v), axis=-1)

t_data = np.concatenate([df['Time'].values.reshape(-1, 1) for df in dfs], axis=0)
T_data = np.concatenate([np.full((df.shape[0], 1), 25 + 273.15) for df in dfs], axis=0)
y = np.concatenate([df['Degradation'].values.reshape(-1, 1) for df in dfs], axis=0)
RPM = np.concatenate([df['RPM'].values.reshape(-1, 1) for df in dfs], axis=0)
Load = np.concatenate([df['Load'].values.reshape(-1, 1) for df in dfs], axis=0)

X = np.concatenate([vibration_features, t_data, T_data], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val, t_train, t_val, T_train, T_val, Load_train, Load_val, RPM_train, RPM_val = train_test_split(
    X, y, t_data, T_data, Load, RPM, test_size=0.1, random_state=42
)


def create_dataset(X, y, t_data, T_data, Load, RPM, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices(((X, t_data, T_data),(y, (Load, RPM))))
    return dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Create datasets
train_dataset = create_dataset(X_train, y_train, t_train, T_train, Load_train, RPM_train)
val_dataset = create_dataset(X_val, y_val, t_val, T_val, Load_val, RPM_val)


def build_model(input_shape=(16,)):
    inp = layers.Input(shape=input_shape)
    x = layers.Reshape((1,input_shape[0]))(inp)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(64, 2, padding='same', activation='relu')(x)
    x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=64, dropout=0.0,
                  enable_hc=True, use_sink_gate=True, expansion_rate=4, dynamic_hc=True, max_len=5000)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    return Model(inp, out)



callbacks = [ 
    LossLogger(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-10),
    tf.keras.callbacks.ModelCheckpoint(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras", save_best_only=True, monitor='val_loss')

]

model = PCM(model_fn=build_model, input_shape=(16,))
model.compile(AdamW(learning_rate=1e-3))
model.summary()


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=callbacks
)

model.save(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")

#save the histroy in csv file
history_df = pd.DataFrame(history.history)
history_df.to_csv(f"{WEIGHTS_DIR}/{MODEL_NAME}_training_history.csv", index=False)

# model.evaluate(val_ds)
# model.save(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")