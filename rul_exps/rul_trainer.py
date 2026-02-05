import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
import numpy as np
import pandas as pd
from utils.preprocess import process_features
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras .models import Model
from tensorflow.keras .optimizers import AdamW
from FLUID import FLUID


MODEL_NAME = 'RUL_Test_without_PCNN'
FEATURE_DIR = 'tf_features_xjtu'
WEIGHTS_DIR = 'model_weights'
STATS_DIR = 'statistics'
bearings = ['Bearing1_1', 'Bearing1_2', 'Bearing1_3']


# Load and preprocess data
dfs = [pd.read_csv(f'{FEATURE_DIR}/{bearing}_features_with_labels.csv') for bearing in bearings]

# Process features
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

# Combine features and normalize
X = np.concatenate([vibration_features, t_data, T_data], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)


#build BO tuned model
def build_model(input_shape=(16,)):
    inp = layers.Input(shape=input_shape)
    x = layers.Reshape((1,input_shape[0]))(inp)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(64, 2, padding='same', activation='relu')(x)
    x = FLUID(d_model=64, num_heads=4, topk=8, num_layers=1, ff_dim=32,dropout=0.0,
                  enable_hc=True, use_sink_gate=True, expansion_rate=4, dynamic_hc=True, max_len=1000)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    return Model(inp, out)


model = build_model()
model.summary()
model.compile(AdamW(learning_rate=6.07e-4),loss='mse', metrics=['mae'])

callbacks = [ 
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-15),
    tf.keras.callbacks.ModelCheckpoint(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras", save_best_only=True, monitor='val_loss')
]


history = model.fit(X,y,epochs=100,validation_split=0.1, callbacks=callbacks)
model.evaluate(X,y)
# model.save(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")