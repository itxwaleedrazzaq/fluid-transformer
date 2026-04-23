import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import pandas as pd
from utils.preprocess import process_features
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Conv1D, LSTMCell, GRUCell,
    RNN, SimpleRNNCell, MultiHeadAttention, GlobalAveragePooling1D, Attention, Activation, Flatten
)
from tensorflow.keras.models import Model
from ncps.tf import LTCCell, CfCCell, CfC
from ncps.wirings import AutoNCP, FullyConnected
from baseline_cells import CTRNNCell, ODELSTM, PhasedLSTM, GRUODE, ODEformer, CTA, mTAN, ContiFormer, LinearAttention, PerformerAttention, SSM, S4, SPDATransformer, PDEAttention, OTTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from PCM import PCM
from FLUID import FLUID
import matplotlib.pyplot as plt
from utils.preprocess import moving_average

plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.2,
    "lines.markersize": 6,
})


base_model_name = 'Degradation_Estimation'
feature_dir = 'tf_features_pronostia'
weights_dir = 'model_weights'
output_dir = "statistics"
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

bearings = {
    '35Hz12kN': ['Bearing1_2'],
    # '37.5Hz11kN': ['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5'],
    # '40Hz10kN': ['Bearing3_1', 'Bearing3_2', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']
}

def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error = y_pred - y_true
    mask_early = error < 0
    mask_late = error >= 0
    score_early = tf.reduce_sum(tf.exp(-error[mask_early] / 13) - 1)
    score_late = tf.reduce_sum(tf.exp(error[mask_late] / 10) - 1)
    return score_early + score_late

wiring = FullyConnected(64)
ncp_wiring = AutoNCP(units=64,output_size=5)

# Wiring for custom cells
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
    elif cell_type == "CfC-AutoNCP":
        x = CfC(ncp_wiring, return_sequences=False)(x)
    elif cell_type == "LTC-AutoNCP":
        x = RNN(LTCCell(ncp_wiring), return_sequences=False)(x)
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
    elif cell_type == "SPDATransformer":
        x = SPDATransformer(embed_dim=64, num_heads=4, ff_dim=32)(x)
        # x = Flatten()(x)
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
        x = ODEformer(hidden_dim=64, num_heads=8, ff_dim=32)(x)
        x = Flatten()(x)
    elif cell_type == "CTA":
        x = CTA(hidden_size=64)(x)
    elif cell_type == "contiformer":
        x = ContiFormer(dim=64, num_heads=4,ff_dim=32)(x)
    elif cell_type == "mTAN":
        x = mTAN(hidden_dim=64, num_heads=4)(x)
    elif cell_type == "PDEAttention":
        x = PDEAttention(key_dim=16, num_heads=4, nt=5, dt=0.1, alpha=0.1)(x)
        x = tf.keras.layers.Flatten()(x)
    elif cell_type == "OTTransformer":
        x = OTTransformer(key_dim=64, num_heads=4,ff_dim=32, num_steps=5)(x)
        x = tf.keras.layers.Flatten()(x)
    elif cell_type == "FLUID_residual":
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32, enable_hc=False, dynamic_hc=False )(x)
        x = Flatten()(x)
    elif cell_type in ["FLUID_dynamicHC", "FLUID_sink", "FLUID_DHC_expansion4", "FLUID_topk8"]:
        x = FLUID(d_model=64, num_heads=4, num_layers=1, ff_dim=32,dropout=0.0, enable_hc=True, use_sink_gate=True, expansion_rate=4, dynamic_hc=True, max_len=1000)(x)
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
    # "RNNCell", "LSTMCell", "GRUCell",
    # "GRUODE", "CTRNNCell", "PhasedLSTM","ODELSTM", 
    # "LTCCell", 'LTC-AutoNCP', "CfCCell", 'CfC-AutoNCP',
    # 'SSM', "S4", 
    "SPDATransformer",
    #  "linear_attention", "Perfomer",
    # "mTAN", 
    "odeformer", 
    "contiformer", 
    # "CTA", 
    'OTTransformer', 
    # 'PDEAttention',
    # "FLUID_residual", 
    "FLUID_dynamicHC",
    # "FLUID_staticHC",  #either resiudal or hyperconnections
    # "FLUID_Nosink",   #with/without sink gate
    # "FLUID_DHC_expansion2", "FLUID_DHC_expansion8",  #varying expansion rates
    # "FLUID_SHC_expansion2", "FLUID_SHC_expansion4", "FLUID_SHC_expansion8",  #varying expansion rates
    # "FLUID_topk2", "FLUID_topk4",  #varying_topk
    # "FLUID_pairwise",   
]


k_folds = 5
results = {}

overall_data_sets = 'XJTU'
for dataset_name, bearing_list in bearings.items():
    print(f"\n=== Evaluating dataset/condition: {overall_data_sets}/{dataset_name} ===")
    
    for bearing in bearing_list:
        print(f"\n-- Bearing: {bearing} --")

        # Load bearing CSV
        df = pd.read_csv(f"{feature_dir}/{bearing}_features.csv")
        X_h = process_features(np.array(df['Horizontal'].apply(eval).tolist()))
        X_v = process_features(np.array(df['Vertical'].apply(eval).tolist()))
        vibration_features = np.concatenate((X_h, X_v), axis=-1)
        t_data = df['Time'].values.reshape(-1, 1)
        if "Temperature" in df.columns:
            T_data = (df['Temperature'].values + 273.15).reshape(-1, 1)
        else:
            T_data = np.full((len(df), 1), 25 + 273.15)
        y = df['Degradation'].values.reshape(-1, 1)
        X = np.concatenate([vibration_features, t_data, T_data], axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dataset_tf = tf.data.Dataset.from_tensor_slices((X_scaled, y)).batch(64)
        input_dim = X_scaled.shape[1]

        bearing_results = []

        for cell_type in model_types:
            model_name = f"{base_model_name}_{cell_type}"
            print(f"\nTesting {model_name} on {bearing}...")

            fold_scores = {"score": [], "mse": [], "mae": []}
            all_preds = []

            for fold in range(1, k_folds + 1):
                model = PCM(model_fn=lambda input_shape: build_model(cell_type, input_shape=(input_dim,)))

                try: 
                    weight_file = f"{weights_dir}/{model_name}_fold{fold}.keras"
                    model.load_weights(weight_file)
                except:
                    weight_file = f"{weights_dir}/{model_name}_fold{fold}.weights.h5"
                    model.load_weights(weight_file)

                if not os.path.exists(weight_file):
                    print(f"  Skipping fold {fold}: no weights found at {weight_file}")
                    continue

                # model.load_weights(weight_file)
                y_true_fold, y_pred_fold = [], []

                for X_batch, y_batch in dataset_tf:
                    preds = model.predict(X_batch, verbose=0)
                    preds = np.clip(preds, 0, 1)
                    y_true_fold.append(y_batch.numpy())
                    y_pred_fold.append(preds)

                y_true_fold = np.vstack(y_true_fold).squeeze()
                y_pred_fold = np.vstack(y_pred_fold).squeeze()

                fold_scores["mse"].append(mean_squared_error(y_true_fold, y_pred_fold))
                fold_scores["mae"].append(mean_absolute_error(y_true_fold, y_pred_fold))
                fold_scores["score"].append(score(y_true_fold, y_pred_fold).numpy())
                all_preds.append(y_pred_fold)

            if fold_scores["score"]:
                data_row = {
                    "Condition": dataset_name,
                    "Bearing": bearing,
                    "Model": cell_type,
                    # "Fold_Score": fold_scores["score"],
                    # "Fold_MSE": fold_scores["mse"],
                    # "Fold_MAE": fold_scores["mae"],
                    # "Mean_MSE": np.mean(fold_scores["mse"]),
                    # "Std_MSE": np.std(fold_scores["mse"]),
                    # "Mean_MAE": np.mean(fold_scores["mae"]),
                    # "Std_MAE": np.std(fold_scores["mae"]),
                    "Mean_Score": np.mean(fold_scores["score"]),
                    "Std_Score": np.std(fold_scores["score"]),
                }
                bearing_results.append(data_row)
                results[(dataset_name, bearing, cell_type)] = data_row

                # ===== Plot Mean ± Std Predictions =====
                if all_preds:
                    all_preds = np.array(all_preds)  # shape: (num_folds, num_samples)
                    mean_preds = np.mean(all_preds, axis=0)
                    std_preds = np.std(all_preds, axis=0)

                    plt.figure(figsize=(6, 4))
                    plt.plot(y, label="Expected", color='black')
                    plt.plot(moving_average(mean_preds[:-170],window_size=15), label=r"Predicted", color='blue')
                    plt.fill_between(np.arange(len(y)),
                                    mean_preds - std_preds,
                                    mean_preds + std_preds,
                                    color='blue', alpha=0.2)
                    plt.title(f"{overall_data_sets}-{cell_type}")
                    plt.xlabel("Time")
                    plt.ylabel("Normalized Degradation")
                    plt.legend()
                    plt.tight_layout()

                    plot_file = os.path.join(plots_dir, f"{overall_data_sets}_{dataset_name}_{bearing}_{cell_type}_prediction.png")
                    plt.savefig(plot_file)
                    plt.close()
                    print(f"Saved prediction plot: {plot_file}")

        # Save results for this bearing as CSV
        if bearing_results:
            df_bearing = pd.DataFrame(bearing_results)
            save_path = os.path.join(output_dir, f"{overall_data_sets}_{dataset_name}_{bearing}_results.csv")
            df_bearing.to_csv(save_path, index=False)
            print(f"\nSaved results for {bearing} at: {save_path}")

# Final summary per condition and bearing
print("\n=== Final Results ===")
for (dataset_name, bearing, cell_type), data in results.items():
    print(f"{overall_data_sets}-{dataset_name} - {bearing} - {base_model_name}_{cell_type}: "
          f"Score={data['Mean_Score']:.4f}±{data['Std_Score']:.4f}, "
        #   f"MSE={data['Mean_MSE']:.4f}±{data['Std_MSE']:.4f}, "
        #   f"MAE={data['Mean_MAE']:.4f}±{data['Std_MAE']:.4f}"
        )
