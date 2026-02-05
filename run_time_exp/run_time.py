import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTMCell, GRUCell,
    RNN, SimpleRNNCell, MultiHeadAttention, Flatten, Attention
)
from tensorflow.keras.models import Model
from ncps.tf import LTCCell, CfCCell
from ncps.wirings import FullyConnected
from baseline_cells import CTRNNCell, ODELSTM, CTGRU, GRUD, PhasedLSTM, GRUODE, ODEformer, CTA, mTAN, ContiFormer, LinearAttention, PerformerAttention, SSM,S4
from LANFormer import LANFormer

# CONFIG
stat_dir = 'statistics'
sequence_length = 1024
hidden_dim = 64
batch_size = 1
num_runs = 10
num_heads = 4
np.random.seed(42)

# Wiring for LTC and CfC
wiring = FullyConnected(hidden_dim)

# MODEL BUILDER
def build_model(cell_type, seq_len=sequence_length, hidden_dim=hidden_dim, num_heads=num_heads):
    inp = Input(shape=(seq_len, hidden_dim))

    if cell_type == "RNNCell":
        x = RNN(SimpleRNNCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "LSTMCell":
        x = RNN(LSTMCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "GRUCell":
        x = RNN(GRUCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "LTCCell":
        x = RNN(LTCCell(wiring), return_sequences=False)(inp)
    elif cell_type == "CfCCell":
        x = RNN(CfCCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "ODELSTM":
        x = RNN(ODELSTM(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "GRUODE":
        x = RNN(GRUODE(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "CTRNNCell":
        x = RNN(CTRNNCell(hidden_dim, num_unfolds=5, method='euler'), return_sequences=False)(inp)
    elif cell_type == "SSM":
        x = SSM(dim=hidden_dim)(inp)
        x = Flatten()(x)
    elif cell_type == "S4":
        x = S4(d_model=hidden_dim)(inp)
        x = Flatten()(x)
    elif cell_type == "Performer":
        x = PerformerAttention(dim=hidden_dim, num_heads=num_heads)(inp)
        x = Flatten()(x)
    elif cell_type == "Attention":
        x = Attention()([inp, inp])
        x = Flatten()(x)
    elif cell_type == "MultiHeadAttention":
        x = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)(inp, inp)
        x = Flatten()(x)
    elif cell_type == 'linear_attention':
        x = LinearAttention(dim=hidden_dim, heads=num_heads)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif cell_type == "odeformer":
        x = ODEformer(hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=hidden_dim)(inp)
        x = Flatten()(x)
    elif cell_type == "CTA":
        x = CTA(hidden_size=hidden_dim)(inp)
    elif cell_type == "mTAN":
        x = mTAN(hidden_dim=hidden_dim, num_heads=num_heads)(inp)
    elif cell_type == "contiformer":
        x = ContiFormer(dim=hidden_dim, num_heads=num_heads, ff_dim=hidden_dim)(inp)
    elif cell_type == "LANFormer_residual":
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, enable_hc=False, dynamic_hc=False )(x)
        x = Flatten()(x)
    elif cell_type in ["LANFormer_dynamicHC", "LANFormer_sink", "LANFormer_DHC_expansion4", "LANFormer_topk8", "LANFormer_without_PCNN",]:
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, topk=8, enable_hc=True, use_sink_gate=True, expansion_rate=4, dynamic_hc=True )(x)
        x = Flatten()(x)
    elif cell_type in ["LANFormer_staticHC","LANFormer_SHC_expansion4",]:
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, enable_hc=True, dynamic_hc=False )(x)
        x = Flatten()(x)
    elif cell_type == "LANFormer_Nosink":
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, enable_hc=True, dynamic_hc=True, use_sink_gate=False )(x)
        x = Flatten()(x)
    elif cell_type == "LANFormer_DHC_expansion2":
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, expansion_rate=2 )(x)
        x = Flatten()(x)
    elif cell_type == "LANFormer_DHC_expansion8":
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, expansion_rate=8 )(x)
        x = Flatten()(x)
    elif cell_type == "LANFormer_SHC_expansion2":
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, expansion_rate=2 , enable_hc=True, dynamic_hc=False )(x)
        x = Flatten()(x)
    elif cell_type == "LANFormer_SHC_expansion8":
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, expansion_rate=8 , enable_hc=True, dynamic_hc=False )(x)
        x = Flatten()(x)
    elif cell_type == "LANFormer_topk2":
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, topk=2 )(x)
        x = Flatten()(x)
    elif cell_type == "LANFormer_topk4":
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, topk=4 )(x)
        x = Flatten()(x)
    elif cell_type == "LANFormer_pairwise":
        x = LANFormer(d_model=64, num_heads=4, num_layers=1, ff_dim=32, use_pairwise=True )(x)
        x = Flatten()(x)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    return Model(inp, x)

# RUNTIME + MEMORY FUNCTION
def measure_runtime_and_memory(model, num_runs=10):
    dummy_input = np.random.randn(batch_size, sequence_length, hidden_dim).astype(np.float32)
    runtimes = []

    _ = model(dummy_input)

    gpu_available = len(tf.config.list_physical_devices("GPU")) > 0

    if gpu_available:
        tf.config.experimental.reset_memory_stats("GPU:0")

    for _ in range(num_runs):
        start = time.time()
        _ = model(dummy_input)
        end = time.time()
        runtimes.append(end - start)

    runtimes = np.array(runtimes)
    mean_rt = runtimes.mean()
    std_rt = runtimes.std()
    throughput = 1.0 / mean_rt

    if gpu_available:
        print('Using GPU')
        mem_info = tf.config.experimental.get_memory_info("GPU:0")
        gpu_mem_usage = mem_info["peak"] / (1024 ** 2)
    else:
        import psutil
        print('Using CPU')
        process = psutil.Process(os.getpid())
        gpu_mem_usage = process.memory_info().rss / (1024 ** 2)

    del model
    time.sleep(2)

    return mean_rt, std_rt, throughput, gpu_mem_usage

# MODEL TYPES
model_types = [
    "RNNCell","LSTMCell","GRUCell",
    "GRUODE", "CTRNNCell", "PhasedLSTM",
    "ODELSTM", "CfCCell", "LTCCell", 'SSM',
    "MultiHeadAttention", "Attention", "linear_attention", "Perfomer",
    "mTAN", "odeformer", "contiformer", "CTA",
    "LANFormer_residual", "LANFormer_dynamicHC",
    "LANFormer_staticHC",  #either resiudal or hyperconnections
    "LANFormer_Nosink", 'LANFormer_sink',   #with/without sink gate
    "LANFormer_DHC_expansion2", "LANFormer_DHC_expansion4", "LANFormer_DHC_expansion8",  #varying expansion rates
    "LANFormer_SHC_expansion2", "LANFormer_SHC_expansion4", "LANFormer_SHC_expansion8",  #varying expansion rates
    "LANFormer_topk2", "LANFormer_topk4", "LANFormer_topk8",  #varying_topk
    "LANFormer_pairwise",
]

# MAIN LOOP
results = []

for cell_type in model_types:
    print(f"\nBenchmarking {cell_type}...")
    model = build_model(cell_type)
    mean_rt, std_rt, throughput, mem_usage = measure_runtime_and_memory(model, num_runs)

    results.append({
        "Model": cell_type,
        "Sequence Length (n)": sequence_length,
        "Hidden Dim (k)": hidden_dim,
        "Mean Runtime (s)": round(mean_rt, 4),
        "Std Dev (s)": round(std_rt, 4),
        "Throughput (seq/s)": round(throughput, 2),
        "Peak Memory (MB)": round(mem_usage, 2),
    })

df = pd.DataFrame(results)
print("\n=== Runtime + Memory Benchmark Results ===")
print(df.to_string(index=False))

df.to_csv(f"stats.csv", index=False)