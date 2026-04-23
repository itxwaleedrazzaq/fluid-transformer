import os
# os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gymnasium as gym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


from FLUID import FLUID
from baseline_cells import ContiFormer, ODEformer, OTTransformer, SPDATransformer, CTRNNCell

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
})

STEPS = 1000
NUM_RUNS = 10
EPISODES = 10
WEIGHTS_DIR = "model_weights2"
MODEL_LIST = ["SPDA", "ODEFormer", "ContiFormer", "OTTransformer", "FLUID"]


def build_model(name, input_shape=(None, 96, 96, 3)):
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(10, (3, 3), activation='relu', strides=2))(inp)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(20, (5, 5), activation='relu', strides=2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(30, (5, 5), activation='relu', strides=2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    if name == "FLUID":
        x = FLUID(d_model=64, num_heads=16, ff_dim=64, topk=16, max_len=5000)(x)
    elif name == "ContiFormer":
        x = tf.keras.layers.Dense(64)(x)
        x = ContiFormer(dim=64, num_heads=16, ff_dim=64)(x)
    elif name == "ODEFormer":
        x = ODEformer(hidden_dim=64, num_heads=16, ff_dim=64)(x)
    elif name == "OTTransformer":
        x = OTTransformer(key_dim=64, num_heads=16, ff_dim=64)(x)
    elif name == "SPDA":
        x = SPDATransformer(embed_dim=64, num_heads=16, ff_dim=64)(x)
    # elif name == "CTRNN":
    #     RNN(CTRNNCell(units=64, method='euler', num_unfolds=5),return_sequences=True)(x)
    else:
        raise ValueError(f"Unknown model name: {name}")

    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    out = tf.keras.layers.Dense(5, activation='softmax')(x)

    return tf.keras.Model(inp, out)


def run_episode(env, model):
    obs, _ = env.reset()

    total_reward = 0
    steps = 0
    success = False

    while True:
        obs_input = np.expand_dims(np.expand_dims(np.array(obs), axis=0), axis=1)

        action_probs = model.predict(obs_input, verbose=0).squeeze()
        action = int(np.argmax(action_probs))

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        success = (terminated and not truncated) or (total_reward > 880)

        if terminated or truncated:
            break

    return success, total_reward, steps

NUM_RUNS = 10
EPISODES = 10

results = []
all_successes = []   # for boxplot (per model)

for name in MODEL_LIST:
    print(f"\nModel: {name}")

    model = build_model(name)
    model.load_weights(f"{WEIGHTS_DIR}/CarRacing_{name}.keras")

    env = gym.make("CarRacing-v3",continuous=False,lap_complete_percent=0.75)

    run_success_means = []   # one value per run
    run_rewards = []
    run_lengths = []

    for run in range(NUM_RUNS):

        successes = []
        rewards = []
        lengths = []

        for ep in range(EPISODES):
            success, reward, steps = run_episode(env, model)

            successes.append(success)
            rewards.append(reward)
            lengths.append(steps)

            print(f"Run {run+1} Ep {ep+1}: success={success}, reward={reward:.1f}, steps={steps}")

        run_success_means.append(np.mean(successes))
        run_rewards.append(np.mean(rewards))
        run_lengths.append(np.mean(lengths))

    env.close()

    results.append({
        "model": name,
        "success_mean": np.mean(run_success_means),
        "success_std": np.std(run_success_means, ddof=1),
    })

    # for boxplot: flatten all episode-level successes across runs
    all_successes.append(np.array(run_success_means))

df = pd.DataFrame(results)
print("\n=== FINAL RESULTS ===")
print(df)
df.to_csv('loop_car.csv')

plt.figure(figsize=(10, 6))
plt.boxplot(all_successes, tick_labels=MODEL_LIST)
plt.xlabel("Model")
plt.ylabel("Mean Success per Run")
plt.title("Success Rate Distribution Under Noise")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.savefig('carloop.png')
plt.show()