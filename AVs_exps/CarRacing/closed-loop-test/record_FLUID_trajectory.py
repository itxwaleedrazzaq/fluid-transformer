import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras._tf_keras.keras.layers import (
    Input, Dense, Flatten, Dropout,
    TimeDistributed, Conv2D, GlobalMaxPooling1D
)
from keras._tf_keras.keras.models import Model

from FLUID import FLUID

STEPS = 782
MODEL_NAME = "CarRacing_FLUID"
WEIGHTS_DIR = "model_weights2"


def build_model(input_shape=(None, 96, 96, 3)):
    inp = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(10, (3, 3), activation='relu', strides=2))(inp)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv2D(20, (5, 5), activation='relu', strides=2))(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv2D(30, (5, 5), activation='relu', strides=2))(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Flatten())(x)

    x = FLUID(d_model=64, num_heads=16, ff_dim=64, topk=16, max_len=5000)(x)

    x = GlobalMaxPooling1D()(x)

    x = Dense(64, activation='relu')(x)
    out = Dense(5, activation='softmax')(x)

    return Model(inp, out)


def main():
    env = gym.make("CarRacing-v1", continuous=False)

    obs = env.reset()

    # -------- STORAGE --------
    trajectory_x = []
    trajectory_y = []
    actions = []
    rewards = []

    for t in range(STEPS):

        obs_input = np.expand_dims(np.expand_dims(np.array(obs), axis=0), axis=1)

        action_probs = model.predict(obs_input, verbose=0).squeeze()
        action = np.argmax(action_probs)

        obs, reward, done, info = env.step(action)

        pos = env.unwrapped.car.hull.position
        x, y = float(pos.x), float(pos.y)

        trajectory_x.append(x)
        trajectory_y.append(y)
        actions.append(action)
        rewards.append(reward)

        if done:
            break

    env.close()

    # -------- SAVE CSV --------
    df = pd.DataFrame({
        "step": np.arange(len(trajectory_x)),
        "x": trajectory_x,
        "y": trajectory_y,
        "action": actions,
        "reward": rewards
    })

    csv_path = "fluid_trajectory_log.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV -> {csv_path}")

    # -------- PLOT --------
    plt.figure()
    plt.plot(trajectory_x, trajectory_y, linewidth=2)
    plt.title("CarRacing FLUID Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    model = build_model()
    model.load_weights(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")

    main()