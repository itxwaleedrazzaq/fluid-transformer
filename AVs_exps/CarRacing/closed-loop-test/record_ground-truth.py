import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import PPO

STEPS = 860
MODEL_NAME = "Ground_Truth"

env = gym.make("CarRacing-v1", continuous=False)

model = PPO.load("/home/waleed/idea9/recorder/memory.zip")

obs = env.reset()

trajectory_x = []
trajectory_y = []

for t in range(STEPS):
    env.render()

    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

    pos = env.unwrapped.car.hull.position
    trajectory_x.append(float(pos.x))
    trajectory_y.append(float(pos.y))

    if done:
        break

track = env.unwrapped.track
center_x = np.array([tile[2] for tile in track])
center_y = np.array([tile[3] for tile in track])

center_x = np.append(center_x, center_x[0])
center_y = np.append(center_y, center_y[0])

dx = np.gradient(center_x)
dy = np.gradient(center_y)

length = np.sqrt(dx**2 + dy**2)
length = np.maximum(length, 1e-8)  # avoid division by zero

dx /= length
dy /= length

nx = -dy
ny = dx

track_width = 7

left_x = center_x + nx * track_width / 2
left_y = center_y + ny * track_width / 2

right_x = center_x - nx * track_width / 2
right_y = center_y - ny * track_width / 2

env.close()


max_len = max(len(center_x),len(trajectory_x))

def pad(arr, length):
    arr = np.array(arr)
    if len(arr) < length:
        return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)
    return arr[:length]

df = pd.DataFrame({
    "step": np.arange(max_len),

    "center_x": pad(center_x, max_len),
    "center_y": pad(center_y, max_len),

    "left_x": pad(left_x, max_len),
    "left_y": pad(left_y, max_len),

    "right_x": pad(right_x, max_len),
    "right_y": pad(right_y, max_len),

    f"{MODEL_NAME}_x": pad(trajectory_x, max_len),
    f"{MODEL_NAME}_y": pad(trajectory_y, max_len),
})

df.to_csv(f"record_{MODEL_NAME}.csv", index=False)
print('saved')

plt.figure(figsize=(8, 8))

plt.fill(
    np.concatenate([left_x, right_x[::-1]]),
    np.concatenate([left_y, right_y[::-1]]),
    color='gray',
    alpha=0.6,
    label="Track"
)

plt.plot(center_x, center_y, '--', color='black', linewidth=2, label="Center")
plt.plot(trajectory_x, trajectory_y, 'r', linewidth=2, label="Agent")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("GROUND TRUTH TRAJECTORY")
plt.show()

