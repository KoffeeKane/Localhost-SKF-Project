from stable_baselines3 import PPO

from g1_stair_env import G1StairEnv


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = G1StairEnv(mode="stairs")


# ============================================================
# CREATE PPO MODEL
# ============================================================

model = PPO(
    "MlpPolicy",
    env,

    verbose=1,

    learning_rate=3e-4,

    n_steps=2048,

    batch_size=64,

    gamma=0.99,

    gae_lambda=0.95,

    ent_coef=0.01,

    clip_range=0.2,

    device="auto"
)


# ============================================================
# TRAIN
# ============================================================

TRAINING_STEPS = 500_000

print("\nStarting training...")
print("Training steps:", TRAINING_STEPS)


model.learn(
    total_timesteps=TRAINING_STEPS,
    progress_bar=True
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    "g1_staircase_ppo_v2"
)


env.close()


print("\nTraining finished.")
print("Saved as g1_staircase_ppo_v2.zip")