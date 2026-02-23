# RL Robotic Arm

A deep reinforcement learning project for dexterous robotic manipulation, combining **SAC** (Soft Actor-Critic), **RHER** (Relabeled Hindsight Experience Replay), **GCRL** (Goal-Conditioned Reinforcement Learning), and **SeqGRU** (Sequential GRU-based policy network) to train a robotic arm to grasp and lift objects with precise position and orientation control.

---

## Overview

The agent is trained entirely in simulation using **Isaac Sim 5.0** (Isaac Lab). The environment features:

- **Robot**: A Franka Panda arm equipped with a Shadow dexterous hand, capable of fine-grained finger-level control.
- **Objects**: Randomly spawned YCB-style objects placed on the table — including a **chili bottle**, a **power drill**, and a **mustard bottle**. Object type and initial pose are randomized each episode to encourage generalization.
- **Table**: A fixed surface where the object is placed at the start of each episode. The robot must reach, grasp, and lift the object to a target goal pose.

Two task variants are supported:
- **`lift`** — position-only goal: lift the object to a target 3D position.
- **`lift_orientation`** — full 6-DoF goal: match both target position and orientation.

---

## Method

The learning pipeline combines several techniques:

- **SAC (GCRL variant)**: Off-policy actor-critic algorithm adapted for goal-conditioned settings, providing sample efficiency and stable training.
- **RHER**: A relabeling strategy over hindsight experience to densify the sparse reward signal across multi-stage manipulation tasks.
- **GCRL**: Goals are represented explicitly and fed into the policy, enabling generalization across diverse target configurations.
- **SeqGRU**: A recurrent GRU encoder processes observation sequences, giving the policy temporal context for more robust manipulation behavior.

---

## Setup

**Requirements**: Python 3.10+, Isaac Lab (Isaac Sim 5.0), PyTorch 2.4+ with CUDA 11.8.

1. **Install Isaac Lab** following the [official guide](https://isaac-sim.github.io/IsaacLab/).

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the environment** is discoverable by ensuring `franka_env` and `drl` are on your Python path (run from the project root).

---

## Training

```bash
python franka_train.py --task lift --num-envs 16 --num_cycles 200
```

- `--task`: choose `lift` or `lift_orientation`
- `--num-envs`: number of parallel simulation environments
- `--num_cycles`: training cycles per epoch
- `--resume_path`: path to a previous run to resume (e.g., `runs/exp1`)

Training logs and checkpoints are saved under `runs/`.

---

## Evaluation

```bash
python franka_eval.py --num-envs 4
```

Loads the best saved policy from `runs/best_policy/` and runs a visual rollout in Isaac Sim.

---

## Project Structure

```
RL_RoboticArm_Final/
├── franka_train.py       # Training entry point
├── franka_eval.py        # Evaluation entry point
├── franka_env/           # Isaac Lab environment (robot, objects, MDP)
├── drl/                  # RL algorithms (SAC, RHER, SeqGRU, memory, utils)
├── runs/                 # Saved checkpoints and logs
└── requirements.txt
```