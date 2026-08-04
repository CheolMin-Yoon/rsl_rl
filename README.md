# RSL-RL

**RSL-RL** is a GPU-accelerated, lightweight learning library for robotics research. Its compact design allows
researchers to prototype and test new ideas without the overhead of modifying large, complex libraries. RSL-RL can also
be used out-of-the-box by installing it via [PyPI](https://pypi.org/project/rsl-rl-lib/), supports multi-GPU training,
and features common algorithms for robot learning.

## Key Features

- **Minimal, readable codebase** with clear extension points for rapid prototyping.
- **Robotics-first methods** including PPO and Student-Teacher Distillation.
- **High-throughput training** with native Multi-GPU support.
- **Proven performance** in numerous research publications.

## Fork Extension: Fixed-Weight Multi-Critic PPO

This fork adds `MultiCriticPPO`: one actor with an ordered independent scalar critic per objective. It uses the stock
`OnPolicyRunner`; only the algorithm configuration and environment extras contract change:

The reusable aggregation follows
[HoST's fixed multi-critic implementation](https://github.com/InternRobotics/HoST/tree/70bb580949a336a920833700e4b5dc3bf7fe87ce/rsl_rl),
without its task-specific interpolation losses.

```python
train_cfg["algorithm"] = {
    "class_name": "MultiCriticPPO",
    "objective_names": ("locomotion", "manipulation", "tracking"),
    "reward_group_weights": (0.5, 1.5, 2.0),
    "normalize_advantage_per_mini_batch": False,
    "share_cnn_encoders": False,
    "rnd_cfg": None,
    "symmetry_cfg": None,
    # ordinary PPO settings ...
}
```

The environment must return `extras["objective_rewards"]` as a floating tensor with exact shape
`[num_envs, num_objectives]`. Its final axis follows `objective_names` exactly; the ordinary scalar `rewards` tensor is
left unchanged for runner logging. Each objective GAE is normalized over the completed rollout, then
`reward_group_weights` forms the actor advantage. Checkpoints store and require exact objective order and weight values.

Feed-forward models and `torch.compile` are supported. Actor/critic broadcast and gradient all-reduce use the stock
multi-GPU lifecycle; this fork's tests cover those hooks, not a multi-process NCCL run. Recurrent models, RND, symmetry,
per-mini-batch advantage normalization, and shared actor/critic parameters are intentionally rejected because they do
not yet have an unambiguous multi-critic contract.

## Learning Environments

RSL-RL is currently used by the following robot learning libraries:

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (built on top of NVIDIA Isaac Sim)
- [Legged Gym](https://github.com/leggedrobotics/legged_gym) (built on top of NVIDIA Isaac Gym)
- [mjlab](https://github.com/mujocolab/mjlab) (built on top of MuJoCo Warp)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (built on top of MuJoCo MJX and Warp)

## Installation

Before installing RSL-RL, ensure that Python `3.9+` is available. It is recommended to install the library in a virtual
environment (e.g. using `venv` or `conda`), which is often already created by the used environment library (e.g.
Isaac Lab). If so, make sure to activate it before installing RSL-RL.

### Installing RSL-RL as a dependency

```bash
pip install rsl-rl-lib
```

### Installing RSL-RL for development

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```

## Citation

If you use RSL-RL in your research, please cite the [paper](https://arxiv.org/abs/2509.10771):

```text
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}
```
