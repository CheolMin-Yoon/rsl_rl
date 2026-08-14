# Repository Rules

## Start order

1. Read `git status --short --branch`, the relevant diff, [README.md](README.md), [pyproject.toml](pyproject.toml) and
   the focused tests that own the requested surface.
2. Follow `/home/frlab/research-wiki/AI-Sessions/wiki/harness/policies/project-policy.md`, then read
   `/home/frlab/research-wiki/AI-Sessions/wiki/research/sources/rsl-rl-code.md` as the checked fork portal.
3. Read `/home/frlab/research-wiki/AI-Sessions/wiki/research/methods/ppo.md`; for multi-policy or centralized-critic
   work also read `/home/frlab/research-wiki/AI-Sessions/wiki/research/methods/multi-agent-reinforcement-learning.md`.
4. Apply `/home/frlab/research-wiki/.agents/skills/focus-output/SKILL.md` to every response and
   `/home/frlab/research-wiki/.agents/skills/rsl-rl/SKILL.md` to learner implementation or integration work.
5. Before revision, environment or toolchain work, read the relevant project/toolchain error note in
   `/home/frlab/research-wiki/AI-Sessions/wiki/harness/errors/`.
6. Use `/home/frlab/research-wiki/prompts/reflect.md` only when the durable-capture gate passes or the user requests it.

## Fork contract

- Preserve stock RSL-RL construction and `OnPolicyRunner` lifecycle wherever the requested semantics permit it.
- The public learner surface includes `PPO`, `MultiCriticPPO`, `MultiPolicyPPO` and `SequentialMultiPolicyPPO`.
- Keep simulator, robot, task observation/reward terms and physical action semantics out of this generic fork.
- Preserve named objective/policy identity, action order, checkpoint state and scalar runner logging reward.
- Reject unsupported extension combinations explicitly. Do not imply recurrent, RND, symmetry, export, compile or
  distributed compatibility without the corresponding checked implementation and test.
- Do not sync, rebase or upgrade from upstream unless the user requests it or an accepted consumer requires it.

## Change and acceptance contract

- Preserve unrelated user changes and inspect live source and tests before editing.
- Prefer composition of stock components over copied PPO, storage or runner implementations.
- Add focused tests for every changed public or mathematical contract. Run the full suite for shared model,
  distribution, storage, runner, resolution, checkpoint or fork-sync changes.
- Verify the target Python imports this checkout with the `rsl-rl` skill's provenance script before consumer tests.
- Keep caches, logs, checkpoints, exports and compiled products outside source control.

## Skill routing

- RSL-RL PPO, model/distribution, storage, runner, checkpoint, compile, device and MJLab learner seams: `rsl-rl`.
- ONNX/AOTInductor/TensorRT export, native packaging and low-copy inference: `deploy-policy`.
