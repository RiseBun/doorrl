# DOOR-RL Server Handoff

This document is the practical handoff note for continuing development on a server where the real datasets and benchmark environments live.

The goal is to make the next development stage unambiguous:

1. what this codebase currently does
2. what it does not do yet
3. how to bring it up on a server
4. how to connect `nuScenes`, `nuPlan`, and `NAVSIM`
5. what order to implement the next milestones in

## 1. Project Goal

This project is an MVP implementation scaffold for the paper thesis:

1. `reactive training matters`
2. `object-relational representation matters`
3. `high-fidelity transfer matters`

The current code is **not** a full autonomous driving stack.

It is a research scaffold for:

1. structured scene tokenization
2. decision-sufficient abstraction
3. object-relational world modeling
4. latent imagination training
5. benchmark-facing evaluation adapters

## 2. Current Status

The following pieces already exist and are runnable:

1. unified token schema
2. synthetic dataset for shape debugging
3. token encoder
4. decision-sufficient abstraction module
5. reactive object-relational world model
6. actor-critic head
7. training loop
8. benchmark adapter skeletons
9. tests

Verified locally:

```bash
python3 code/train_debug.py --config code/configs/debug_mvp.json --epochs 1
python3 -m pytest code/tests
```

## 3. Project Layout

```text
code/
  configs/
    debug_mvp.json
    nuplan_stack_template.json
  docs/
    SERVER_HANDOFF.md
    TOKENIZATION_SPEC.md
  src/
    doorrl/
      adapters/
        base.py
        nuscenes_adapter.py
        nuplan_adapter.py
        navsim_adapter.py
      data/
        synthetic.py
      models/
        abstraction.py
        doorrl.py
        encoder.py
        policy.py
        world_model.py
      training/
        losses.py
        trainer.py
      config.py
      schema.py
      utils.py
  tests/
    test_adapters.py
    test_forward.py
  train_debug.py
  README.md
  pyproject.toml
```

## 4. Benchmark Strategy

This repository is now intentionally aligned with your available assets:

1. `nuScenes` for offline tokenization and world-model pretraining
2. `nuPlan` for primary closed-loop evaluation
3. `NAVSIM` for external transfer evaluation

Recommended division of labor:

### nuScenes

Use for:

1. token schema bring-up
2. offline relation-feature construction
3. world-model pretraining
4. representation ablations

Do not use it as the only evidence for reactive interaction learning.

### nuPlan

Use for:

1. primary closed-loop benchmark
2. `reactive vs non-reactive` comparison
3. policy evaluation
4. interaction metrics

This is the main benchmark for the core paper claim:

`reactive training outperforms replay / non-reactive training`

### NAVSIM

Use for:

1. external transfer evaluation
2. log-derived scenario validation
3. checking whether learned abstractions generalize outside the main benchmark

This should be treated as external evidence, not the primary reactive benchmark.

## 5. What Is Not Implemented Yet

The following are still placeholders or design stubs:

1. real `nuScenes` devkit integration
2. real `nuPlan` devkit integration
3. real `NAVSIM` integration
4. benchmark metric logging
5. true world-model rollout training
6. replay buffer infrastructure
7. holistic-latent baseline
8. object-only baseline
9. high-fidelity visual evaluation adapter
10. distributed or multi-GPU training

## 6. Server Setup Checklist

When moving to the server, do this in order.

### Step 1: Copy or clone the project

Expected working directory example:

```bash
/path/to/project/code
```

### Step 2: Create a Python environment

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ./code
pip install pytest
```

If your server already has a managed conda environment, that is also fine.

### Step 3: Verify the scaffold

```bash
python3 code/train_debug.py --config code/configs/debug_mvp.json --epochs 1
python3 -m pytest code/tests
```

Do not start integrating real benchmarks until both commands pass.

### Step 4: Create a real benchmark config

Use:

```text
code/configs/nuplan_stack_template.json
```

Copy it and fill in:

1. `nuscenes_root`
2. `nuplan_root`
3. `navsim_root`

Suggested filename:

```text
code/configs/server_nuplan_stack.json
```

## 7. Expected Data Roots

The code currently assumes config-driven roots, not hardcoded paths.

Expected fields:

```json
"benchmark": {
  "offline_dataset": "nuscenes",
  "closed_loop_benchmark": "nuplan",
  "external_evaluation": "navsim",
  "nuplan_mode": "closed_loop_reactive",
  "nuscenes_root": "/path/to/nuscenes",
  "nuplan_root": "/path/to/nuplan",
  "navsim_root": "/path/to/navsim"
}
```

The current code does not inspect the filesystem yet. That logic still needs to be implemented inside the adapters.

## 8. Immediate Development Order

Do these tasks in this exact order.

### Phase A: Finish nuScenes tokenization

Implement in:

1. `src/doorrl/adapters/nuscenes_adapter.py`

Goal:

convert nuScenes scenes into the normalized scene schema used by the model.

Minimum fields to expose:

1. ego pose and velocity
2. dynamic objects
3. map / lane context
4. next-step targets
5. relation tokens

Exit condition:

you can build a real `SceneBatch` from nuScenes samples without synthetic data.

### Phase B: Finish nuPlan observation conversion

Implement in:

1. `src/doorrl/adapters/nuplan_adapter.py`

Goal:

convert nuPlan planner observations into normalized scene records and support:

1. non-reactive mode
2. reactive mode

Exit condition:

the same policy interface can consume `nuPlan` scene tokens in both settings.

### Phase C: Add relation features

This is one of the most important scientific steps.

The first stable relation-token set should include:

1. relative position
2. relative velocity
3. distance
4. time-to-collision proxy
5. lane conflict flag
6. visibility / occlusion
7. priority / right-of-way
8. interaction-active flag

Exit condition:

you can compare:

1. `object-only`
2. `object + relation`
3. `object + relation + visibility prior`

### Phase D: Add representation baselines

You need at least these baselines:

1. holistic latent
2. object-only latent
3. object + relation latent
4. object + relation + visibility prior

Exit condition:

you can produce the paper's representation sufficiency table.

### Phase E: Add nuPlan evaluation metrics

Minimum metrics:

1. collision rate
2. route completion
3. near-collision rate
4. merge success
5. yield compliance
6. cut-in robustness

Exit condition:

you can run:

1. `non-reactive train + non-reactive test`
2. `non-reactive train + reactive test`
3. `reactive train + reactive test`

### Phase F: Add NAVSIM transfer evaluation

Implement in:

1. `src/doorrl/adapters/navsim_adapter.py`

Goal:

evaluate whether the representation and policy transfer beyond the primary benchmark.

Exit condition:

you can report transfer metrics in an external benchmark.

## 9. Scientific Center of the Codebase

When development expands, it will be easy to accidentally turn this into a generic simulator or generic driving stack.

Do not let that happen.

Every major addition should strengthen one of these three claims:

1. `reactive training matters`
2. `object-relational representation matters`
3. `high-fidelity transfer matters`

If a proposed module does not support one of those three claims, it is probably out of scope for the first paper version.

## 10. Main Files To Edit First

If continuing on the server, the first files that should change are:

1. [nuscenes_adapter.py](/Users/hb40355/Desktop/期刊/code/src/doorrl/adapters/nuscenes_adapter.py)
2. [nuplan_adapter.py](/Users/hb40355/Desktop/期刊/code/src/doorrl/adapters/nuplan_adapter.py)
3. [base.py](/Users/hb40355/Desktop/期刊/code/src/doorrl/adapters/base.py)
4. [schema.py](/Users/hb40355/Desktop/期刊/code/src/doorrl/schema.py)
5. [debug_mvp.json](/Users/hb40355/Desktop/期刊/code/configs/debug_mvp.json) or a copied server config
6. [nuplan_stack_template.json](/Users/hb40355/Desktop/期刊/code/configs/nuplan_stack_template.json)

## 11. Recommended First Week On Server

### Day 1

1. verify environment
2. run debug training
3. run tests
4. create real config with dataset roots

### Day 2

1. connect nuScenes devkit
2. load one scene
3. convert one sample into normalized tokens
4. save and inspect a small batch

### Day 3

1. add relation token construction
2. verify masks and token counts
3. visualize 3 to 5 scenes for sanity checking

### Day 4

1. connect nuPlan observation conversion
2. test one non-reactive rollout
3. test one reactive rollout

### Day 5

1. add basic metric logging
2. run a tiny benchmark sanity check
3. define baseline configs

## 12. Practical Warning

Do not start by reproducing the full final paper system.

The correct development order is:

1. tokenization correctness
2. adapter correctness
3. representation ablations
4. reactive vs non-reactive evaluation
5. transfer evaluation

Not:

1. giant end-to-end training stack
2. distributed infra
3. perception front-end
4. visual reconstruction integration

## 13. Minimal Success Criteria

The first real milestone should be considered successful only if all of the following are true:

1. a real nuScenes sample can be tokenized
2. a real nuPlan observation can be tokenized
3. the model can run forward on real token batches
4. reactive and non-reactive nuPlan modes are both reachable
5. the representation ablation can be defined in code, even before full-scale training

## 14. One-Line Reminder

The codebase is not trying to learn a faster renderer.

It is trying to learn a **decision-sufficient reactive object-relational latent MDP** for autonomous driving.
