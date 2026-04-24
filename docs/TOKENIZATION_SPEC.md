# DOOR-RL Tokenization Spec

This document defines the normalized scene schema that all real dataset and benchmark adapters should produce.

The purpose is simple:

1. keep the model input contract stable
2. let different sources share one token interface
3. avoid rewriting the whole model when switching benchmarks

## 1. Core Design Principle

All sources should be converted into a single ego-centric structured representation:

1. `ego tokens`
2. `object tokens`
3. `map tokens`
4. `relation tokens`

This is the central abstraction of the project.

## 2. Required Model Inputs

The model consumes a `SceneBatch` with:

1. `tokens` of shape `[B, S, D]`
2. `token_mask` of shape `[B, S]`
3. `token_types` of shape `[B, S]`
4. `actions` of shape `[B, A]`
5. `next_tokens` of shape `[B, S, D]`
6. `rewards` of shape `[B]`
7. `continues` of shape `[B]`

Defined in:

[schema.py](/Users/hb40355/Desktop/期刊/code/src/doorrl/schema.py)

## 3. Normalized Record Contract

Before batching, each adapter should produce a normalized Python record with this structure:

```python
{
  "ego": {...},
  "next_ego": {...},
  "objects": [{...}, ...],
  "map_elements": [{...}, ...],
  "relations": [{...}, ...],
  "action": [a0, a1],
  "reward": float,
  "continue": float,
}
```

The actual batching into tensors is handled by:

[base.py](/Users/hb40355/Desktop/期刊/code/src/doorrl/adapters/base.py)

## 4. Token Categories

Current token categories are:

1. `ego`
2. `vehicle`
3. `pedestrian`
4. `cyclist`
5. `map`
6. `signal`
7. `relation`
8. `pad`

Defined in:

[schema.py](/Users/hb40355/Desktop/期刊/code/src/doorrl/schema.py)

## 5. Current Raw Feature Layout

The current normalized converter writes the following feature slots:

1. `0`: x
2. `1`: y
3. `2`: vx or speed
4. `3`: vy
5. `4`: length
6. `5`: width
7. `6`: risk
8. `7`: visibility
9. `8`: ttc
10. `9`: lane_conflict
11. `10`: priority
12. `11`: distance
13. `12`: heading
14. `13`: is_interactive
15. `14`: token_type_id

The model config currently uses `raw_dim = 40`, so the remaining dimensions are still free for expansion.

## 6. Ego Token Minimum Fields

The `ego` token should include at least:

1. x
2. y
3. speed or vx
4. vy
5. heading
6. size

If the source benchmark exposes richer planner state, add it only if it supports the main paper questions.

## 7. Object Token Minimum Fields

Each dynamic object should include:

1. token type
2. relative x
3. relative y
4. relative vx
5. relative vy
6. length
7. width
8. visibility
9. optional risk proxy

Keep all coordinates in the ego-centric local frame whenever possible.

## 8. Map Token Minimum Fields

The first map-token version should include:

1. local lane centerline points
2. lane boundary or drivable hints
3. optional priority or traffic rule context

Do not overcomplicate this layer in the first pass.

## 9. Relation Token Minimum Fields

This is the most important part of the representation.

The first stable relation-token design should include:

1. source-relative position
2. source-relative velocity
3. distance
4. time-to-collision proxy
5. lane conflict indicator
6. visibility or occlusion score
7. interaction priority
8. interaction active flag

This is the feature group most closely tied to the paper's scientific novelty.

## 10. Source-Specific Notes

### nuScenes

Use for:

1. offline object extraction
2. map token construction
3. relation-token construction
4. next-step supervision

### nuPlan

Use for:

1. planner observation conversion
2. closed-loop token extraction
3. reactive vs non-reactive experiments

### NAVSIM

Use for:

1. external evaluation tokenization
2. transfer benchmarking

## 11. Action Encoding

The current code uses a simple action vector of dimension `2`.

This is intentionally minimal.

On the server, one of these two strategies should be chosen and kept consistent:

1. low-level control style:
   - acceleration
   - steering or yaw-rate
2. planner command style:
   - compact motion command embedding
   - short-horizon trajectory embedding

For `nuPlan`, a planner-oriented action encoding is likely the better long-term choice.

## 12. Reward And Continue Fields

Even if the benchmark does not expose rewards directly, the normalized record should still populate:

1. `reward`
2. `continue`

For early bring-up, these can be proxy targets.

Examples:

1. progress reward
2. collision penalty
3. route adherence reward
4. terminal on collision

## 13. Sanity Checks For Every New Adapter

Before using a new adapter in training, verify:

1. ego token is always present
2. token counts never exceed configured limits
3. `token_mask` aligns with valid entries
4. padded tokens are zeroed
5. token types are correct
6. next-step targets exist
7. actions have the configured dimension

## 14. Required Visualization Before Training

Before trusting any token pipeline, visualize at least:

1. ego pose
2. dynamic objects
3. map elements
4. relation edges or relation summaries

Do this for:

1. one nuScenes sample
2. one nuPlan non-reactive observation
3. one nuPlan reactive observation

If the visualization looks wrong, do not continue to training.

## 15. What Should Stay Stable

Try to keep these stable across benchmarks:

1. ego-centric coordinates
2. token category semantics
3. relation feature meanings
4. action dimensionality or action-encoding protocol

This is what makes cross-benchmark comparison meaningful.

## 16. What Can Change

The following may evolve without breaking the project:

1. exact feature count beyond the first 15 dimensions
2. relation feature formulas
3. map-token density
4. reward shaping
5. action encoding detail

## 17. Final Rule

Do not keep adding features just because a benchmark exposes them.

Only add a feature if it supports one of these:

1. reactive interaction modeling
2. decision sufficiency
3. transfer evaluation
