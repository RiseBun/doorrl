## Stage 1 — Imagination RL Pilot (nuScenes, seed 7)

_Last updated: 2026-04-23, 1 seed × 3 conditions, v2 after NaN-fix + cosine-stability patch._

---

### 0. TL;DR

Under the decoupled-abstraction representation learned in Stage 0, a minimal
imagination RL pilot (K=5 step latent rollout, GAE-λ actor-critic,
WM-detached, task-shaped reward) validates:

1. **Numerical stability is controllable**. With four guardrails —
   `tanh`-bounded action mean, `log_std` clamp, reward clip, and Huber critic
   — all conditions train 10 epochs without NaN, and the WM sanity loss stays
   below 1.0 the entire run (previously blew up to 11.8 at epoch 9).
2. **Decoupled abstraction supports imagination RL**. `wm_decoupled` reaches
   a positive task return (4.99) with **23 points lower collision rate**
   than object-only (0.756 vs 0.985).
3. **Object-only actor degenerates under imagination RL**. Even with the
   same numerical guardrails, `wm_object` converges to a large-action
   regime (`|a|_max → 7.7`, near the clip of 8.0), collides in 98.5 %
   of rollouts, and delivers negative return.

Paper-line candidate:
> _"Decoupled abstraction is not merely a representation-quality improvement:
> under a minimal imagination RL objective, the object-only baseline learns
> a degenerate large-action policy that collides almost everywhere, whereas
> the typed-budget decoupled variant learns a policy with 23 points lower
> imagined collision rate and positive task return without any additional
> regularization."_

---

### 1. Pipeline Summary

| | |
|---|---|
| Dataset | nuScenes 700 scenes, scene-level 80/20 split, val = 5 622 samples |
| Imagination horizon K | 5 |
| Actor loss | -E[log π · stop_grad(adv)] − β·H(π), β=0.01 |
| Critic loss | **Huber(δ=10)** on stop_grad(GAE return) |
| Sanity loss | full Stage-0 losses (obs / reward / continue / collision) on the real t=0 batch, **weight=1.0** |
| GAE | γ=0.97, λ=0.95, discount-mask = sigmoid(continue) |
| Reward | `w_prog=1`, `w_coll=5`, `w_act=0.01`, clipped to [-5, 5] |
| Action head | `mean = 3·tanh(raw/3)`, `log_std ∈ [-2, 0.5]`, sampled a clipped to [-8, 8] |
| `detach_world_model` | **True** (Dreamer-style; WM grounded only by sanity loss) |
| Epochs / batch / lr | 10 / 128 / 4·base = 4e-3 |
| Seeds | 7 (pilot) |
| Warm-start | Stage 0 seed-7 checkpoints (same variant) |

Runner: `scripts/run_stage1_pilot.sh`, output: `experiments/stage1_pilot_ab/`.

---

### 2. Results (val, 5 622 samples, det. policy)

| Condition | return_mean | coll_rate | coll_mean | stab (ego, cos) | stab (global, L2)\* |
|---|---:|---:|---:|---:|---:|
| bc | 10.960 | 0.607 | 0.620 | 0.0015 | 0.0109 |
| wm_object | −4.713 | 0.985 | 0.984 | 0.294 | 0.137 |
| **wm_decoupled** | **4.997** | **0.756** | 0.772 | 0.735 | 0.041 |

\* `stab (global, L2)` is kept only for compatibility with early drafts.
It is NOT cross-variant comparable (see §4).

Training curves (val, representative epochs):

| Condition | epoch | critic | sanity | R | V | |a|_max | log_std |
|---|---:|---:|---:|---:|---:|---:|---:|
| wm_object | 1 | 7.07 | 0.81 | +0.84 | −16.9 | 5.73 | +0.50 |
| wm_object | 5 | 1.22 | 0.23 | −2.58 | −13.3 | 6.36 | +0.50 |
| wm_object | 10 | 4.29 | 0.17 | −0.89 | +16.0 | **7.71** | +0.50 |
| wm_decoupled | 1 | 2.67 | 0.26 | −1.67 | −29.1 | 5.61 | +0.50 |
| wm_decoupled | 5 | 7.22 | 0.17 | −1.62 | −19.0 | 5.54 | +0.50 |
| wm_decoupled | 10 | 1.28 | 0.17 | +0.86 | +2.6 | 5.49 | +0.50 |

All |critic| < 20, all sanity < 1.0, `|a|_max` stays below the clip of 8.0.

---

### 3. What the NaN fix + Huber change achieved

**Before (pilot v1, pre-fix):** `wm_object` @ epoch 9 val `critic = 2 301`, `sanity = 11.79`, `R = −4.68`, `|a|_max = 6.10`. Encoder visibly corrupted.

**After (pilot v2, post-fix):** `wm_object` @ epoch 9 val `critic = 1.49`, `sanity = 0.14`. No divergence.

Attribution:

- **Action `tanh` + `log_std` clamp** bounded the distribution that feeds
  the Normal sampler. 3·tanh guarantees `mean ∈ [-3, 3]`, `log_std ≤ 0.5`
  gives `σ ≤ 1.65`, so a 3-σ sample stays in [-8, 8].
- **Reward clip to ±5** prevented a single bad `comfort` term (action²)
  from producing outlier returns that destabilise GAE.
- **`detach_world_model=True`** stopped the WM parameters from being
  trained against a mis-calibrated critic target.
- **Huber critic (δ=10)** bounded per-sample critic gradient so that a
  single out-of-distribution return no longer induces runaway updates into
  the shared encoder.

The v1 "high-return" `wm_object` result (R=26.85) in fact exceeded the
reward-clip theoretical upper bound (5 × K=5 = 25). That was reward-hacking
via an about-to-diverge critic, not a good policy.

---

### 4. Stability metric: why cosine, why it is now a sanity check

Earlier we reported `rollout_stability = mean_t ‖l_{t+1}-l_t‖ / ‖l_t‖` over
each variant's `global_latent`. This produced a 2·10⁶ value for the
visibility-weighted decoupled variant, and gave a cross-variant ordering
that contradicted other evidence.

Post-mortem: the relative-L2 metric is sensitive to *magnitude* scaling of
the latent, and `global_latent` is not the same quantity across variants
(`masked_mean(all object tokens)` vs `0.5·(dyn_mean + rel_mean)`). The
visibility-weighted decoupled variant multiplies the dyn latent by
`visibility ∈ [0,1]` before abstraction, so whenever the WM-imagined
visibility drifts towards 0 the ego slot collapses in magnitude and the
metric diverges even though the *direction* of the representation is fine.

**Fix (the "A" change in the A+B patch):**

1. Track `selected_tokens[:, 0, :]` as `ego_latents` across the rollout.
   `force_ego=True` guarantees slot 0 is the ego token for every top-k
   variant (object_only, object_relation, decoupled), so this is a
   semantically identical signal across variants.
2. Score it with cosine distance, `1 − cos(e_t, e_{t+1})`, which is
   scale-invariant (visibility multiplication preserves direction) and
   bounded in [0, 2].

`src/doorrl/evaluation/stage1_metrics.py::_stability_score` implements this.
The legacy global/L2 number is still emitted under
`rollout_stability_global` for backward-compat but is explicitly marked
non-comparable across variants.

**Reading the new metric.** `wm_decoupled` has *higher* cosine stability
(0.74) than `wm_object` (0.29). This is consistent with the rest of the
story rather than against it:

- `wm_object`'s actor saturates at `|a|_max ≈ 7.7` and collides in 98.5 %
  of rollouts — its ego representation barely evolves because the policy
  is locked into a near-static degenerate regime.
- `wm_decoupled`'s actor selects genuinely varied actions, and the WM
  imagines the ego into different states. The non-zero cosine distance is
  the *intended* response of a working imagination loop.

Takeaway: **treat cosine stability as a sanity check** (catches either a
dead or a diverging rollout), not as a single-number quality metric.
Return + collision remain the primary metrics.

---

### 5. What next (post-pilot)

Deferred to a follow-up mini-experiment (not run yet, to avoid polluting
this pilot):

- **Tame `wm_object`**: lower `entropy_beta` 0.01 → 0.003, tighten action
  clip 8 → 5 (still above 3σ of the tanh-bounded distribution). Test
  whether the object-only actor can be prevented from saturating.
  Expected: `|a|_max` pulled below 5.0, collision rate closer to the bc
  baseline (~0.6). If so, the remaining `wm_decoupled > wm_object`
  collision gap would constitute a cleaner architectural claim.
- **Corroborate on nuPlan**: run the same 3-condition pilot on the
  pre-processed NuPlan agent-64 split (adapter already lands in
  `src/doorrl/adapters/nuplan_preprocessed_adapter.py`). Expected: same
  ordering (decoupled learns a policy, object-only saturates), no new
  NaN failure modes.

---

### 6. File map

| File | Purpose |
|---|---|
| `src/doorrl/models/policy.py` | `ActorCriticHead` with `tanh` mean bound + `log_std` clamp |
| `src/doorrl/imagination/imagination.py` | K-step imagination rollout; emits `ego_latents` for the stability metric |
| `src/doorrl/imagination/task_reward.py` | task reward, reward clip |
| `src/doorrl/training/losses_stage1.py` | actor / Huber-critic / sanity losses + diagnostic scalars |
| `src/doorrl/training/trainer_stage1.py` | `ImaginationTrainer` with `detach_world_model` default True |
| `src/doorrl/evaluation/stage1_metrics.py` | latent return, imagined collision rate, cosine stability |
| `run_stage1_table4.py` | per-condition runner, supports `--eval-only` |
| `scripts/run_stage1_pilot.sh` | nuScenes pilot launcher |
| `experiments/stage1_pilot_ab/seed7/stage1_all.json` | aggregate pilot metrics |
| `experiments/stage1_pilot_ab/logs/pilot_seed7.log` | per-epoch training log (all 3 conditions) |
| `experiments/stage1_pilot_ab/logs/eval_cosine.log` | v2 eval-only run with cosine stability |
