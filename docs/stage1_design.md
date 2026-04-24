# Stage 1 — Minimum Imagination RL Design (DOOR-RL)

*Status: design draft, not implemented. Needs sign-off on §0-§3 before coding.*

---

## 0. Goal (one sentence)

Demonstrate that the **decoupled abstraction from Stage 0 can support latent-space policy learning**, via a small but complete imagination-RL loop, without needing CARLA / 3DGS / nuPlan closed-loop.

What we are **not** doing in this stage:
- No full Dreamer reconstruction / KL / stoch-latent stack. We reuse the deterministic token-space world model.
- No external simulator. All rollouts are inside the learned world model.
- No large model scale-up. We keep `model_dim=128`, 2 transformer layers.
- No new data. We reuse the 700-scene nuScenes split from Stage 0.

## 1. Claim we want to test

*Under a fixed multi-step imagination budget (K=5), a policy trained with decoupled abstraction achieves higher imagined return, lower imagined collision, and more stable multi-step latent rollout than policies trained on top of naive Object+Relation, Object-only, or Holistic representations.*

Three numbers support this claim:

| Metric | Direction | Stage-1 definition |
|---|---|---|
| `Latent Return` | ↑ | Mean of Σ_{t=0..K-1} r̂_t under the trained policy, r̂_t from the world-model reward head + task shaping (see §2.3). |
| `Imagined Collision Rate` | ↓ | Fraction of imagined rollouts where max_t σ(collision_pred_t) > 0.5. |
| `Rollout Stability` | ↓ | ‖latent_{t+1} − latent_t‖ / ‖latent_t‖ averaged over t ∈ [1, K−1], measured on held-out val scenes, BC action sequence. Captures "does the world model diverge in imagination." |

## 2. Architecture

### 2.1 What's new

Two small files, one optional:

- `src/doorrl/imagination/imagination.py` (new) — one function `imagine_trajectory(model, batch, horizon, deterministic)`.
- `src/doorrl/imagination/task_reward.py` (new) — one function `task_reward(batch_like, action) -> Tensor[B]`.
- `src/doorrl/training/trainer_stage1.py` (new) — `ImaginationTrainer` subclassing `DoorRLTrainer`, adds AC loss on the imagined trajectory.
- (Optional) `src/doorrl/imagination/latent_rollout.py` — a slightly cheaper pure-latent rollout that skips re-encoding. Start without; add only if token-space re-encoding is a wallclock bottleneck.

Nothing else changes. The Stage 0 model (encoder, abstraction, WM, actor-critic head) is reused **as-is**.

### 2.2 Imagination loop

```
Input: batch (real first frame, B samples)
latent_0   = encode(tokens_0)                     # [B, S, D]
absto_0    = abstract(latent_0, mask_0)           # selected_tokens [B, K, D]
policy_0   = actor_critic(absto_0.global_latent)  # action_mean, log_std, value

for t in 0..K-1:
    action_t  = sample(policy_t)                  # stochastic during training
    wm_out_t  = world_model(absto_t.selected_tokens, absto_t.selected_mask, action_t)
    # predicted_next_tokens in raw 40-dim space [B, K, 40]

    # Write predictions back into the 97-slot layout, preserving the
    # *non-selected* tokens from t (treated as static "other-agent" context).
    tokens_{t+1}       = tokens_t.clone()
    tokens_{t+1}[sel_ids_t] = wm_out_t.predicted_next_tokens

    latent_{t+1}       = encode(tokens_{t+1}, token_types_0)   # types don't change
    absto_{t+1}        = abstract(latent_{t+1}, mask_0)
    policy_{t+1}       = actor_critic(absto_{t+1}.global_latent)

    rewards[t]         = wm_out_t.predicted_reward + task_reward(...)
    collisions[t]      = sigmoid(wm_out_t.predicted_collision)
    continues[t]       = sigmoid(wm_out_t.predicted_continue)

return imagined trajectory:
    actions  [B, K, A]       log_probs [B, K]
    values   [B, K+1]        rewards   [B, K]
    collisions[B, K]         continues [B, K]
    stability: ‖absto_{t+1}.global_latent − absto_t.global_latent‖
```

Key choices:
1. **Re-encode at each step** (option *a* in the trade-off): slow (~2×) vs pure-latent rollout but reuses Stage 0 encoder verbatim, no new consistency loss, and prevents silent architectural drift between imagination and real-data evaluation.
2. **Non-selected tokens are frozen across steps.** The world model only predicts `K` slots. For the remaining `S−K` tokens we keep t=0 values (they represent "background agents the abstraction judged less decision-relevant"). This is a deliberate simplification; swap to also rolling them forward only if ablations show it matters.
3. **Token types are fixed across imagination** (no PED becomes VEH). This keeps the abstraction path stable.
4. Rollouts are **single-trajectory** (no multi-sample tree). Horizon `K=5` (≈ 0.5 s at 10 Hz).

### 2.3 Task reward (imagined only)

There's no ground-truth RL reward on nuScenes — Stage 0 had `reward ≡ 0`. For Stage 1 we define a shaped imagined reward deliberately simple and interpretable:

```
r_t = w_prog * v_forward(ego_t+1)
    - w_coll * sigmoid(collision_pred_t)
    - w_act  * ‖action_t‖²
```

Default weights: `w_prog=1.0, w_coll=5.0, w_act=0.01`.

- `v_forward` is token-dim 2 (`vx` of ego in ego-relative frame, from `predicted_next_tokens[:, 0, 2]`).
- No BC-in-reward component; BC is a separate training signal (§3), not a reward shaper.

This reward is additive on top of the world-model's reward head output. In practice the reward head will contribute near zero until we train it (Stage 0 never did), so initially `r_t ≈ task_reward`.

### 2.4 Actor-Critic loss (imagination gradient)

Use GAE-lambda with bootstrapping from the final value:

```
values   = V(absto_0..K)                    # [B, K+1]
deltas_t = r_t + γ * continues_t * V_{t+1} - V_t
adv_t    = Σ_{l=0..K-t-1} (γ λ continues)^l δ_{t+l}
ret_t    = adv_t + V_t
```

Losses:

```
L_actor   = -mean( log π(action_t | s_t) * stop_grad(adv_t) )
              - β_ent * mean( H(π(.|s_t)) )
L_critic  = mean( (V(s_t) - stop_grad(ret_t))^2 )
L_sanity  = Stage 0 losses on the real t=0 batch (obs/reward/continue/collision)
            keeps WM from collapsing while AC adjusts it.
```

Total: `L = L_actor + α_c * L_critic + α_s * L_sanity`. Defaults: `γ=0.97, λ=0.95, β_ent=0.01, α_c=0.5, α_s=1.0`.

**Crucially**, gradients flow **through the imagined rollout into the world model** — this is the full Dreamer-style imagination. If we want a fairer "model-free RL" baseline we detach the WM in that condition (see §3).

## 3. Experimental conditions (5, all seed-matched)

| ID | Representation | Policy learns from | Rollout? | Wallclock / seed (est.) |
|---|---|---|---|---|
| `bc` | object_only (frozen) | real actions (MSE) | no | ~20 min |
| `ac1` | object_only (frozen) | 1-step real-transition AC (TD(0)) | K=1, real next-state | ~30 min |
| `wm_holistic` | `holistic_16slot` (init from Stage 0 ckpt) | K=5 imagination AC + sanity | yes | ~60-90 min |
| `wm_object` | `object_only` (init from Stage 0 ckpt) | K=5 imagination AC + sanity | yes | ~60-90 min |
| `wm_decoupled` | `object_relation_decoupled_visibility` (init) | K=5 imagination AC + sanity | yes | ~60-90 min |

`bc` and `ac1` are representation-agnostic *floors* — they should **not** benefit from imagination, so the gap between them and `wm_*` measures how much imagination contributes at all.

`wm_holistic` vs `wm_object` vs `wm_decoupled` measures **which representation best supports imagination RL**, holding everything else fixed.

Seeds: 7, 42, 2026 (same as Stage 0). Report mean ± std.

Training budget per run: 10 epochs over the 560 train scenes, batch 128 (reusing Stage-0 optimisations). 3 seeds × 5 conditions = 15 runs, worst case ~75 min × 15 × (1/3 for seed-parallel) = ~6 hours wallclock on one H20.

## 4. Evaluation protocol

Run **on the 140 val scenes held out from Stage 0**. Per variant, per seed:

1. Encode each val sample; use the **trained** policy (deterministic, `action_mean`) to rollout K=5 imagined steps.
2. Record `latent_return`, `imagined_collision_rate`, `rollout_stability`.
3. Aggregate over val scenes → one number per (seed, condition, metric). Over 3 seeds → mean ± std.

One supplementary sanity check: **imagined collision rate vs Stage-0 IntRec@1m correlation**. We expect representations that score well on Stage 0 Interaction Recall to also produce safer imagined rollouts; a monotone relationship (decoupled > object_only > naive) on *both* Stage 0 and Stage 1 is the strongest single-paragraph story.

## 5. Open design questions

These are genuinely unsettled; flagging them so you can push back before implementation.

1. **Sanity loss weight α_s.** Too high → WM can't adapt to imagination; too low → WM drifts and imagined rollouts lose grounding. Start with 1.0, consider a schedule (decay from 1.0 → 0.1 over training).
2. **Entropy bonus schedule.** Keeping `β_ent=0.01` constant is simple but may under-explore; Dreamer typically anneals. Start constant, revisit if policies collapse.
3. **Should we also keep the `ac1` baseline using real actions for bootstrapping?** Alternative: pure model-free PPO-style. PPO on offline nuScenes is awkward (no reset). Current proposal (TD(0) on real (s, s')) is the most honest single-step MFRL we can do on offline data — still worth a sentence in the paper.
4. **Token-space vs pure-latent rollout.** See §2.2 choice 1. If wallclock becomes the bottleneck with 5-step imagination × 15 runs, swap to pure-latent rollout; otherwise keep it explicit for interpretability.
5. **Frozen vs trainable representation during Stage 1.** Proposed: trainable (encoder + abstraction + WM + policy all fine-tune together). Alternative: freeze encoder + abstraction, only train WM + policy. The latter is stricter — any policy-quality gap is then *purely* due to what the frozen representation supplies. *This might actually be the cleanest experiment to pair with Stage 0.* Please flag which one you want.

## 6. Minimum deliverables

- [ ] `src/doorrl/imagination/imagination.py` (~80 lines)
- [ ] `src/doorrl/imagination/task_reward.py` (~30 lines)
- [ ] `src/doorrl/training/trainer_stage1.py` (~120 lines, extends `DoorRLTrainer`)
- [ ] `src/doorrl/evaluation/stage1_metrics.py` (~100 lines)
- [ ] `run_stage1_table4.py` (~150 lines, mimics `run_stage0_table3.py` structure)
- [ ] `scripts/run_stage1_3seeds.sh` (orchestrates 5 conditions × 3 seeds, reuses `run_fast_3seeds.sh`'s concurrency pattern)
- [ ] Sanity test: 2 scenes × 2 epochs × 1 condition, wallclock < 3 min; must show non-degenerate rollouts (return ≠ 0, loss decreases).

## 7. File map (projected)

```
src/doorrl/imagination/
  __init__.py
  imagination.py           # imagine_trajectory(model, batch, K, deterministic)
  task_reward.py           # task_reward(batch_like, action)
src/doorrl/training/
  trainer_stage1.py        # ImaginationTrainer(DoorRLTrainer)
  losses_stage1.py         # GAE + actor/critic losses
src/doorrl/evaluation/
  stage1_metrics.py        # latent_return, imagined_collision_rate, rollout_stability
run_stage1_table4.py       # orchestrator, --condition {bc, ac1, wm_holistic, wm_object, wm_decoupled, all}
scripts/
  run_stage1_3seeds.sh     # 5 × 3 pipeline
  aggregate_stage1_seeds.py
docs/
  stage1.md                # the post-experiment report (to be written after runs)
```

---

## Decision checklist before implementation

Please confirm / adjust these, then I'll start writing the modules:

- [ ] **Horizon K = 5.** OK, or do you want K=3 (cheaper) / K=10 (richer but risk drift)?
- [ ] **Task reward weights** `w_prog=1.0, w_coll=5.0, w_act=0.01`. OK, or lean more on safety / comfort?
- [ ] **Re-encode at each step** (§2.2 choice 1) vs pure-latent rollout. Default: re-encode.
- [ ] **Frozen Stage-0 backbone** or trainable during Stage 1 (§5 Q5). Default: trainable end-to-end.
- [ ] **5 conditions as listed** (bc / ac1 / wm_holistic / wm_object / wm_decoupled). Add `wm_naive` (object_relation) to make the diagnostic story match Stage 0's 6 variants? I'd say *yes, add it* — costs one more run but makes the Stage-1 table exactly mirror Stage-0 and strengthens "naive collapses under imagination too". 6 conditions × 3 seeds ≈ +1.5 h.
- [ ] **Budget**: ~6-8 hours wallclock for the full 3-seed, 5-6-condition table. OK, or want a cheaper first pass (1 seed, 3 conditions) to sanity-check the imagination loop?
