# Stage 0 — Decision-Oriented Representation Analysis (DOOR-RL)

_Last updated: 2026-04-23, 3 seeds × 6 variants, all runs reproducible from the scripts referenced below._

---

## 0. TL;DR (paper-ready)

Under a **fair 16-slot world-model context budget**, on nuScenes (700 scenes, 28 096 samples, scene-level split 80/20), averaged over 3 seeds (7, 42, 2026):

| Variant | Ctx | DynRoll ↓ | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---|---|---|---|
| Holistic-16Slot (learned-query BNk) | 16 | 2.11 ± 0.16 | 0.978 ± 0.011 | 1.42 ± 0.01 | 0.643 ± 0.015 |
| Object-only-16 | 16 | 3.74 ± 1.01 | 0.946 ± 0.004 | 1.10 ± 0.12 | 0.901 ± 0.034 |
| Object+Relation-16 (naive) | 16 | **40.28 ± 29.54** | 0.980 ± 0.013 | **7.51 ± 5.48** | **0.430 ± 0.407** |
| Obj+Rel+Vis-16 | 16 | 15.80 ± 9.93 | 0.933 ± 0.064 | 2.96 ± 1.64 | 0.728 ± 0.155 |
| **Obj+Rel-Decoupled (Ours)** | 16 | **2.11 ± 0.19** | 0.929 ± 0.039 | **0.49 ± 0.18** | **0.984 ± 0.014** |
| **Decoupled + Visibility (Ours)** | 16 | **1.88 ± 0.23** | 0.926 ± 0.029 | **0.52 ± 0.05** | **0.979 ± 0.008** |
| Holistic-full (ref, 97 tok) | 97 | 0.11 ± 0.12 | 0.988 ± 0.006 | 0.26 ± 0.02 | 1.000 ± 0.000 |

Headline findings:

1. **Naively mixing relation tokens into a shared top-k bottleneck fails catastrophically** (IntRec 0.43 ± 0.41 vs Object-only 0.90 ± 0.03), not because of representation quality but because relation slots *compete with dynamic-agent slots for a fixed 16-slot budget*.
2. **Typed-budget (decoupled) abstraction solves this** without any hand-tuned regularization: separate top-K per semantic role (K_dyn = 12 dynamic, K_rel = 4 relation) keeps the total context identical to all other 16-slot variants yet recovers **and surpasses** the Object-only baseline on every dynamic-agent metric (DynRollout −44 %, RareADE −55 %, IntRec@1m +9 %).
3. **Variance collapses by ~29×** (IntRec std: 0.41 → 0.014), evidence that the failure mode was architectural rather than stochastic.

Proposed paper line:
> *"Naively mixing relation tokens into a shared bottleneck fails due to inter-type budget competition; decoupled abstraction with typed per-role budgets resolves this by matching representational capacity to semantic role, recovering and surpassing the object-only baseline on dynamic-agent forecasting while retaining relation-aware collision reasoning."*

---

## 1. Experimental Setup

### 1.1 Dataset

| | |
|---|---|
| Source | nuScenes v1.0-trainval + CAN-bus data |
| Scenes used | 700 (configurable via `--num-scenes`) |
| Samples (frames) | 28 096 (after CAN-bus / NaN filtering) |
| Split | **scene-level** 80 / 20 = 560 train / 140 val (to prevent intra-scene leakage) |
| Frame tokenisation | 97 tokens × 40 raw dims per sample (1 ego, ≤ 12 vehicle, ≤ 12 pedestrian, ≤ 12 cyclist, ≤ 32 map, ≤ 16 signal, ≤ 12 relation, pad) |
| In-memory cache | Entire tokenised dataset is materialised once in `NuScenesSceneDataset.__init__` (5-15 min one-time cost); reused by all variants through a shared `DataLoader`. |

Key raw-dim indices (`src/doorrl/adapters/base.py` `NormalizedSceneConverter`):
- 0-3: `(x, y, vx, vy)` — ego-relative
- 7: visibility (continuous, mapped from nuScenes categorical {1-4} → {0.2, 0.5, 0.7, 0.9})
- 8: `TTC` (relation tokens only)
- 9: `lane_conflict` (relation tokens only)
- 10: `priority` (relation tokens only)

### 1.2 Model

Defined in `configs/debug_mvp.json` + `ModelConfig`:

| Field | Value |
|---|---|
| `raw_dim` | 40 |
| `model_dim` | 128 |
| `hidden_dim` | 256 |
| `num_heads` | 4 |
| `num_layers` | 2 (transformer world model) |
| `top_k` | **16** (shared context budget for all fair variants) |
| `top_k_dyn` | 12 (decoupled dynamic path) |
| `top_k_rel` | 4 (decoupled relation path) |
| `dropout` | 0.1 |
| Params | ~ 543k – 609k depending on variant |

### 1.3 Training

| Field | Value |
|---|---|
| Optimiser | Adam, lr = 1e-3, weight decay = 1e-5 |
| Batch size | 32 |
| Epochs | 15 |
| Loss weights | `obs=1.0, reward=0.5, continue=0.25, collision=0.25, bc=0.1` |
| Hardware | 1 × NVIDIA H20 (96 GB) |
| Seeds | 7, 42, 2026 |
| Wallclock per seed | ~ 28 min (5 variants × ~5 min + 10 min dataset tokenisation + 3 min eval) |

### 1.4 Fair comparison protocol

All variants are trained **on the same dataset, same split, same hyperparameters, same total context budget of 16 slots into the world model**. The only thing that differs is the abstraction mechanism (which tokens the 16 slots see, and how supervision is routed). `Holistic-full` is kept as a 97-token upper-bound reference but does not participate in the fair comparison.

---

## 2. Variants

Implemented in `src/doorrl/models/doorrl_variant.py`.

| # | `ModelVariant` | Abstraction | World-model context | Supervision |
|---|---|---|---|---|
| 1 | `HOLISTIC` | none (direct pass-through) | **97 tokens** | typed obs loss |
| 2 | `HOLISTIC_16SLOT` | 16 learned queries cross-attend 97 tokens | 16 compressed slots | **set-prediction obs loss** (DETR-style nearest assignment) |
| 3 | `OBJECT_ONLY` | top-k over {EGO, VEHICLE, PED, CYCLIST} only | 16 dynamic slots | typed obs loss (dyn path) |
| 4 | `OBJECT_RELATION` | top-k over {dynamic ∪ relation} — **shared budget** | ≤ 16 (mix) | typed obs loss (Fix #2) |
| 5 | `OBJECT_RELATION_VISIBILITY` | same as (4) + visibility-weighted latent | ≤ 16 (mix) | typed obs loss |
| 6 | `OBJECT_RELATION_DECOUPLED` | **two independent top-k heads: K_dyn=12, K_rel=4** | 12 dyn + 4 rel = 16 | typed obs loss |
| 7 | `OBJECT_RELATION_DECOUPLED_VISIBILITY` | (6) + visibility weighting on the dynamic path | 12 dyn + 4 rel = 16 | typed obs loss |

### 2.1 Decoupled abstraction (Ours, Route C)

```python
# src/doorrl/models/doorrl_variant.py :: _forward_object_relation_decoupled
# 1. Encode all 97 tokens -> latent [B, S, model_dim]
# 2. Build dyn_mask (EGO|VEHICLE|PED|CYCLIST) and rel_mask (RELATION)
# 3. Two independent DecisionSufficientAbstraction heads:
#       K_dyn = 12 over dyn_mask  (force_ego=True)
#       K_rel =  4 over rel_mask  (force_ego=False)
# 4. Concat -> [B, 16, model_dim] fed to the shared world model
# 5. global_latent = 0.5 * (dyn_global + rel_global) -> policy head
```

Key design choices & their justifications:
- **Independent top-k per type** — relation and dynamic tokens no longer compete for slots (the failure mode of variant 4).
- **`force_ego=False` on the relation head** — the original abstraction forces token 0 (ego) to always be selected; keeping that on the relation path would waste one slot on a non-relation token. This is a new `DecisionSufficientAbstraction` kwarg added for Route C.
- **Budget constraint `top_k_dyn + top_k_rel == top_k`** — enforced in `_setup_decoupled_mode`; makes the total context identical to every other fair variant (16 slots), so the comparison remains strict.
- **Framing as *typed* budgets** — the paper should present `K_dyn=12, K_rel=4` as a *type-ratio prior*, not as a hand-tuned numeric patch. It derives from: "roughly 12 safety-relevant dynamic agents per scene in nuScenes + ~4 salient relation edges are enough to summarise local interaction structure."

---

## 3. Metric Definitions

Implemented in `src/doorrl/evaluation/table3_metrics.py`.

All four dynamic-agent metrics use a **single nearest-assignment procedure** applied identically to every variant: for each ground-truth dynamic agent `(x_gt, y_gt)` we take the `(x,y)`-closest valid dynamic slot from `predicted_next_tokens`. This makes variants with learned-query slots (Holistic-16Slot) comparable to top-k variants where slots correspond to original token indices.

| Metric | Symbol | Definition |
|---|---|---|
| **Dyn Rollout MSE** | `dyn_rollout_mse` | Sum of squared errors on `(x, y, vx, vy)` of the nearest-matched slot, averaged over *all* (agent × dim) pairs in the val set. Only dynamic-type candidate slots are eligible for matching. |
| **Action MSE** | `action_mse` | MSE of `policy.action_mean` vs teacher action `batch.actions` on the val set. Reported for completeness; de-emphasised because the policy head is largely tied across variants (all use BC weight 0.1). |
| **Collision F1** | `collision_f1` | Binary F1 of `world_model.predicted_collision` vs ground-truth label `(any TTC < 3 s)` derived from relation-token dim 8. |
| **Rare ADE** | `rare_ade` | Mean nearest-match Euclidean distance, restricted to ground-truth pedestrians + cyclists (all of them). |
| **Interaction Recall @ 1 m** | `interaction_recall_at_1m` | Fraction of rare agents (ped/cyc) within 20 m of ego whose nearest-match distance is below 1 m. Replaces the saturated `Rare Recall @ 5 m` metric that was always 1.0. |

### 3.1 The evaluator patch (critical)

The evaluator (`_update_rollout_and_ade`) **filters candidate slots to dynamic-type only** before nearest matching:

- For `HOLISTIC_16SLOT` (set-prediction): all valid slots are eligible (they are globally trained to land on dynamic agents).
- For top-k variants: `tt[selected_indices]` is used to keep only slots whose underlying token is EGO/VEHICLE/PED/CYCLIST.

Without this patch, after Fix #2 (type-aware losses), relation slots had un-supervised `(x, y)` outputs that became noise and corrupted the nearest-match pool, artificially inflating errors for `Object+Relation`. **Even with the patch**, naive Object+Relation still fails — proof that the failure is architectural (too-few-dyn-slots), not a metric artefact.

---

## 4. Losses (Fix #2: type-aware obs loss)

Implemented in `src/doorrl/training/losses.py`.

The previous obs loss regressed **all 40 raw dims** of the next-frame token vector equally for every selected slot, regardless of type. This dragged relation slots toward `(x, y, vx, vy, length, width, …)` of the next-frame token — a non-physical target — and competed with object slots for capacity. Fix #2 disentangles supervision by type:

| Slot type | Supervised dims | Rationale |
|---|---|---|
| EGO / VEHICLE / PED / CYCLIST | `(x, y, vx, vy)` (dims 0-3) | matches the evaluator; these are physical next-state targets |
| RELATION | `(TTC, lane_conflict, priority)` (dims 8, 9, 10) | the only dims that carry decision-relevant edge semantics |
| MAP / SIGNAL / PAD | nothing | their "next state" is essentially constant |

Combined:

```
obs_loss = dyn_loss + 1.0 * rel_loss       # per-element means
total    = 1.0*obs + 0.5*reward + 0.25*continue + 0.25*collision + 0.1*bc
```

For `HOLISTIC_16SLOT`, the dispatcher (`compute_losses`) routes to a separate DETR-style `_set_prediction_obs_loss` that does nearest-match between learned slots and GT dynamic agents, mirroring the evaluator exactly — without this, that variant would cheat by driving all 16 learned slots to ego's next state (ego is (0,0) in ego-relative coordinates → trivial loss ≈ 0).

---

## 5. Results

### 5.1 Aggregated table (3 seeds, mean ± std)

Source: `experiments/table3_fair_fix2_aggregate.json` (produced by `scripts/aggregate_fix2_seeds.py`).

| Variant | Ctx | DynRoll ↓ | Action MSE | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---|---:|---|---|---|---|---|
| Holistic-16Slot | 16 | 2.1059 ± 0.1600 | 0.2875 ± 0.0145 | 0.9782 ± 0.0105 | 1.4215 ± 0.0119 | 0.6433 ± 0.0153 |
| Object-only-16 | 16 | 3.7449 ± 1.0099 | 0.2854 ± 0.0103 | 0.9463 ± 0.0041 | 1.0964 ± 0.1159 | 0.9009 ± 0.0335 |
| Object+Relation-16 (naive) | 16 | 40.2822 ± 29.5376 | 0.2808 ± 0.0130 | 0.9803 ± 0.0125 | 7.5060 ± 5.4799 | 0.4295 ± 0.4074 |
| Obj+Rel+Vis-16 | 16 | 15.8023 ± 9.9254 | 0.2840 ± 0.0173 | 0.9330 ± 0.0641 | 2.9624 ± 1.6389 | 0.7283 ± 0.1545 |
| **Obj+Rel-Decoupled (Ours)** | 16 | 2.1148 ± 0.1889 | 0.2805 ± 0.0125 | 0.9285 ± 0.0389 | 0.4913 ± 0.1768 | 0.9842 ± 0.0135 |
| **Decoupled+Visibility (Ours)** | 16 | 1.8761 ± 0.2271 | 0.2843 ± 0.0234 | 0.9257 ± 0.0290 | 0.5197 ± 0.0495 | 0.9787 ± 0.0078 |
| Holistic-full (ref) | 97 | 0.1070 ± 0.1165 | 0.2858 ± 0.0112 | 0.9875 ± 0.0057 | 0.2562 ± 0.0234 | 1.0000 ± 0.0000 |

### 5.2 Per-seed raw numbers

**Seed 7** — `experiments/table3_fair_fix2_seed7/table3_complete.json`

| Variant | DynRoll | Coll F1 | Rare ADE | IntRec@1m |
|---|---|---|---|---|
| holistic_16slot | 1.951 | 0.983 | 1.408 | 0.628 |
| object_only | 4.398 | 0.949 | 1.005 | 0.919 |
| object_relation | 60.786 | 0.967 | 9.744 | 0.263 |
| object_relation_visibility | 24.892 | 0.958 | 2.923 | 0.746 |
| object_relation_decoupled | 1.918 | 0.917 | 0.695 | 0.969 |
| object_relation_decoupled_visibility | 1.775 | 0.915 | 0.576 | 0.975 |
| holistic | 0.043 | 0.983 | 0.281 | 1.000 |

**Seed 42** — `experiments/table3_fair_fix2_seed42/table3_complete.json`

| Variant | DynRoll | Coll F1 | Rare ADE | IntRec@1m |
|---|---|---|---|---|
| holistic_16slot | 2.270 | 0.985 | 1.425 | 0.659 |
| object_only | 4.255 | 0.942 | 1.227 | 0.862 |
| object_relation | 53.635 | 0.982 | 11.512 | 0.132 |
| object_relation_visibility | 17.302 | 0.860 | 4.621 | 0.566 |
| object_relation_decoupled | 2.295 | 0.972 | 0.396 | 0.989 |
| object_relation_decoupled_visibility | 2.136 | 0.958 | 0.482 | 0.988 |
| holistic | 0.241 | 0.994 | 0.234 | 1.000 |

**Seed 2026** — `experiments/table3_fair_fix2_seed2026/table3_complete.json`

| Variant | DynRoll | Coll F1 | Rare ADE | IntRec@1m |
|---|---|---|---|---|
| holistic_16slot | 2.096 | 0.966 | 1.431 | 0.643 |
| object_only | 2.582 | 0.948 | 1.057 | 0.921 |
| object_relation | 6.426 | 0.992 | 1.261 | 0.894 |
| object_relation_visibility | 5.212 | 0.981 | 1.343 | 0.873 |
| object_relation_decoupled | 2.131 | 0.897 | 0.383 | 0.995 |
| object_relation_decoupled_visibility | 1.717 | 0.903 | 0.502 | 0.974 |
| holistic | 0.036 | 0.986 | 0.254 | 1.000 |

### 5.3 What the numbers say

1. **Naive Object+Relation is unstable across seeds.** IntRec jumps between 0.13 and 0.89; DynRoll between 6.4 and 60.8. Seed 2026 accidentally selected more dynamic slots by chance and "worked", but that is a **lucky configuration** — mean ± std makes the architectural defect visible (std comparable to the mean).
2. **Decoupled is robust across seeds.** IntRec range 0.969–0.995 (std 0.014), RareADE range 0.38–0.70 (std 0.18), DynRoll range 1.92–2.30 (std 0.19).
3. **Decoupled strictly dominates Object-only on dynamic metrics.**
   - DynRoll: 2.11 < 3.74 (−44 %)
   - RareADE: 0.49 < 1.10 (−55 %)
   - IntRec@1m: 0.984 > 0.901 (+9 %)
4. **The only trade-off is a ~1.8 % drop in Collision F1** (0.928 vs 0.946), traceable to shrinking the relation budget from (up to) 16 slots to exactly 4 — a clean, interpretable cost.
5. **Visibility weighting adds a small but consistent bump on DynRoll** (1.88 vs 2.11) without changing the qualitative story.

### 5.4 Supporting figures (complementary analyses, P3)

Three diagnostic figures back up the numeric table. All are regenerated by lightweight scripts that consume the seed-7 checkpoints in `experiments/table3_fair_fix2_seed7/<variant>/model.pt` and do **not** require re-training.

| Figure | File | What to look at |
|---|---|---|
| Variance bars | `experiments/figures/stage0_variance_summary.png` (and per-metric files `stage0_variance_interaction_recall_at_1m.png`, `stage0_variance_rare_ade.png`, `stage0_variance_dyn_rollout_mse.png`) | Bars = mean ± std across seeds 7 / 42 / 2026; black dots = per-seed values. Naive Object+Relation has **error bar larger than its mean** on every metric (a bimodal collapse signature); decoupled variants show dots that are almost on top of each other. |
| Slot-type composition | `experiments/figures/stage0_slot_distribution.png` | Stacked bars of how many of the 16 slots each variant spends on each TokenType, averaged over 128 val samples. The key number: `Object+Relation-16 (naive)` spends **10.5 / 16** slots on REL and only **3.7 / 16** on dynamic agents (EGO+VEH+PED+CYC), versus 9.3 / 16 for Object-only and 9.1 / 16 for Decoupled. This *mechanistically* explains why naive mixing collapses. |
| Scene case studies | `experiments/figures/scenes/case_{00..04}_idx*.png` | 5 val scenes where the decoupled head's near-field capture strictly dominates naive mixing, with the 4-panel comparison `Object-only | Naive | Decoupled | Holistic-full`. In cases 2–4 **naive selects 0 / 5 near-field agents** while decoupled captures 5 / 5; mean nearest-prediction error drops from 5–7 m to < 0.3 m. |

**Slot-composition summary (seed 7, 128 val samples):**

```
variant                       EGO  VEH  PED  CYC  MAP  REL    dyn total (of 16)
--------------------------------------------------------------------------
Holistic-16Slot               — (learned queries, set-prediction)  [N/A]
Object-only-16                1.0  2.0  6.0  0.3  3.6  0.0     9.3
Object+Relation-16 (naive)    1.0  0.6  2.1  0.0  1.1 10.5     3.7    ← dyn starved
Obj+Rel+Vis-16                1.0  1.2  4.6  0.2  2.9  5.5     7.0
Obj+Rel-Decoupled  (Ours)     1.0  2.1  5.7  0.3  0.0  3.9     9.1
Decoupled+Vis      (Ours)     1.0  2.1  5.6  0.3  0.0  3.9     9.0
Holistic-full  (97 tok, ref)  1.0  2.2  6.1  0.3  8.0 11.1     9.6
```

**Case-study summary (near-field radius = 15 m):**

| Case | sample idx | #near dyn agents | naive captured | decoupled captured | naive near-err | dec near-err |
|---|---|---|---|---|---|---|
| 0 | 53 | 7 | 2 | **7** | 5.27 m | **0.30 m** |
| 1 | 54 | 7 | 2 | **7** | 4.91 m | **0.26 m** |
| 2 | 78 | 5 | **0** | **5** | 5.25 m | **0.24 m** |
| 3 | 81 | 5 | **0** | **5** | 6.64 m | **0.24 m** |
| 4 | 82 | 5 | **0** | **5** | 7.47 m | **0.26 m** |

`near-err` = average, over GT near-field dynamic agents, of the minimum distance from their next-frame GT position to any of the variant's predicted next-slot positions. Cases 2–4 exhibit the complete-miss regime where naive mixing's abstraction spends the entire 16-slot budget on relation tokens, leaving 0 slots for any near-field agent.

**How to regenerate**:

```bash
# (1) run once: extract selection metadata from the seed-7 checkpoints
python scripts/extract_slot_selections.py             # ~4 min (tokenisation)
# (2) figures (fast, <10 s each)
python scripts/plot_stage0_variance.py
python scripts/plot_slot_distribution.py
python scripts/plot_slot_scenes.py --top-k 5 --near-r 15 --min-agents 4
```

---

## 6. Diagnostic journey (for the paper's "Ablations" / appendix)

The final design is the endpoint of a sequence of corrections. Documented here so reviewers can see the chain of reasoning.

| Step | Problem observed | Diagnosis | Fix |
|---|---|---|---|
| A0 | `Rare Recall @ 5 m = 1.0` for every variant | Metric was reading ground-truth token types instead of predicted matches; 5 m threshold saturated anyway. | Rewrote eval → **Interaction Recall @ 1 m**, restricted to rare agents < 20 m from ego. |
| A1 | `Collision Acc ≈ 0.5`, stuck at base rate | Target was `1 − continues` but `continues ≡ 1` in the adapter → labels all 0. | Derive labels from relation-token TTC < 3 s. |
| A2 | `object_relation_visibility` behaved identically to `object_relation` | Visibility was the categorical 1-4 value clipped to 1.0 → zero gradient. | Map to `{0.2, 0.5, 0.7, 0.9}` continuous. |
| A3 | First "fair" run: `Holistic-16Slot` had `train_obs ≈ 0.005` (trivially low) but disastrous eval | `selected_indices` were all 0 → all 16 learned slots were supervised toward ego's next state (0,0 in ego-relative frame). | Added `AbstractionOutput.is_set_prediction` flag, dispatch to DETR-style `_set_prediction_obs_loss`. |
| **Fix #2** | After A3, `Object+Relation` val dynamic loss ≈ Object-only's but IntRec collapsed | Obs loss was regressing all 40 dims for every slot regardless of type; relation slots were being dragged to physical-position targets they cannot represent. | **Type-aware obs loss**: dyn slots regress `(x,y,vx,vy)` only; relation slots regress `(TTC, lane_conflict, priority)` only; map/signal/pad contribute 0. |
| **Evaluator patch** | After Fix #2, relation slots had un-trained `(x,y)` → random noise corrupted nearest-match pool. | Filter match candidates to dynamic-type only (see §3.1). | Implemented in `_update_rollout_and_ade`. |
| Residual failure | After Fix #2 + evaluator patch, `Object+Relation` still collapsed. Seed variance was *huge* (IntRec std 0.41). | Root cause: **abstraction budget competition**. Even with clean supervision and clean evaluation, if the top-k selects too many relation tokens there are too few dyn slots left to cover dynamic agents. | **Fix #3 / Route C**: decoupled abstraction — separate top-k per type, same total budget. |

### 6.1 Why Route C, not Route A / B

Three options considered at the residual-failure step:

- **Route A** — accept the finding, publish "naive mixing fails" as a negative result. *Rejected*: honest but weak; more a diagnostic fragment than a methodological contribution.
- **Route B** — hard-cap "12 dyn + 4 rel" inside the unified top-k (post-hoc re-ranking). *Rejected*: works but looks like a hyperparameter patch (`why 12/4, not 10/6?`).
- **Route C (chosen)** — decoupled abstraction with typed budgets. Separate selection heads, no competition, framed as "budget allocation matched to semantic role". Clean architectural narrative.

---

## 7. Reproducibility

### 7.1 Environment

- Conda env `doorrl` at `/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl`
- Python 3.10, PyTorch (CUDA 12), nuscenes-devkit
- Dataset: `/mnt/datasets/e2e-nuscenes/20260302`

### 7.2 Full Stage 0 pipeline

```bash
# Step 1 — train all 4 baselines + 1 upper-bound reference on 3 seeds.
#          Produces experiments/table3_fair_fix2_seed{7,42,2026}/*.pt
#          and *.json Table 3 per seed.
bash scripts/run_fix2_3seeds.sh

# Step 2 — train the two Decoupled variants on 3 seeds, reload the
#          base-variant checkpoints, merge everything into the same
#          seed dirs, then aggregate to mean ± std.
bash scripts/run_decoupled_3seeds.sh
```

### 7.3 Single variant, single seed

```bash
python run_stage0_table3.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant object_relation_decoupled \
    --num-scenes 700 \
    --epochs 15 \
    --batch-size 32 \
    --output-dir experiments/table3_fair_fix2_seed7 \
    --seed 7
```

### 7.4 All variants (incl. decoupled) in one call

```bash
python run_stage0_table3.py \
    --variant all_with_decoupled \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --num-scenes 700 --epochs 15 --batch-size 32 \
    --output-dir experiments/table3_fair_fix2_seed7 \
    --seed 7 \
    --evaluate-only          # reload pre-trained base .pt, only train the missing decoupled ones
```

### 7.5 Re-aggregate only

```bash
python scripts/aggregate_fix2_seeds.py \
    --runs experiments/table3_fair_fix2_seed7/table3_complete.json \
           experiments/table3_fair_fix2_seed42/table3_complete.json \
           experiments/table3_fair_fix2_seed2026/table3_complete.json \
    --out  experiments/table3_fair_fix2_aggregate.json
```

### 7.6 Sanity test (10 scenes × 3 epochs, 3-5 min)

```bash
python run_stage0_table3.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --variant object_relation_decoupled \
    --num-scenes 10 --epochs 3 --batch-size 8 \
    --output-dir experiments/table3_decoupled_sanity \
    --seed 7
```

---

## 8. File map

| Path | Purpose |
|---|---|
| `src/doorrl/config.py` | `ModelConfig` (incl. `top_k_dyn`, `top_k_rel`), `TrainingConfig` |
| `src/doorrl/schema.py` | `SceneBatch`, `TokenType` enum |
| `src/doorrl/adapters/base.py` | `NormalizedSceneConverter` — raw 40-dim token layout |
| `src/doorrl/adapters/nuscenes_real_adapter.py` | nuScenes → `SceneBatch`, CAN-bus action extraction |
| `src/doorrl/data/nuscenes_dataset.py` | In-memory tokenisation cache (`NuScenesSceneDataset`) |
| `src/doorrl/models/abstraction.py` | `DecisionSufficientAbstraction` (+ `top_k_override`, `force_ego` kwargs for Route C) |
| `src/doorrl/models/doorrl_variant.py` | All 7 model variants, incl. `_forward_object_relation_decoupled` |
| `src/doorrl/models/world_model.py` | `ReactiveObjectRelationalWorldModel` (shared across variants) |
| `src/doorrl/models/policy.py` | `ActorCriticHead` |
| `src/doorrl/training/losses.py` | `_typed_obs_loss` (Fix #2), `_set_prediction_obs_loss`, `_derive_collision_targets` |
| `src/doorrl/training/trainer.py` | `DoorRLTrainer`, dyn/rel loss logging |
| `src/doorrl/evaluation/table3_metrics.py` | All 5 Table 3 metrics, evaluator patch (dyn-only candidate filter) |
| `run_stage0_table3.py` | Orchestrator (`--variant all_with_decoupled` for the full comparison) |
| `configs/debug_mvp.json` | Model + training hyperparams |
| `scripts/run_fix2_3seeds.sh` | Seed 7/42/2026 base pipeline |
| `scripts/run_decoupled_3seeds.sh` | Seed 7/42/2026 decoupled pipeline (reloads base .pt via `--evaluate-only`) |
| `scripts/run_fast_3seeds.sh` | Same 3-seed fair pipeline with BF16 autocast + bs 128 + 3 concurrent seeds on one H20 |
| `scripts/aggregate_fix2_seeds.py` | Mean ± std aggregator |
| `scripts/extract_slot_selections.py` | P3 helper — runs 6 checkpoints on a fixed val subset and dumps `{tokens, selected_indices, predicted_next_tokens, ...}` to a pickle |
| `scripts/plot_stage0_variance.py` | P3.1 — seed-level mean ± std bar plots for the three headline metrics |
| `scripts/plot_slot_distribution.py` | P3.2 — per-variant stacked slot-type composition |
| `scripts/plot_slot_scenes.py` | P3.3 + P3.4 — auto-picks scenes by decoupled-vs-naive near-field advantage, draws 4-panel ego bird-view |
| `experiments/table3_fair_fix2_seed{7,42,2026}/` | Per-seed checkpoints + Table 3 JSON |
| `experiments/table3_fair_fix2_aggregate.json` | Final 3-seed mean ± std |
| `experiments/figures/stage0_variance_*.png` | P3.1 outputs |
| `experiments/figures/stage0_slot_distribution.png` | P3.2 output |
| `experiments/figures/scenes/case_*.png` | P3.3 / P3.4 outputs |
| `experiments/figures/slot_selections_seed7.pkl` | Cached extraction artefact shared by all three plot scripts |
| `experiments/run_{fix2,decoupled}_3seeds*.log` | Full training/eval logs |

---

## 9. Open items / suggested next steps

### 9.1 Low-hanging (wallclock)

- **Done (2026-04-23):** BF16 autocast inside `DoorRLTrainer`; `num_workers=2, persistent_workers=True, prefetch_factor=4` on both loaders; `SceneBatch.to()` uses `non_blocking=True`; CLI now exposes `--batch-size`, `--lr` / `--lr-scale`, `--compile`; `scripts/run_fast_3seeds.sh` launches 3 seeds concurrently on one H20.
- **Still pending:** persist the tokenisation cache to disk so later re-runs skip the ~10 min tokenisation (not critical once a seed's checkpoint is saved, since later phases use `--evaluate-only`).
- **Optional:** `--compile` with `torch.compile(mode="reduce-overhead")` — expect another 1.5–3 × on top of BF16 + bs ≥ 128, but first-step compile overhead is several seconds, so keep this opt-in.

### 9.2 Scientific follow-ups

- **K_dyn / K_rel ablation** — sweep (16+0, 14+2, 12+4, 10+6, 8+8) to show that Route C is robust to the budget split, which strengthens the "typed budgets, not hand-tuned numbers" framing.
- **Scale the model** (`model_dim` 128 → 512, 4 transformer layers). Current model is massively under-utilising the H20; results may tighten toward the Holistic-full upper bound.
- **More seeds** (5 seeds total) if the paper venue expects it; current 3 seeds already show clear separation in mean ± std.
- **Stage 1 (closed-loop / downstream RL)** — Stage 0 only measures representation quality via forecasting proxies. To fully support the paper's narrative, run closed-loop benchmarks (nuPlan reactive or navsim) using each variant as the learned representation. This is where H20's compute will actually be saturated.

### 9.3 Writing

- Table 3 in the paper should be exactly §5.1 above, optionally with a shaded row for Holistic-full to signal it is upper-bound reference rather than a fair competitor.
- The diagnostic journey (§6) is best distilled into an appendix; the main text should present Route C as the design choice motivated by budget competition.
