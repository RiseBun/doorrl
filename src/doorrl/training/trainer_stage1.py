"""Stage 1 ImaginationTrainer.

Subclasses the Stage 0 trainer so BF16, gradient clipping, optimizer
setup, and device management are reused verbatim. The only difference
is the loss computation: depending on the condition, we either
  * do BC (no imagination),
  * imagine K=1 step and do 1-step AC,
  * imagine K steps and do GAE AC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import torch

from doorrl.config import TrainingConfig
from doorrl.imagination.imagination import imagine_trajectory
from doorrl.imagination.task_reward import DEFAULT_REWARD_CFG, TaskRewardCfg
from doorrl.schema import SceneBatch
from doorrl.training.losses_stage1 import (
    Stage1LossCfg,
    ac1_loss,
    bc_loss,
    stage1_losses,
)
from doorrl.training.trainer import DoorRLTrainer


Condition = Literal["bc", "ac1", "wm"]


@dataclass
class Stage1Cfg:
    condition: Condition = "wm"
    horizon: int = 5
    # Default True (Dreamer-style): actor/critic gradients do NOT flow
    # into the world model; the WM is trained only by the Stage 0
    # sanity loss. In the pilot run we observed that allowing WM to be
    # driven by a mis-calibrated critic target (values on the order of
    # 1e4-1e6) corrupted the obs prediction in 2-3 epochs and produced
    # NaN. Keeping the WM clean via sanity loss also lets the policy
    # plan against a stable dynamics model. Users who explicitly want
    # end-to-end WM shaping from the AC loss can pass
    # ``detach_world_model=False`` to restore the original behaviour.
    detach_world_model: bool = True
    loss_cfg: Stage1LossCfg = Stage1LossCfg()
    reward_cfg: TaskRewardCfg = DEFAULT_REWARD_CFG
    # Hard cap on sampled action magnitude. Default 8.0 (the imagination
    # module's default — safe for tanh(3) × log_std<=0.5). Lower (e.g.
    # 5.0) to block an over-exploring actor from living at the clip,
    # which the pilot showed `wm_object` tends to do.
    action_sample_clip: float = 8.0


class ImaginationTrainer(DoorRLTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        device: torch.device,
        stage1_cfg: Stage1Cfg,
        use_amp: bool | None = None,
    ) -> None:
        super().__init__(model=model, config=config, device=device, use_amp=use_amp)
        self.stage1_cfg = stage1_cfg

    # Override fit() only to change the per-epoch log line.
    def fit(self, train_loader, val_loader=None) -> None:
        for epoch in range(self.config.epochs):
            train_stats = self.run_epoch(train_loader, training=True)
            message = self._format_log(epoch, "train", train_stats)
            if val_loader is not None:
                val_stats = self.run_epoch(val_loader, training=False)
                message += "   " + self._format_log(epoch, "val", val_stats)
            print(message)

    def _format_log(self, epoch: int, tag: str, s: Dict[str, float]) -> str:
        cond = self.stage1_cfg.condition
        if cond == "bc":
            return (
                f"epoch={epoch+1} {tag}_total={s.get('total', 0):.3f} "
                f"bc={s.get('bc_mse', 0):.3f} "
                f"sanity={s.get('sanity', 0):.3f}"
            )
        return (
            f"epoch={epoch+1} {tag}_total={s.get('total', 0):.3f} "
            f"actor={s.get('actor', 0):.3f} "
            f"critic={s.get('critic', 0):.3f} "
            f"sanity={s.get('sanity', 0):.3f} "
            f"R={s.get('reward_mean', 0):.3f} "
            f"V={s.get('value_mean', 0):.3f} "
            f"Cmax={s.get('collision_max', 0):.3f} "
            f"|a|max={s.get('action_abs_max', 0):.2f} "
            f"log_std={s.get('log_std_mean', 0):+.2f}"
        )

    def run_epoch(self, loader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        aggregate: Dict[str, float] = {}
        steps = 0
        cond = self.stage1_cfg.condition
        horizon = self.stage1_cfg.horizon if cond == "wm" else 1

        for batch in loader:
            if not isinstance(batch, SceneBatch):
                raise TypeError("loader must yield SceneBatch instances")
            batch = batch.to(self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self._autocast_dtype,
                enabled=self.use_amp,
            ):
                if cond == "bc":
                    output = self.model(batch)
                    loss, stats = bc_loss(output, batch, self.config)
                elif cond == "ac1":
                    traj = imagine_trajectory(
                        self.model, batch,
                        horizon=1, deterministic=False,
                        reward_cfg=self.stage1_cfg.reward_cfg,
                        detach_world_model=True,
                        action_sample_clip=self.stage1_cfg.action_sample_clip,
                    )
                    loss, stats = ac1_loss(traj, batch, self.config,
                                           self.stage1_cfg.loss_cfg)
                else:  # "wm"
                    traj = imagine_trajectory(
                        self.model, batch,
                        horizon=horizon, deterministic=False,
                        reward_cfg=self.stage1_cfg.reward_cfg,
                        detach_world_model=self.stage1_cfg.detach_world_model,
                        action_sample_clip=self.stage1_cfg.action_sample_clip,
                    )
                    loss, stats = stage1_losses(traj, batch, self.config,
                                                self.stage1_cfg.loss_cfg)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            for key, value in stats.items():
                aggregate[key] = aggregate.get(key, 0.0) + value
            steps += 1

        if steps == 0:
            return aggregate
        return {k: v / steps for k, v in aggregate.items()}
