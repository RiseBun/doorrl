from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from doorrl.config import TrainingConfig
from doorrl.schema import SceneBatch
from doorrl.training.losses import compute_losses


class DoorRLTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        use_amp: bool | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        # BF16 autocast on CUDA. H20 has native BF16, no GradScaler needed
        # (BF16 has FP32-equivalent dynamic range). Safe no-op on CPU.
        if use_amp is None:
            use_amp = (device.type == "cuda")
        self.use_amp = use_amp
        self._autocast_dtype = torch.bfloat16

    def fit(self, train_loader, val_loader=None) -> None:
        for epoch in range(self.config.epochs):
            train_stats = self.run_epoch(train_loader, training=True)
            message = (
                f"epoch={epoch + 1} "
                f"train_total={train_stats['total']:.4f} "
                f"train_obs={train_stats['obs']:.4f} "
                f"(dyn={train_stats.get('obs_dyn', 0.0):.4f}, "
                f"rel={train_stats.get('obs_rel', 0.0):.4f})"
            )
            if val_loader is not None:
                val_stats = self.run_epoch(val_loader, training=False)
                message += (
                    f" val_total={val_stats['total']:.4f}"
                    f" val_obs={val_stats['obs']:.4f}"
                    f" (dyn={val_stats.get('obs_dyn', 0.0):.4f}, "
                    f"rel={val_stats.get('obs_rel', 0.0):.4f})"
                )
            print(message)

    def run_epoch(self, loader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        aggregate: Dict[str, float] = {
            "total": 0.0,
            "obs": 0.0,
            "obs_dyn": 0.0,
            "obs_rel": 0.0,
            "reward": 0.0,
            "continue": 0.0,
            "collision": 0.0,
            "bc": 0.0,
        }
        steps = 0
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
                output = self.model(batch)
                loss, stats = compute_losses(batch, output, self.config)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            for key, value in stats.items():
                aggregate[key] = aggregate.get(key, 0.0) + value
            steps += 1

        if steps == 0:
            return aggregate
        return {key: value / steps for key, value in aggregate.items()}
