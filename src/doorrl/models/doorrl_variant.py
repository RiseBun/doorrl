"""模型变体定义 - 支持消融实验"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import torch
from torch import nn

from doorrl.config import ModelConfig
from doorrl.models.abstraction import (
    AbstractionOutput,
    DecisionSufficientAbstraction,
)
from doorrl.models.doorrl import DoorRLModel, DoorRLOutput
from doorrl.models.encoder import TokenEncoder
from doorrl.models.policy import ActorCriticHead
from doorrl.models.world_model import ReactiveObjectRelationalWorldModel
from doorrl.schema import SceneBatch, TokenType


class ModelVariant(str, Enum):
    """
    模型变体 - 用于消融实验

    变体说明:
      1. HOLISTIC           : 整体表示, 把全部 97 个 token 直接喂给 world model
                              (upper-bound reference, 信息量 97 tokens).
      2. HOLISTIC_16SLOT    : 公平版整体表示. 用 ``top_k`` 个 learned queries 去
                              cross-attend 全部 token, 压缩成 top_k 个 slot
                              (默认 16), 与其他 16-slot variant 严格等 context.
      3. OBJECT_ONLY        : 仅对象 token, 忽略关系 token, 在其上做 top-k 抽象.
      4. OBJECT_RELATION    : 对象 + 关系 token + top-k 抽象 (标准 DOOR-RL).
                              在共享 16-slot 预算里同时容纳两类 token, 因此
                              relation 与 dynamic agent 互相竞争名额 ("naive
                              mixing"). Stage-0 表里出现的 dyn 槽位被 relation
                              挤掉的失败模式即来自这里.
      5. OBJECT_RELATION_VISIBILITY
                            : 在 OBJECT_RELATION 基础上对 latent 做可见性加权.
      6. OBJECT_RELATION_DECOUPLED
                            : *Route C* (decoupled / typed-budget abstraction).
                              两路独立 top-k:
                                - dyn 路 K_dyn (默认 12) over EGO/VEHICLE/PED/CYCLIST
                                - rel 路 K_rel (默认  4) over RELATION
                              拼接成 K_dyn+K_rel 个 slot 喂给 world model.
                              relation 不再与 dynamic agent 抢预算, 同时下游
                              context budget 仍与其他 16-slot variant 一致.
      7. OBJECT_RELATION_DECOUPLED_VISIBILITY
                            : DECOUPLED + 可见性加权 dynamic 路 latent.
    """
    HOLISTIC = "holistic"
    HOLISTIC_16SLOT = "holistic_16slot"
    OBJECT_ONLY = "object_only"
    OBJECT_RELATION = "object_relation"
    OBJECT_RELATION_VISIBILITY = "object_relation_visibility"
    OBJECT_RELATION_DECOUPLED = "object_relation_decoupled"
    OBJECT_RELATION_DECOUPLED_VISIBILITY = "object_relation_decoupled_visibility"


class DoorRLModelVariant(DoorRLModel):
    """
    DOOR-RL模型变体
    
    支持通过配置切换不同的表示方式
    """
    
    def __init__(
        self, 
        config: ModelConfig,
        variant: ModelVariant = ModelVariant.OBJECT_RELATION
    ) -> None:
        super().__init__(config)
        self.variant = variant
        
        # 根据变体调整模型行为
        if variant == ModelVariant.HOLISTIC:
            # Holistic: 使用所有token的全局池化，不进行top-k选择
            self._setup_holistic_mode()
        elif variant == ModelVariant.HOLISTIC_16SLOT:
            # Fair holistic: 16 learned queries cross-attend 全部 token -> 16 slot
            self._setup_holistic_16slot_mode(config)
        elif variant == ModelVariant.OBJECT_ONLY:
            # Object-only: 只使用对象token，忽略关系
            self._setup_object_only_mode()
        elif variant == ModelVariant.OBJECT_RELATION:
            # Object-relation: 使用对象+关系token (标准DOOR-RL)
            self._setup_object_relation_mode()
        elif variant == ModelVariant.OBJECT_RELATION_VISIBILITY:
            # Object-relation-visibility: 添加可见性先验
            self._setup_visibility_mode()
        elif variant == ModelVariant.OBJECT_RELATION_DECOUPLED:
            # Decoupled (Route C): typed budgets, no slot competition.
            self._setup_decoupled_mode(config, with_visibility=False)
        elif variant == ModelVariant.OBJECT_RELATION_DECOUPLED_VISIBILITY:
            self._setup_decoupled_mode(config, with_visibility=True)
    
    def _setup_holistic_mode(self):
        """Holistic表示：不使用abstraction，直接池化所有token"""
        # 对于holistic，我们需要修改forward逻辑
        # 这里设置标志位，在forward中处理
        self.use_holistic = True

    def _setup_holistic_16slot_mode(self, config: ModelConfig):
        """
        公平版 Holistic-16Slot: 用 ``top_k`` 个 learned queries 通过一次
        cross-attention, 把全部 97 个 token 压缩到 top_k 个 holistic slot.
        这样 holistic variant 和 16-slot object/relation variant 的 world
        model 输入维度一致, Table 3 才是真正的"表示质量比较"而不是"信息量比较".
        """
        self.use_holistic = False
        self.holistic_num_slots = config.top_k
        # 一组可学习 queries, 以小方差初始化以便稳定训练.
        self.holistic_queries = nn.Parameter(
            torch.randn(config.top_k, config.model_dim) * 0.02
        )
        self.holistic_cross_attn = nn.MultiheadAttention(
            embed_dim=config.model_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.holistic_slot_norm = nn.LayerNorm(config.model_dim)

    def _setup_object_only_mode(self):
        """Object-only表示：过滤掉关系token"""
        self.use_holistic = False
        self.filter_relations = True
    
    def _setup_object_relation_mode(self):
        """Object-relation表示：标准DOOR-RL"""
        self.use_holistic = False
        self.filter_relations = False
    
    def _setup_visibility_mode(self):
        """Object-relation-visibility表示：添加可见性加权"""
        self.use_holistic = False
        self.filter_relations = False
        self.use_visibility_weighting = True

    def _setup_decoupled_mode(self, config: ModelConfig, with_visibility: bool):
        """Decoupled abstraction (Route C).

        Two parallel ``DecisionSufficientAbstraction`` modules with
        per-type budgets (``top_k_dyn`` + ``top_k_rel`` == ``top_k``).
        Forward concatenates the dynamic and relation slot sets so the
        downstream world model receives the same total context size as
        every other 16-slot variant in fair Stage 0.
        """
        self.use_holistic = False
        self.use_decoupled = True
        self.use_decoupled_visibility = with_visibility
        self.top_k_dyn = config.top_k_dyn
        self.top_k_rel = config.top_k_rel
        # Sanity: keep the combined budget identical to the shared one,
        # otherwise the variant is no longer comparable to other 16-slot
        # variants in the table.
        if self.top_k_dyn + self.top_k_rel != config.top_k:
            raise ValueError(
                f"Decoupled budgets must sum to top_k={config.top_k}; "
                f"got top_k_dyn={self.top_k_dyn}, "
                f"top_k_rel={self.top_k_rel}."
            )
        # Two independent abstraction heads. ``force_ego=False`` on the
        # relation head because RELATION never includes ego; otherwise
        # the head would waste a slot trying to force-select ego.
        self.abstraction_dyn = DecisionSufficientAbstraction(
            config, top_k_override=self.top_k_dyn, force_ego=True,
        )
        self.abstraction_rel = DecisionSufficientAbstraction(
            config, top_k_override=self.top_k_rel, force_ego=False,
        )
    
    def forward(self, batch: SceneBatch) -> DoorRLOutput:
        """
        前向传播 - 根据变体调整
        
        不同变体的处理逻辑:
        1. Holistic: 直接池化所有token，不使用abstraction
        2. Object-only: 过滤掉关系token
        3. Object-relation: 标准流程
        4. Object-relation-visibility: 添加可见性加权
        """
        if self.variant == ModelVariant.HOLISTIC:
            return self._forward_holistic(batch)
        elif self.variant == ModelVariant.HOLISTIC_16SLOT:
            return self._forward_holistic_16slot(batch)
        elif self.variant == ModelVariant.OBJECT_ONLY:
            return self._forward_object_only(batch)
        elif self.variant == ModelVariant.OBJECT_RELATION_VISIBILITY:
            return self._forward_with_visibility(batch)
        elif self.variant in (
            ModelVariant.OBJECT_RELATION_DECOUPLED,
            ModelVariant.OBJECT_RELATION_DECOUPLED_VISIBILITY,
        ):
            return self._forward_object_relation_decoupled(batch)
        else:
            # 标准object-relation
            return self._forward_standard(batch)
    
    def _forward_standard(self, batch: SceneBatch) -> DoorRLOutput:
        """标准DOOR-RL前向传播"""
        # 1. 编码所有token
        latent = self.encoder(batch.tokens, batch.token_types)
        
        # 2. 决策充分抽象 (top-k选择)
        abstraction = self.abstraction(latent, batch.token_mask)
        
        # 3. 世界模型预测
        world_model = self.world_model(
            selected_tokens=abstraction.selected_tokens,
            selected_mask=abstraction.selected_mask,
            actions=batch.actions,
        )
        
        # 4. 策略输出
        policy = self.policy(abstraction.global_latent)
        
        return DoorRLOutput(
            abstraction=abstraction,
            world_model=world_model,
            policy=policy,
        )
    
    def _forward_holistic(self, batch: SceneBatch) -> DoorRLOutput:
        """
        Holistic表示：全局池化，不区分对象
        
        这种方式将所有token (ego, objects, map, relations) 同等对待，
        直接进行全局平均池化得到global_latent
        """
        # 1. 编码所有token
        latent = self.encoder(batch.tokens, batch.token_types)
        
        # 2. 全局平均池化 (代替abstraction)
        mask = batch.token_mask.unsqueeze(-1).float()  # [B, S, 1]
        global_latent = (latent * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        
        # 3. 为了兼容，创建假的abstraction输出
        from doorrl.models.abstraction import AbstractionOutput
        abstraction = AbstractionOutput(
            selected_tokens=latent,  # 使用所有token
            selected_mask=batch.token_mask,
            selected_indices=torch.arange(latent.size(1), device=latent.device).unsqueeze(0).expand(batch.tokens.size(0), -1),
            importance=torch.ones(latent.size(0), latent.size(1), device=latent.device) / latent.size(1),
            global_latent=global_latent,
        )
        
        # 4. 世界模型预测
        world_model = self.world_model(
            selected_tokens=latent,
            selected_mask=batch.token_mask,
            actions=batch.actions,
        )
        
        # 5. 策略输出
        policy = self.policy(global_latent)
        
        return DoorRLOutput(
            abstraction=abstraction,
            world_model=world_model,
            policy=policy,
        )
    
    def _forward_holistic_16slot(self, batch: SceneBatch) -> DoorRLOutput:
        """
        公平 Holistic: 16 learned queries cross-attend 全部 97 token
        -> 16 compressed slot. World model 拿到的 context 数量与其他
        16-slot variant 严格一致, 差别只在 "每个 slot 里编码的是什么".

        Notes
        -----
        * ``selected_indices`` / ``selected_mask`` 仍需填写以兼容下游
          AbstractionOutput 接口, 但此 variant 下它们不对应原始 token
          位置. 所以评估代码 (``table3_metrics.py``) 必须用
          nearest-assignment 做匹配, 不能依赖 selected_indices 直接索引.
        * 用 token_mask 作为 key-padding-mask, 保证 padding token 不会
          被 query 关注到.
        """
        from doorrl.models.abstraction import AbstractionOutput

        latent = self.encoder(batch.tokens, batch.token_types)  # [B, S, D]
        batch_size, seq_len, dim = latent.shape
        num_slots = self.holistic_num_slots

        queries = self.holistic_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]

        key_padding_mask = ~batch.token_mask  # [B, S] True = pad
        # Defensive: if a row is entirely padded MultiheadAttention returns
        # NaN for that row. In practice EGO (idx 0) is always valid so this
        # should not trigger, but guard anyway.
        all_pad_rows = key_padding_mask.all(dim=1)
        if all_pad_rows.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad_rows, 0] = False

        slots, _ = self.holistic_cross_attn(
            queries, latent, latent, key_padding_mask=key_padding_mask
        )  # [B, K, D]
        slots = self.holistic_slot_norm(slots + queries)

        slot_mask = torch.ones(
            batch_size, num_slots, dtype=torch.bool, device=latent.device
        )
        # Placeholder indices - not usable for per-slot->orig-token lookup.
        sel_indices = torch.zeros(
            batch_size, num_slots, dtype=torch.long, device=latent.device
        )
        global_latent = slots.mean(dim=1)  # [B, D]
        importance = (
            torch.ones(batch_size, num_slots, device=latent.device) / float(num_slots)
        )

        abstraction = AbstractionOutput(
            selected_tokens=slots,
            selected_mask=slot_mask,
            selected_indices=sel_indices,
            importance=importance,
            global_latent=global_latent,
            is_set_prediction=True,
        )

        world_model = self.world_model(
            selected_tokens=slots,
            selected_mask=slot_mask,
            actions=batch.actions,
        )
        policy = self.policy(global_latent)

        return DoorRLOutput(
            abstraction=abstraction,
            world_model=world_model,
            policy=policy,
        )

    def _forward_object_only(self, batch: SceneBatch) -> DoorRLOutput:
        """
        Object-only表示：只使用对象token，忽略关系
        
        这种方式过滤掉所有RELATION类型的token，
        只保留EGO, VEHICLE, PEDESTRIAN, CYCLIST
        """
        from doorrl.schema import TokenType
        from doorrl.models.abstraction import AbstractionOutput
        
        # 1. 编码所有token
        latent = self.encoder(batch.tokens, batch.token_types)
        
        # 2. 创建object-only mask (过滤掉RELATION)
        object_mask = batch.token_mask.clone()
        relation_mask = (batch.token_types == int(TokenType.RELATION))
        object_mask = object_mask & ~relation_mask
        
        # 3. 如果没有任何object token，回退到原始mask
        if object_mask.sum() == 0:
            object_mask = batch.token_mask
        
        # 4. 应用abstraction (只在object token上)
        abstraction = self.abstraction(latent, object_mask)
        
        # 5. 世界模型预测
        world_model = self.world_model(
            selected_tokens=abstraction.selected_tokens,
            selected_mask=abstraction.selected_mask,
            actions=batch.actions,
        )
        
        # 6. 策略输出
        policy = self.policy(abstraction.global_latent)
        
        return DoorRLOutput(
            abstraction=abstraction,
            world_model=world_model,
            policy=policy,
        )
    
    def _forward_with_visibility(self, batch: SceneBatch) -> DoorRLOutput:
        """
        Object-relation-visibility表示：添加可见性先验
        
        在abstraction之前，根据可见性对token进行加权
        """
        from doorrl.models.abstraction import AbstractionOutput
        
        # 1. 编码所有token
        latent = self.encoder(batch.tokens, batch.token_types)
        
        # 2. 提取可见性特征 (在token的第7维)
        visibility = batch.tokens[:, :, 7:8]  # [B, S, 1]
        visibility = visibility.clamp(0.0, 1.0)
        
        # 3. 应用可见性加权
        weighted_latent = latent * visibility
        
        # 4. 应用abstraction
        abstraction = self.abstraction(weighted_latent, batch.token_mask)
        
        # 5. 世界模型预测
        world_model = self.world_model(
            selected_tokens=abstraction.selected_tokens,
            selected_mask=abstraction.selected_mask,
            actions=batch.actions,
        )
        
        # 6. 策略输出
        policy = self.policy(abstraction.global_latent)
        
        return DoorRLOutput(
            abstraction=abstraction,
            world_model=world_model,
            policy=policy,
        )


    def _forward_object_relation_decoupled(self, batch: SceneBatch) -> DoorRLOutput:
        """Decoupled (Route C) abstraction.

        Steps:
          1. Encode all tokens.
          2. Build a *dynamic-only* mask (EGO/VEHICLE/PEDESTRIAN/CYCLIST).
             Optionally apply visibility weighting on the dynamic-path
             latent (token visibility lives at raw-feature index 7, see
             NormalizedSceneConverter._fill_token).
          3. Build a *relation-only* mask.
          4. Run two independent ``DecisionSufficientAbstraction`` heads
             with budgets ``K_dyn`` and ``K_rel`` (sum == top_k).
          5. Concatenate (selected_tokens, selected_mask, selected_indices)
             along the slot dim. The dyn slots come first so that the
             type-aware loss / evaluator can still rely on
             ``token_types[selected_indices]`` to distinguish them.
          6. Average the two global-latent vectors as the policy head's
             input.

        Notes
        -----
        * ``selected_indices`` returned here are *valid token indices*
          (point at real tokens in ``batch.tokens``). Hence the
          existing typed-obs-loss path (Fix #2) and the patched
          dynamic-only nearest-match evaluator both work without changes:
          dyn slots route to (x,y,vx,vy) supervision and into the match
          pool; rel slots route to (TTC, lane_conflict, priority)
          supervision and are excluded from the match pool.
        * No fallback if a scene has zero relation tokens: the rel head
          will still emit ``K_rel`` placeholder slots, but their
          ``selected_mask`` rows are False so they contribute neither to
          the world model (key-padding-mask) nor to any loss.
        """
        latent = self.encoder(batch.tokens, batch.token_types)  # [B, S, D]
        token_types = batch.token_types
        token_mask = batch.token_mask

        dyn_mask = torch.zeros_like(token_mask, dtype=torch.bool)
        for t in (
            int(TokenType.EGO),
            int(TokenType.VEHICLE),
            int(TokenType.PEDESTRIAN),
            int(TokenType.CYCLIST),
        ):
            dyn_mask |= (token_types == t)
        dyn_mask &= token_mask

        rel_mask = (token_types == int(TokenType.RELATION)) & token_mask

        # Dynamic-path latent (optionally visibility-weighted).
        if getattr(self, "use_decoupled_visibility", False):
            visibility = batch.tokens[:, :, 7:8].clamp(0.0, 1.0)
            dyn_latent = latent * visibility
        else:
            dyn_latent = latent

        dyn_abs = self.abstraction_dyn(dyn_latent, dyn_mask)
        rel_abs = self.abstraction_rel(latent, rel_mask)

        selected_tokens = torch.cat(
            [dyn_abs.selected_tokens, rel_abs.selected_tokens], dim=1
        )
        selected_mask = torch.cat(
            [dyn_abs.selected_mask, rel_abs.selected_mask], dim=1
        )
        selected_indices = torch.cat(
            [dyn_abs.selected_indices, rel_abs.selected_indices], dim=1
        )
        # Re-normalise importances jointly so they sum to 1 across the
        # combined slot set (purely cosmetic; downstream code does not
        # currently consume importance, but keep it well-formed).
        importance = torch.cat(
            [dyn_abs.importance, rel_abs.importance], dim=1
        )
        importance = importance / importance.sum(dim=1, keepdim=True).clamp_min(1e-6)
        global_latent = 0.5 * (dyn_abs.global_latent + rel_abs.global_latent)

        abstraction = AbstractionOutput(
            selected_tokens=selected_tokens,
            selected_mask=selected_mask,
            selected_indices=selected_indices,
            importance=importance,
            global_latent=global_latent,
            is_set_prediction=False,
        )

        world_model = self.world_model(
            selected_tokens=selected_tokens,
            selected_mask=selected_mask,
            actions=batch.actions,
        )
        policy = self.policy(global_latent)
        return DoorRLOutput(
            abstraction=abstraction,
            world_model=world_model,
            policy=policy,
        )


def create_model_variant(
    config: ModelConfig,
    variant: ModelVariant = ModelVariant.OBJECT_RELATION
) -> DoorRLModelVariant:
    """
    工厂函数：创建指定变体的模型
    
    Args:
        config: 模型配置
        variant: 模型变体
    
    Returns:
        对应的模型实例
    """
    model = DoorRLModelVariant(config, variant)
    print(f"Created model variant: {variant.value}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model
