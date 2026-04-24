"""DOOR-RL评估指标系统"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class EvaluationMetrics:
    """
    评估指标容器
    
    包含自动驾驶决策学习的关键指标:
    1. 世界模型预测精度
    2. 碰撞预测准确率
    3. 策略性能指标
    """
    
    # 世界模型指标
    observation_mse: List[float] = field(default_factory=list)
    reward_mse: List[float] = field(default_factory=list)
    continue_accuracy: List[float] = field(default_factory=list)
    collision_accuracy: List[float] = field(default_factory=list)
    
    # 策略指标
    action_mse: List[float] = field(default_factory=list)
    value_loss: List[float] = field(default_factory=list)
    
    # 高级指标
    ttc_accuracy: List[float] = field(default_factory=list)
    relation_prediction_accuracy: List[float] = field(default_factory=list)
    
    def update(self, batch_metrics: Dict[str, float]):
        """更新指标"""
        for key, value in batch_metrics.items():
            if hasattr(self, key) and isinstance(getattr(self, key), list):
                getattr(self, key).append(value)
    
    def compute_summary(self) -> Dict[str, float]:
        """计算所有指标的平均值"""
        summary = {}
        
        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, list) and len(attr_value) > 0:
                summary[attr_name] = np.mean(attr_value)
        
        return summary
    
    def print_report(self):
        """打印评估报告"""
        summary = self.compute_summary()
        
        print("\n" + "="*60)
        print("DOOR-RL Evaluation Report")
        print("="*60)
        
        print("\n1. World Model Prediction:")
        print(f"   Observation MSE:     {summary.get('observation_mse', 0.0):.4f}")
        print(f"   Reward MSE:          {summary.get('reward_mse', 0.0):.4f}")
        print(f"   Continue Accuracy:   {summary.get('continue_accuracy', 0.0):.4f}")
        print(f"   Collision Accuracy:  {summary.get('collision_accuracy', 0.0):.4f}")
        
        print("\n2. Policy Performance:")
        print(f"   Action MSE:          {summary.get('action_mse', 0.0):.4f}")
        print(f"   Value Loss:          {summary.get('value_loss', 0.0):.4f}")
        
        print("\n3. Relation Understanding:")
        print(f"   TTC Accuracy:        {summary.get('ttc_accuracy', 0.0):.4f}")
        print(f"   Relation Accuracy:   {summary.get('relation_prediction_accuracy', 0.0):.4f}")
        
        print("="*60)


class WorldModelEvaluator:
    """
    世界模型评估器
    
    评估世界模型的预测能力
    """
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.metrics = EvaluationMetrics()
    
    def evaluate_batch(self, batch, model_output) -> Dict[str, float]:
        """
        评估单个batch的预测
        
        Args:
            batch: SceneBatch
            model_output: DoorRLOutput
        
        Returns:
            指标字典
        """
        batch_metrics = {}
        
        # 1. Observation预测精度
        pred_next = model_output.world_model.predicted_next_tokens
        true_next = batch.next_tokens
        mask = batch.token_mask.unsqueeze(-1).float()
        
        obs_mse = ((pred_next - true_next) * mask).pow(2).sum() / mask.sum().clamp(min=1.0)
        batch_metrics['observation_mse'] = obs_mse.item()
        
        # 2. Reward预测精度
        pred_reward = model_output.world_model.predicted_reward
        true_reward = batch.rewards
        reward_mse = (pred_reward - true_reward).pow(2).mean()
        batch_metrics['reward_mse'] = reward_mse.item()
        
        # 3. Continue预测准确率
        pred_continue = model_output.world_model.predicted_continue
        true_continue = batch.continues
        pred_continue_binary = (pred_continue > 0.5).float()
        continue_acc = (pred_continue_binary == true_continue).float().mean()
        batch_metrics['continue_accuracy'] = continue_acc.item()
        
        # 4. Collision预测准确率
        pred_collision = model_output.world_model.predicted_collision
        # 假设collision ground truth可以从关系token中提取 (简化)
        # 实际需要更复杂的逻辑
        batch_metrics['collision_accuracy'] = 0.5  # Placeholder
        
        # 更新累积指标
        self.metrics.update(batch_metrics)
        
        return batch_metrics


class PolicyEvaluator:
    """
    策略评估器
    
    评估策略网络的输出质量
    """
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.metrics = EvaluationMetrics()
    
    def evaluate_batch(self, batch, model_output) -> Dict[str, float]:
        """评估策略输出"""
        batch_metrics = {}
        
        # 1. Action预测精度 (Behavior Cloning)
        pred_action = model_output.policy.action_mean
        true_action = batch.actions
        action_mse = (pred_action - true_action).pow(2).mean()
        batch_metrics['action_mse'] = action_mse.item()
        
        # 2. Value预测
        pred_value = model_output.policy.value
        # Value的ground truth需要从rollout中计算 (简化处理)
        batch_metrics['value_loss'] = 0.0  # Placeholder
        
        # 更新累积指标
        self.metrics.update(batch_metrics)
        
        return batch_metrics


def evaluate_model(
    model,
    data_loader,
    device: torch.device,
    verbose: bool = True
) -> EvaluationMetrics:
    """
    完整评估模型
    
    Args:
        model: DoorRLModel
        data_loader: DataLoader
        device: 计算设备
        verbose: 是否打印报告
    
    Returns:
        EvaluationMetrics对象
    """
    model.eval()
    
    wm_evaluator = WorldModelEvaluator(device)
    policy_evaluator = PolicyEvaluator(device)
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch)
            
            # 评估世界模型
            wm_evaluator.evaluate_batch(batch, output)
            
            # 评估策略
            policy_evaluator.evaluate_batch(batch, output)
    
    # 合并指标
    combined_metrics = EvaluationMetrics()
    combined_metrics.observation_mse = wm_evaluator.metrics.observation_mse
    combined_metrics.reward_mse = wm_evaluator.metrics.reward_mse
    combined_metrics.continue_accuracy = wm_evaluator.metrics.continue_accuracy
    combined_metrics.collision_accuracy = wm_evaluator.metrics.collision_accuracy
    combined_metrics.action_mse = policy_evaluator.metrics.action_mse
    combined_metrics.value_loss = policy_evaluator.metrics.value_loss
    
    if verbose:
        combined_metrics.print_report()
    
    return combined_metrics
