"""nuScenes动作和奖励提取器"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from nuscenes.nuscenes import NuScenes


class NuScenesActionExtractor:
    """
    从nuScenes数据中提取真实的动作和奖励信号
    
    动作提取策略:
    1. 优先使用CAN总线数据 (最准确)
    2. 备选：从连续帧的位姿差异计算
    3. 动作定义: [acceleration, steering_angle]
    
    奖励设计策略:
    1. 安全性奖励 (碰撞避免)
    2. 舒适性奖励 (加速度/转向平滑)
    3. 效率奖励 (速度跟踪)
    4. 规则遵守 (车道保持)
    """
    
    def __init__(self, nusc: NuScenes, use_can_bus: bool = True):
        self.nusc = nusc
        self.use_can_bus = use_can_bus
        
        if use_can_bus:
            try:
                from nuscenes.can_bus.can_bus_api import NuScenesCanBus
                self.can_bus = NuScenesCanBus(dataroot=nusc.dataroot)
            except Exception:
                self.can_bus = None
        else:
            self.can_bus = None
    
    def extract_action_from_can(
        self, 
        scene_name: str, 
        sample: Dict[str, Any]
    ) -> Optional[List[float]]:
        """
        从CAN总线提取真实动作
        
        Returns:
            [acceleration (m/s²), steering_angle (rad)] 或 None
        """
        if self.can_bus is None:
            return None
        
        try:
            pose_messages = self.can_bus.get_messages(scene_name, 'pose')

            if not pose_messages:
                return None

            sample_time = sample['timestamp']
            closest_pose = min(pose_messages,
                             key=lambda p: abs(p['utime'] - sample_time))

            accel = closest_pose.get('accel', [0.0, 0.0, 0.0])
            longitudinal_accel = float(accel[0]) if len(accel) > 0 else 0.0
            if not math.isfinite(longitudinal_accel):
                longitudinal_accel = 0.0

            # nuScenes CAN bus: the only steering-related message is
            # `steeranglefeedback` (units: degrees, steering-wheel angle).
            # Earlier versions of this file referenced a non-existent
            # `steer_report` message and threw on every sample, nuking the
            # entire CAN-based action. Keep the steer extraction in an
            # inner try so that a missing/invalid steer feed does not throw
            # away the accel we already computed.
            steering_angle = 0.0
            try:
                steer_messages = self.can_bus.get_messages(
                    scene_name, 'steeranglefeedback'
                )
                if steer_messages:
                    closest_steer = min(
                        steer_messages,
                        key=lambda s: abs(s['utime'] - sample_time),
                    )
                    raw_deg = float(closest_steer.get('value', 0.0))
                    rad = raw_deg * math.pi / 180.0
                    steering_angle = max(-math.pi, min(math.pi, rad))
                    if not math.isfinite(steering_angle):
                        steering_angle = 0.0
            except Exception:
                steering_angle = 0.0

            return [longitudinal_accel, steering_angle]

        except Exception as e:
            print(f"Warning: Failed to extract CAN action: {e}")
            return None
    
    def extract_action_from_pose(
        self,
        current_sample: Dict[str, Any],
        next_sample: Optional[Dict[str, Any]],
        dt: float = 0.5
    ) -> List[float]:
        """
        从位姿差异计算动作 (备选方案)
        
        Args:
            current_sample: 当前帧
            next_sample: 下一帧 (用于计算速度变化)
            dt: 时间间隔 (默认0.5s)
        
        Returns:
            [acceleration, steering_angle]
        """
        from pyquaternion import Quaternion
        
        # 获取当前位姿
        current_sd = self.nusc.get('sample_data', 
                                   current_sample['data']['LIDAR_TOP'])
        current_pose = self.nusc.get('ego_pose', current_sd['ego_pose_token'])
        current_trans = np.array(current_pose['translation'])
        current_rot = Quaternion(current_pose['rotation'])
        
        if next_sample is None:
            # 没有下一帧，返回零动作
            return [0.0, 0.0]
        
        # 获取下一帧位姿
        next_sd = self.nusc.get('sample_data', 
                               next_sample['data']['LIDAR_TOP'])
        next_pose = self.nusc.get('ego_pose', next_sd['ego_pose_token'])
        next_trans = np.array(next_pose['translation'])
        next_rot = Quaternion(next_pose['rotation'])
        
        # 计算位移和速度变化
        displacement = next_trans - current_trans
        current_vel = current_rot.rotate([1, 0, 0])  # 简化假设
        
        # 计算加速度 (纵向)
        acceleration = displacement[0] / (dt ** 2)
        
        # 计算转向角 (从航向变化)
        heading_change = next_rot.yaw_pitch_roll[0] - current_rot.yaw_pitch_roll[0]
        steering_angle = heading_change / dt
        
        # 归一化到合理范围
        acceleration = np.clip(acceleration, -4.0, 4.0)  # ±4 m/s²
        steering_angle = np.clip(steering_angle, -0.5, 0.5)  # ±0.5 rad
        
        return [float(acceleration), float(steering_angle)]
    
    def compute_reward(
        self,
        ego_state: Dict[str, float],
        objects: List[Dict[str, Any]],
        action: List[float],
        prev_action: Optional[List[float]] = None,
        target_speed: float = 10.0,  # m/s, 约36 km/h
    ) -> float:
        """
        计算复合奖励信号
        
        奖励组成:
        - 安全性 (40%): 碰撞风险惩罚
        - 舒适性 (30%): 动作平滑性
        - 效率 (20%): 速度跟踪
        - 规则 (10%): 车道保持
        """
        # 1. 安全性奖励 (基于碰撞风险)
        safety_reward = self._compute_safety_reward(objects)
        
        # 2. 舒适性奖励 (动作变化惩罚)
        comfort_reward = self._compute_comfort_reward(action, prev_action)
        
        # 3. 效率奖励 (速度跟踪)
        efficiency_reward = self._compute_efficiency_reward(ego_state, target_speed)
        
        # 4. 规则奖励 (简化为横向偏移惩罚)
        rule_reward = self._compute_rule_reward(ego_state)
        
        # 加权组合
        total_reward = (
            0.4 * safety_reward +
            0.3 * comfort_reward +
            0.2 * efficiency_reward +
            0.1 * rule_reward
        )
        
        return total_reward
    
    def _compute_safety_reward(
        self, 
        objects: List[Dict[str, Any]]
    ) -> float:
        """安全性奖励：基于最小TTC和距离"""
        if not objects:
            return 1.0  # 没有障碍物，完全安全
        
        # 找到最危险的对象
        min_ttc = float('inf')
        min_distance = float('inf')
        
        for obj in objects:
            ttc = obj.get('ttc', 999.0)
            distance = obj.get('distance', 999.0)
            
            if ttc < min_ttc:
                min_ttc = ttc
            if distance < min_distance:
                min_distance = distance
        
        # TTC惩罚
        if min_ttc < 2.0:  # 2秒内碰撞，严重惩罚
            ttc_penalty = -10.0
        elif min_ttc < 5.0:  # 5秒内碰撞，中等惩罚
            ttc_penalty = -2.0
        else:
            ttc_penalty = 0.0
        
        # 距离惩罚
        if min_distance < 5.0:  # 5米内，严重惩罚
            dist_penalty = -5.0
        elif min_distance < 10.0:  # 10米内，轻微惩罚
            dist_penalty = -1.0
        else:
            dist_penalty = 0.0
        
        return max(ttc_penalty + dist_penalty, -10.0)
    
    def _compute_comfort_reward(
        self,
        action: List[float],
        prev_action: Optional[List[float]] = None
    ) -> float:
        """舒适性奖励：惩罚剧烈的动作变化"""
        if prev_action is None:
            return 0.0
        
        # 计算动作变化 (jerk)
        accel_change = action[0] - prev_action[0]
        steer_change = action[1] - prev_action[1]
        
        # 加速度变化惩罚
        if abs(accel_change) > 2.0:  # 加速度突变 > 2 m/s²
            accel_penalty = -2.0
        elif abs(accel_change) > 1.0:
            accel_penalty = -0.5
        else:
            accel_penalty = 0.0
        
        # 转向变化惩罚
        if abs(steer_change) > 0.3:  # 转向突变 > 0.3 rad
            steer_penalty = -2.0
        elif abs(steer_change) > 0.15:
            steer_penalty = -0.5
        else:
            steer_penalty = 0.0
        
        return accel_penalty + steer_penalty
    
    def _compute_efficiency_reward(
        self,
        ego_state: Dict[str, float],
        target_speed: float
    ) -> float:
        """效率奖励：鼓励接近目标速度"""
        current_speed = ego_state.get('speed', 0.0)
        speed_error = abs(current_speed - target_speed)
        
        if speed_error < 1.0:  # 误差 < 1 m/s
            return 1.0
        elif speed_error < 3.0:  # 误差 < 3 m/s
            return 0.5
        elif speed_error < 5.0:  # 误差 < 5 m/s
            return 0.0
        else:
            return -1.0
    
    def _compute_rule_reward(
        self,
        ego_state: Dict[str, float]
    ) -> float:
        """规则奖励：鼓励车道保持 (简化版)"""
        # 简化假设：y坐标偏离中心线的程度
        lateral_offset = abs(ego_state.get('y', 0.0))
        
        if lateral_offset < 0.5:  # 0.5米内
            return 1.0
        elif lateral_offset < 1.0:
            return 0.5
        elif lateral_offset < 2.0:
            return 0.0
        else:
            return -1.0
    
    def extract_sample_sequence(
        self,
        scene_name: str,
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        提取完整场景序列的动作和奖励
        
        Returns:
            每个样本的 {action, reward, prev_action}
        """
        sequence_info = []
        prev_action = None
        
        for i, sample in enumerate(samples):
            # 提取动作
            action = self.extract_action_from_can(scene_name, sample)
            
            if action is None:
                # 备选：从位姿计算
                next_sample = samples[i + 1] if i + 1 < len(samples) else None
                action = self.extract_action_from_pose(sample, next_sample)
            
            # 获取ego和objects (需要预先提取)
            ego_state = {'speed': 0.0, 'y': 0.0}  # TODO: 从实际数据填充
            objects = []  # TODO: 从实际数据填充
            
            # 计算奖励
            reward = self.compute_reward(
                ego_state=ego_state,
                objects=objects,
                action=action,
                prev_action=prev_action
            )
            
            sequence_info.append({
                'action': action,
                'reward': reward,
                'prev_action': prev_action,
            })
            
            prev_action = action
        
        return sequence_info
