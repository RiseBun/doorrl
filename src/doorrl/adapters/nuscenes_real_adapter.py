"""NuScenes真实数据Adapter - 将nuScenes数据转换为DOOR-RL token schema"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from doorrl.adapters.base import (
    AdapterDescription,
    BenchmarkMode,
    NormalizedSceneConverter,
    TokenizationSpec,
)
from doorrl.adapters.nuscenes_action_extractor import NuScenesActionExtractor
from doorrl.schema import TokenType


# nuScenes `visibility_token` is a categorical id in {'1','2','3','4'} mapping
# to visibility bins (v1=0-40%, v2=40-60%, v3=60-80%, v4=80-100%). Map to bin
# midpoints so downstream code can treat it as a [0,1] continuous prior.
_VISIBILITY_LEVEL_TO_RATIO: Dict[str, float] = {
    '1': 0.2,
    '2': 0.5,
    '3': 0.7,
    '4': 0.9,
}


class NuScenesRealDataAdapter:
    """
    将nuScenes真实数据转换为DOOR-RL的token schema
    
    数据流程:
    1. 从nuScenes加载场景样本
    2. 提取ego状态、动态对象、地图元素
    3. 计算关系特征(相对位置、速度、TTC等)
    4. 转换为标准化的token格式
    """
    
    # nuScenes类别到DOOR-RL token类型的映射
    CATEGORY_MAPPING = {
        'vehicle.car': TokenType.VEHICLE,
        'vehicle.bus': TokenType.VEHICLE,
        'vehicle.truck': TokenType.VEHICLE,
        'vehicle.construction': TokenType.VEHICLE,
        'vehicle.emergency': TokenType.VEHICLE,
        'vehicle.motorcycle': TokenType.VEHICLE,
        'vehicle.bicycle': TokenType.CYCLIST,
        'human.pedestrian': TokenType.PEDESTRIAN,
        'movable_object.barrier': TokenType.MAP,
        'movable_object.trafficcone': TokenType.MAP,
    }
    
    def __init__(
        self, 
        spec: TokenizationSpec,
        nuscenes_root: str,
        version: str = 'v1.0-trainval',
        use_can_bus: bool = True,
    ) -> None:
        self.spec = spec
        self.converter = NormalizedSceneConverter(spec)
        self.use_can_bus = use_can_bus
        
        # 初始化nuScenes
        print(f"Loading NuScenes {version} from {nuscenes_root}...")
        self.nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=True)
        
        # 初始化动作提取器
        self.action_extractor = NuScenesActionExtractor(
            nusc=self.nusc,
            use_can_bus=use_can_bus
        )
        
        # 可选: 加载CAN总线数据
        if use_can_bus:
            try:
                from nuscenes.can_bus.can_bus_api import NuScenesCanBus
                self.can_bus = NuScenesCanBus(dataroot=nuscenes_root)
                print("CAN bus loaded successfully")
            except Exception as e:
                print(f"CAN bus not available: {e}")
                self.can_bus = None
        else:
            self.can_bus = None
    
    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            name="nuscenes_real",
            mode=BenchmarkMode.OFFLINE_DATASET,
            purpose="Offline tokenization from real nuScenes annotated data for world-model pretraining.",
            expected_inputs=[
                "nuScenes scene tokens",
                "annotated objects with 3D boxes",
                "ego pose and kinematics",
                "map data (optional)",
            ],
            outputs=[
                "scene tokens (ego, objects, map, relations)",
                "token mask and types",
                "next-step targets",
                "optional rewards and continuation flags",
            ],
        )
    
    def get_scene_samples(self, scene_name: str) -> List[Dict[str, Any]]:
        """获取指定场景的所有样本"""
        scene = None
        for s in self.nusc.scene:
            if s['name'] == scene_name:
                scene = s
                break
        
        if scene is None:
            raise ValueError(f"Scene {scene_name} not found")
        
        samples = []
        sample_token = scene['first_sample_token']
        
        while sample_token:
            sample = self.nusc.get('sample', sample_token)
            samples.append(sample)
            sample_token = sample['next']
        
        return samples
    
    def convert_sample_to_scene_item(
        self, 
        sample: Dict[str, Any],
        next_sample: Optional[Dict[str, Any]] = None,
        compute_relations: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        将单个nuScenes sample转换为DOOR-RL scene item
        
        Args:
            sample: nuScenes sample字典
            next_sample: 下一帧sample (用于计算action和next_tokens)
            compute_relations: 是否计算关系特征
        
        Returns:
            符合DOOR-RL schema的字典
        """
        # 1. 提取ego状态
        ego_state = self._extract_ego_state(sample)
        
        # 2. 提取动态对象
        objects = self._extract_objects(sample)
        
        # 3. 提取地图元素
        map_elements = self._extract_map_elements(sample)
        
        # 4. 计算关系特征
        relations = []
        if compute_relations and len(objects) > 0:
            relations = self._compute_relations(ego_state, objects)
        
        # 5. 提取真实动作
        scene_name = self._get_scene_name(sample)
        action = self.action_extractor.extract_action_from_can(scene_name, sample)
        
        if action is None:
            # 备选：从位姿差异计算
            action = self.action_extractor.extract_action_from_pose(
                sample, next_sample
            )
        
        # 6. 计算真实奖励
        reward = self.action_extractor.compute_reward(
            ego_state=ego_state,
            objects=objects,
            action=action,
            prev_action=None  # TODO: 从序列中获取
        )
        
        # 7. 提取next_ego状态 (用于next_tokens)
        next_ego = ego_state
        if next_sample is not None:
            next_ego = self._extract_ego_state(next_sample)
        
        # 8. 构建标准化记录
        normalized_record = {
            'ego': ego_state,
            'next_ego': next_ego,
            'objects': objects,
            'map_elements': map_elements,
            'relations': relations,
            'action': action,
            'reward': reward,
            'continue': 1.0 if next_sample is not None else 0.0,
        }
        
        # 9. 使用converter转换为token格式
        return self.converter.build_scene_item(normalized_record)
    
    def _extract_ego_state(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """提取ego车辆状态"""
        # 获取LIDAR_TOP的sample_data
        sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        
        # 获取ego pose
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        
        # 获取calibrated sensor
        calibrated_sensor = self.nusc.get(
            'calibrated_sensor', 
            sample_data['calibrated_sensor_token']
        )
        
        # 提取位置和旋转
        translation = ego_pose['translation']
        rotation = Quaternion(ego_pose['rotation'])
        
        vx, vy, v_yaw = 0.0, 0.0, 0.0
        if self.can_bus is not None:
            scene_name = self._get_scene_name(sample)
            try:
                pose = self.can_bus.get_messages(scene_name, 'pose')
                if len(pose) > 0:
                    vx = float(pose[-1]['vel'][0])
                    vy = float(pose[-1]['vel'][1])
                    v_yaw = float(pose[-1]['rotation_rate'][2])
            except Exception:
                pass

        # Guard against NaN/Inf leaking into tokens (some CAN fields can be
        # missing/invalid on the edge samples of a scene).
        if not math.isfinite(vx):
            vx = 0.0
        if not math.isfinite(vy):
            vy = 0.0
        if not math.isfinite(v_yaw):
            v_yaw = 0.0

        return {
            'x': 0.0,
            'y': 0.0,
            'vx': vx,
            'vy': vy,
            'heading': v_yaw,
            'length': 4.5,
            'width': 1.8,
            'speed': math.sqrt(vx**2 + vy**2),
        }
    
    def _extract_objects(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取动态对象信息"""
        objects = []
        
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # 确定token类型
            token_type = self._get_token_type(ann['category_name'])
            if token_type is None:
                continue  # 跳过不关心的类别
            
            # 提取3D box信息
            box = self.nusc.get_box(ann_token)
            
            # 转换到ego坐标系
            sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            calibrated_sensor = self.nusc.get(
                'calibrated_sensor', 
                sample_data['calibrated_sensor_token']
            )
            
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)
            box.translate(-np.array(calibrated_sensor['translation']))
            box.rotate(Quaternion(calibrated_sensor['rotation']).inverse)
            
            # nuScenes box_velocity() returns (nan, nan, nan) for annotations
            # at the very first / last frame of an instance, or when only
            # a single sighting exists. Those NaNs would silently propagate
            # into tokens and turn every loss into NaN, so we sanitize here.
            velocity = self.nusc.box_velocity(ann_token)
            vx = float(velocity[0]) if np.isfinite(velocity[0]) else 0.0
            vy = float(velocity[1]) if np.isfinite(velocity[1]) else 0.0

            heading = float(box.orientation.yaw_pitch_roll[0])
            if not math.isfinite(heading):
                heading = 0.0

            cx = float(box.center[0])
            cy = float(box.center[1])
            if not (math.isfinite(cx) and math.isfinite(cy)):
                continue

            # nuScenes visibility_token is a categorical id in {'1','2','3','4'}
            # mapping to visibility bins v1=0-40%, v2=40-60%, v3=60-80%, v4=80-100%.
            # Previously we float()-cast it blindly, which yielded values in
            # {1.0, 2.0, 3.0, 4.0} that later got clamp(0,1)-ed to 1.0 inside
            # the visibility model variant, making OBJECT_RELATION_VISIBILITY
            # indistinguishable from OBJECT_RELATION. Map to bin midpoints so
            # the prior carries real information in [0,1].
            vtoken = str(ann.get('visibility_token', '4'))
            visibility = _VISIBILITY_LEVEL_TO_RATIO.get(vtoken, 1.0)

            obj = {
                'x': cx,
                'y': cy,
                'vx': vx,
                'vy': vy,
                'length': float(box.wlh[0]),
                'width': float(box.wlh[1]),
                'heading': heading,
                'token_type': self._token_type_to_str(token_type),
                'category': ann['category_name'],
                'visibility': visibility,
            }

            objects.append(obj)
        
        # 限制对象数量
        if len(objects) > self.spec.max_dynamic_objects:
            # 按距离排序，保留最近的对象
            objects.sort(key=lambda obj: math.sqrt(obj['x']**2 + obj['y']**2))
            objects = objects[:self.spec.max_dynamic_objects]
        
        return objects
    
    def _extract_map_elements(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        提取地图元素 (简化版)
        
        nuScenes不直接提供lane信息，这里使用:
        1. 从CAN总线提取route信息
        2. 或者使用静态障碍物作为地图标记
        3. 或者基于历史轨迹推断车道
        """
        map_elements = []
        
        # 方案1: 使用movable_object作为简单的map标记
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # 只关心交通锥和护栏
            if 'trafficcone' in ann['category_name'] or 'barrier' in ann['category_name']:
                box = self.nusc.get_box(ann_token)
                
                # 转换到ego坐标系
                sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
                calibrated_sensor = self.nusc.get(
                    'calibrated_sensor', 
                    sample_data['calibrated_sensor_token']
                )
                
                box.translate(-np.array(ego_pose['translation']))
                box.rotate(Quaternion(ego_pose['rotation']).inverse)
                box.translate(-np.array(calibrated_sensor['translation']))
                box.rotate(Quaternion(calibrated_sensor['rotation']).inverse)
                
                map_elem = {
                    'x': float(box.center[0]),
                    'y': float(box.center[1]),
                    'heading': float(box.orientation.yaw_pitch_roll[0]),
                    'length': float(box.wlh[0]),
                    'width': float(box.wlh[1]),
                }
                map_elements.append(map_elem)
        
        # 限制地图元素数量
        if len(map_elements) > self.spec.max_map_tokens:
            map_elements = map_elements[:self.spec.max_map_tokens]
        
        return map_elements
    
    def _compute_relations(
        self, 
        ego_state: Dict[str, float], 
        objects: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """计算ego与对象之间的关系特征"""
        relations = []
        
        ego_vx = ego_state['vx']
        ego_vy = ego_state['vy']
        
        for obj in objects:
            dx = obj['x'] - ego_state['x']
            dy = obj['y'] - ego_state['y']
            distance = math.sqrt(dx**2 + dy**2)
            
            # 相对速度
            rel_vx = obj['vx'] - ego_vx
            rel_vy = obj['vy'] - ego_vy
            rel_speed = math.sqrt(rel_vx**2 + rel_vy**2)
            
            # 碰撞时间 (TTC)
            ttc = self._compute_ttc(dx, dy, rel_vx, rel_vy)
            
            # 碰撞风险
            risk = 1.0 / max(distance, 1.0)
            
            # 车道冲突判断
            lane_conflict = 1.0 if abs(dy) < 2.0 else 0.0
            
            relation = {
                'x': dx,
                'y': dy,
                'vx': rel_vx,
                'vy': rel_vy,
                'distance': distance,
                'rel_speed': rel_speed,
                'ttc': ttc,
                'risk': risk,
                'lane_conflict': lane_conflict,
                'visibility': obj.get('visibility', 1.0),
                'priority': self._compute_priority(obj, ego_state),
            }
            
            relations.append(relation)
        
        # 限制关系数量
        if len(relations) > self.spec.max_relation_tokens:
            relations.sort(key=lambda r: r['risk'], reverse=True)
            relations = relations[:self.spec.max_relation_tokens]
        
        return relations
    
    def _compute_ttc(self, dx: float, dy: float, rel_vx: float, rel_vy: float) -> float:
        """计算碰撞时间 (Time To Collision)"""
        distance = math.sqrt(dx**2 + dy**2)
        rel_speed_along_line = (dx * rel_vx + dy * rel_vy) / max(distance, 0.1)
        
        if rel_speed_along_line > 0:
            ttc = distance / rel_speed_along_line
        else:
            ttc = 999.0  # 不会碰撞
        
        return min(ttc, 20.0)  # 最大20秒
    
    def _compute_priority(
        self,
        obj: Dict[str, Any],
        ego_state: Dict[str, float]
    ) -> float:
        """
        计算交互优先级
        
        优先级规则:
        1. 行人 > 车辆 (安全考虑)
        2. 近距离 > 远距离
        3. 同车道 > 异车道
        4. 前方 > 后方
        """
        priority = 0.5  # 基础优先级
        
        # 行人优先级更高
        if obj.get('token_type') == 'pedestrian':
            priority += 0.3
        
        # 距离越近优先级越高
        distance = obj.get('distance', 999.0)
        if distance < 10.0:
            priority += 0.2
        elif distance < 20.0:
            priority += 0.1
        
        # 同车道优先级更高
        if obj.get('lane_conflict', 0.0) > 0.5:
            priority += 0.2
        
        # 前方物体优先级更高
        if obj.get('x', 0.0) > 0:
            priority += 0.1
        
        return min(priority, 1.0)
    
    def _get_token_type(self, category_name: str) -> Optional[TokenType]:
        """根据nuScenes类别获取DOOR-RL token类型"""
        for prefix, token_type in self.CATEGORY_MAPPING.items():
            if category_name.startswith(prefix):
                return token_type
        return None
    
    def _token_type_to_str(self, token_type: TokenType) -> str:
        """将TokenType转换为字符串"""
        mapping = {
            TokenType.VEHICLE: 'vehicle',
            TokenType.PEDESTRIAN: 'pedestrian',
            TokenType.CYCLIST: 'cyclist',
            TokenType.MAP: 'map',
        }
        return mapping.get(token_type, 'vehicle')
    
    def _get_scene_name(self, sample: Dict[str, Any]) -> str:
        """获取场景名称"""
        for scene in self.nusc.scene:
            if scene['token'] == sample['scene_token']:
                return scene['name']
        raise ValueError(f"Scene not found for token {sample['scene_token']}")
    
    def expected_normalized_schema(self) -> Dict[str, str]:
        return {
            'ego': 'Ego kinematics in ego-centric coordinates from CAN bus.',
            'objects': 'List of annotated dynamic agents with 3D boxes.',
            'map_elements': 'Local topology or lane tokens (TODO).',
            'relations': 'Decision-sufficient relation tokens (TTC, risk, etc.).',
            'action': 'Extracted from CAN bus or teacher planner.',
            'reward': 'Optional reward target (TODO).',
            'continue': 'Episode continuation flag.',
        }
