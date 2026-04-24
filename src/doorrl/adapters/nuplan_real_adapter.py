"""NuPlan真实数据Adapter - 将nuPlan数据转换为DOOR-RL token schema"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

from doorrl.adapters.base import (
    AdapterDescription,
    BenchmarkMode,
    NormalizedSceneConverter,
    TokenizationSpec,
)
from doorrl.schema import TokenType


class NuPlanRealDataAdapter:
    """
    将nuPlan真实数据转换为DOOR-RL的token schema
    
    数据流程:
    1. 从nuPlan数据库加载场景
    2. 提取ego状态、动态对象、地图元素
    3. 计算关系特征
    4. 支持reactive和non-reactive模式
    5. 转换为标准化的token格式
    """
    
    def __init__(
        self, 
        spec: TokenizationSpec,
        nuplan_root: str,
        map_version: str = 'nuplan-maps-v1.0',
        reactive: bool = True,
    ) -> None:
        self.spec = spec
        self.converter = NormalizedSceneConverter(spec)
        self.reactive = reactive
        self.nuplan_root = Path(nuplan_root)
        self.map_version = map_version
        
        # TODO: 初始化nuPlan数据库连接
        # from nuplan.planning.utils.nuplan_db.nuplan_scenario import NuPlanScenario
        # self.scenario_builder = ScenarioBuilder(...)
        
        print(f"Nuplan adapter initialized (reactive={reactive})")
        print(f"Data root: {nuplan_root}")
        print(f"Map version: {map_version}")
    
    @property
    def mode(self) -> BenchmarkMode:
        return (
            BenchmarkMode.CLOSED_LOOP_REACTIVE
            if self.reactive
            else BenchmarkMode.CLOSED_LOOP_NON_REACTIVE
        )
    
    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            name="nuplan_real",
            mode=self.mode,
            purpose="Primary closed-loop benchmark for reactive vs non-reactive driving evaluation.",
            expected_inputs=[
                "nuPlan scenario database",
                "planner observation",
                "tracked objects",
                "map context",
                "ego command or trajectory target",
            ],
            outputs=[
                "oracle scene tokens",
                "closed-loop metrics",
                "reactive/non-reactive experiment tags",
            ],
        )
    
    def convert_scenario_to_scene_items(
        self,
        scenario: Any,  # NuPlanScenario类型
        num_samples: int = 10,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        将nuPlan场景转换为多个scene items
        
        Args:
            scenario: nuPlan场景对象
            num_samples: 采样的帧数
        
        Returns:
            scene items列表
        """
        scene_items = []
        
        # TODO: 实现nuPlan场景的采样和转换
        # for frame_idx in range(0, scenario.duration, scenario.database_interval):
        #     observation = scenario.get_observation_at(frame_idx)
        #     scene_item = self._convert_observation(observation)
        #     scene_items.append(scene_item)
        
        return scene_items
    
    def convert_observation_to_scene_item(
        self,
        observation: Any,  # Observation类型
    ) -> Dict[str, torch.Tensor]:
        """
        将单个nuPlan observation转换为scene item
        
        Args:
            observation: nuPlan观察对象
        
        Returns:
            符合DOOR-RL schema的字典
        """
        # 1. 提取ego状态
        ego_state = self._extract_ego_state(observation)
        
        # 2. 提取动态对象
        objects = self._extract_objects(observation)
        
        # 3. 提取地图元素
        map_elements = self._extract_map_elements(observation)
        
        # 4. 计算关系特征
        relations = self._compute_relations(ego_state, objects)
        
        # 5. 构建标准化记录
        normalized_record = {
            'ego': ego_state,
            'objects': objects,
            'map_elements': map_elements,
            'relations': relations,
            'action': [0.0, 0.0],  # TODO: 从planner获取
            'reward': 0.0,
            'continue': 1.0,
        }
        
        # 6. 使用converter转换为token格式
        return self.converter.build_scene_item(normalized_record)
    
    def _extract_ego_state(self, observation: Any) -> Dict[str, float]:
        """提取ego车辆状态"""
        # TODO: 从observation中提取ego状态
        # ego_state = observation.ego_state
        
        return {
            'x': 0.0,
            'y': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'heading': 0.0,
            'length': 4.5,
            'width': 1.8,
            'speed': 0.0,
        }
    
    def _extract_objects(self, observation: Any) -> List[Dict[str, Any]]:
        """提取动态对象"""
        objects = []
        
        # TODO: 从observation中提取tracked_objects
        # tracked_objects = observation.tracked_objects
        # for obj in tracked_objects:
        #     obj_dict = {
        #         'x': obj.center.x,
        #         'y': obj.center.y,
        #         'vx': obj.velocity.x,
        #         'vy': obj.velocity.y,
        #         'length': obj.box.length,
        #         'width': obj.box.width,
        #         'heading': obj.center.heading,
        #         'token_type': self._get_token_type(obj.tracked_object_type),
        #     }
        #     objects.append(obj_dict)
        
        return objects
    
    def _extract_map_elements(self, observation: Any) -> List[Dict[str, Any]]:
        """提取地图元素"""
        map_elements = []
        
        # TODO: 从observation中提取地图元素
        # route_roadblocks = observation.route_roadblocks
        # for roadblock in route_roadblocks:
        #     for lane in roadblock.lanes:
        #         map_elements.append({
        #             'x': lane.centerline[0].x,
        #             'y': lane.centerline[0].y,
        #             'heading': lane.centerline.heading,
        #         })
        
        return map_elements
    
    def _compute_relations(
        self,
        ego_state: Dict[str, float],
        objects: List[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        """计算关系特征"""
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
            
            # 碰撞时间
            ttc = self._compute_ttc(dx, dy, rel_vx, rel_vy)
            
            # 碰撞风险
            risk = 1.0 / max(distance, 1.0)
            
            # 车道冲突
            lane_conflict = 1.0 if abs(dy) < 2.0 else 0.0
            
            relation = {
                'x': dx,
                'y': dy,
                'vx': rel_vx,
                'vy': rel_vy,
                'distance': distance,
                'ttc': ttc,
                'risk': risk,
                'lane_conflict': lane_conflict,
                'visibility': 1.0,
                'priority': 0.5,
            }
            
            relations.append(relation)
        
        # 限制数量
        if len(relations) > self.spec.max_relation_tokens:
            relations.sort(key=lambda r: r['risk'], reverse=True)
            relations = relations[:self.spec.max_relation_tokens]
        
        return relations
    
    def _compute_ttc(self, dx: float, dy: float, rel_vx: float, rel_vy: float) -> float:
        """计算碰撞时间"""
        distance = math.sqrt(dx**2 + dy**2)
        rel_speed_along_line = (dx * rel_vx + dy * rel_vy) / max(distance, 0.1)
        
        if rel_speed_along_line > 0:
            ttc = distance / rel_speed_along_line
        else:
            ttc = 999.0
        
        return min(ttc, 20.0)
    
    def _get_token_type(self, object_type: Any) -> str:
        """根据nuPlan对象类型获取token类型字符串"""
        # TODO: 实现nuPlan对象类型到token类型的映射
        return 'vehicle'
    
    def supported_experiments(self) -> Dict[str, str]:
        return {
            'replay_train_replay_test': 'Train and test in non-reactive mode.',
            'replay_train_reactive_test': 'Train in non-reactive mode, test in reactive mode.',
            'reactive_train_reactive_test': 'Train and test in reactive mode.',
        }
    
    def convert_nuplan_observation(self, observation: Any) -> Dict[str, Any]:
        """
        Hook this method to the nuPlan devkit observation object and convert it 
        into the normalized scene schema used by build_scene_item_from_normalized().
        """
        return self.convert_observation_to_scene_item(observation)
    
    def expected_normalized_schema(self) -> Dict[str, str]:
        return {
            'ego': 'Ego kinematics in ego-centric coordinates.',
            'objects': 'List of tracked dynamic agents.',
            'map_elements': 'Route roadblocks and lanes.',
            'relations': 'Decision-sufficient relation tokens.',
            'action': 'Planner action or trajectory target.',
            'reward': 'Optional reward target.',
            'continue': 'Episode continuation flag.',
        }
