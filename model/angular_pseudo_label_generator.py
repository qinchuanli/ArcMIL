"""
Angular Pseudo-Label Generator

Pseudo-label generation module for the current ArcMIL release.
实现异常中心检测(Abnormal Center Detection)和MIL约束。

核心流程:
    1. 计算instance到各类abnormal centers的最大相似度
    2. Softmax归一化 → 软标签分布
    3. MIL约束: 阴性bag中所有instance强制为阴性
    4. 置信度过滤: 仅保留高置信度样本

异常中心检测逻辑:
    - 与另一类中心相似度低的中心 → 判别性强 → 用于伪标签
    - 与另一类中心相似度高的中心 → 类间混淆 → 丢弃
    
数学公式:
    q_i = softmax([max_k∈A_0 Sim(θ_i, ω_k^0), max_k∈A_1 Sim(θ_i, ω_k^1)] / τ)
    
    其中 A_c 是类别c的abnormal center索引集合

参考:
    - current ArcMIL manuscript pseudo-label stage
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
from config import Config

from .angular_transform import AngularCoordinateTransformer


class AngularPseudoLabelGenerator(nn.Module):
    """
    角坐标伪标签生成器
    
    功能:
    1. 基于AFC聚类中心的实例级分类
    2. Abnormal center detection (判别性锚点选择)
    3. MIL约束 (阴性bag全负)
    4. 置信度过滤 (高质量监督信号)
    
    Args:
        temperature: Softmax温度参数 T (越小越尖锐)
        confidence_threshold: 置信度阈值 (仅高于此值的样本用于训练)
        use_abnormal_only: 是否只使用abnormal centers
        abnormal_threshold: 异常中心判定阈值 (余弦相似度)
        eps: 数值稳定性常数
    """
    
    def __init__(
        self,
        temperature: float = 0.2,
        confidence_threshold: float = 0.8,
        use_abnormal_only: bool = True,
        abnormal_threshold: float = 0.9,
        eps: float = 1e-7
    ):
        super().__init__()
        
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.use_abnormal_only = use_abnormal_only
        self.abnormal_threshold = abnormal_threshold
        self.eps = eps
        
        # 角坐标变换器
        self.transformer = AngularCoordinateTransformer(eps=eps, use_float64=False)

        # 使用普通实例变量 (不使用 register_buffer 避免可能的维度修改问题)
        # ⚠️ 原因: register_buffer 在某些情况下会导致 tensor 维度异常变化
        self.centers_class0 = None
        self.centers_class1 = None
        self.abnormal_indices_class0 = None
        self.abnormal_indices_class1 = None
        
        # 统计信息
        self._stats = {
            'total_samples': 0,
            'positive_pseudo_labels': 0,
            'high_confidence_count': 0,
            'mil_forced_negative': 0
        }
    
    def set_centers(
        self,
        centers_class0: torch.Tensor,
        centers_class1: torch.Tensor,
        abnormal_indices_class0: Optional[torch.Tensor] = None,
        abnormal_indices_class1: Optional[torch.Tensor] = None
    ):
        """
        设置预计算的AFC聚类中心
        
        Args:
            centers_class0: [K0, D] 或 [K0, D-1] 类别0的中心 (笛卡尔或角坐标)
            centers_class1: [K1, D] 或 [K1, D-1] 类别1的中心
            abnormal_indices_class0: [n_abn0] 类别0的异常中心索引 (可选)
            abnormal_indices_class1: [n_abn1] 类别1的异常中心索引 (可选)
        """
        if isinstance(centers_class0, np.ndarray):
            centers_class0 = torch.from_numpy(centers_class0).float()
        if isinstance(centers_class1, np.ndarray):
            centers_class1 = torch.from_numpy(centers_class1).float()

        # 获取目标设备 (优先使用已初始化的tensor device, 否则使用cuda:0)
        if hasattr(self, 'centers_class0') and self.centers_class0 is not None:
            target_device = self.centers_class0.device
        else:
            target_device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.centers_class0 = centers_class0.to(target_device)
        self.centers_class1 = centers_class1.to(target_device)

        # 加载 FDR 选定的维度索引 (如果存在)
        self.selected_dims = None
        try:
            import json as _json
            centers_dir = getattr(Config, 'CENTERS_DIR', None)
            if centers_dir:
                selected_dims_path = os.path.join(centers_dir, 'discriminative_selection', 'selected_dims_top128.json')
                if os.path.exists(selected_dims_path):
                    with open(selected_dims_path, 'r') as f:
                        dims_data = _json.load(f)
                        self.selected_dims = dims_data.get('selected_dimensions', dims_data.get('selected_dims', None))
                        if self.selected_dims is not None:
                            self.selected_dims = list(map(int, self.selected_dims))
                            print(f"[PseudoLabel] Loaded {len(self.selected_dims)} selected dimensions from FDR")
        except Exception as e:
            print(f"[PseudoLabel] Warning: Could not load selected_dims: {e}")
        
        # 如果没有提供abnormal indices, 自动计算
        if abnormal_indices_class0 is None or abnormal_indices_class1 is None:
            print("[PseudoLabel] No abnormal indices provided, computing...")
            abnormal_indices_class0, abnormal_indices_class1 = \
                self.identify_abnormal_centers()
        
        if abnormal_indices_class0 is not None:
            self.abnormal_indices_class0 = abnormal_indices_class0.long()
        else:
            self.abnormal_indices_class0 = torch.arange(len(centers_class0))
            
        if abnormal_indices_class1 is not None:
            self.abnormal_indices_class1 = abnormal_indices_class1.long()
        else:
            self.abnormal_indices_class1 = torch.arange(len(centers_class1))
        
        print(f"[PseudoLabel] Centers set:")
        print(f"  Class 0 ({Config.CLASS0_NAME}): {len(self.centers_class0)} centers")
        print(f"  Class 1 ({Config.CLASS1_NAME}): {len(self.centers_class1)} centers")
        print(f"  Abnormal centers (Class 0): {len(self.abnormal_indices_class0)} / {len(self.centers_class0)}")
        print(f"  Abnormal centers (Class 1): {len(self.abnormal_indices_class1)} / {len(self.centers_class1)}")
    
    def identify_abnormal_centers(
        self,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        异常中心检测 (Abnormal Center Detection)
        
        核心思想:
        - 计算类间中心相似度矩阵
        - 对于Class 0的每个中心, 找到与Class 1所有中心的最大相似度
        - 如果最大相似度 < threshold → 该中心与Class 1差异大 → 判别性强 → abnormal
        - 如果最大相似度 ≥ threshold → 该中心与Class 1相似 → 可能混淆 → normal
        
        Returns:
            abnormal_idx0: Tensor 类别0的abnormal中心索引
            abnormal_idx1: Tensor 类别1的abnormal中心索引
        """
        if threshold is None:
            threshold = self.abnormal_threshold
        
        c0 = self.centers_class0  # [K0, D]
        c1 = self.centers_class1  # [K1, D]
        
        # 检测输入是笛卡尔还是角坐标
        is_cartesian = (c0.shape[-1] == c1.shape[-1]) and (c0.shape[-1] > c0.shape[0])
        
        if is_cartesian:
            # 笛卡尔坐标: 直接用余弦相似度
            sim_cross = torch.mm(c0, c1.T)  # [K0, K1]
        else:
            # 角坐标: 使用角坐标相似度
            sim_cross = self.transformer.compute_angular_similarity(c0, c1)  # [K0, K1]
        
        # Class 0: 对每个center找与Class 1的最大相似度
        max_sim_to_c1 = sim_cross.max(dim=1)[0]  # [K0]
        abnormal_mask_c0 = max_sim_to_c1 < threshold
        
        # Class 1: 对每个center找与Class 0的最大相似度
        max_sim_to_c0 = sim_cross.max(dim=0)[0]  # [K1]
        abnormal_mask_c1 = max_sim_to_c0 < threshold
        
        abnormal_idx0 = torch.where(abnormal_mask_c0)[0]
        abnormal_idx1 = torch.where(abnormal_mask_c1)[0]
        
        print(f"\n[Abnormal Center Detection]")
        print(f"  Threshold: {threshold}")
        print(f"  Cross-similarity matrix shape: {sim_cross.shape}")
        print(f"  Class 0: {len(abnormal_idx0)} abnormal / {len(c0)} total")
        print(f"  Class 1: {len(abnormal_idx1)} abnormal / {len(c1)} total")
        
        # 打印详细信息
        if len(abnormal_idx0) > 0 and len(abnormal_idx0) < 10:
            print(f"  Class 0 abnormal indices: {abnormal_idx0.tolist()}")
            print(f"  Class 0 max sim to C1: {max_sim_to_c0[abnormal_idx0].tolist()}")
        if len(abnormal_idx1) > 0 and len(abnormal_idx1) < 10:
            print(f"  Class 1 abnormal indices: {abnormal_idx1.tolist()}")
            print(f"  Class 1 max sim to C0: {max_sim_to_c1[abnormal_idx1].tolist()}")
        
        return abnormal_idx0, abnormal_idx1
    
    def generate(
        self,
        features: torch.Tensor,
        bag_label: int,
        mode: str = 'cartesian'
    ) -> Dict[str, torch.Tensor]:
        """
        为一个bag中的instances生成伪标签
        
        完整流程:
        1. 特征转换到角坐标 (如果需要)
        2. 计算到各类abnormal centers的最大相似度
        3. Softmax归一化 → 软标签
        4. MIL约束
        5. 置信度过滤
        
        Args:
            features: [N, D] instances特征 (笛卡尔坐标)
            bag_label: int bag标签 (0=Normal/Primary, 1=Tumor/Metastatic)
            mode: 'cartesian' 或 'angular' (输入格式)
        
        Returns:
            dict {
                'pseudo_labels': [N, 2] 软标签分布 (one-hot-like),
                'confidence': [N] 置信度 (max概率),
                'valid_mask': [N] 高置信度掩码,
                'sim_to_class0': [N] 到Class 0的最大相似度,
                'sim_to_class1': [N] 到Class 1的最大相似度
            }
        """
        N = features.shape[0]
        device = features.device

        # 验证centers已加载
        if self.centers_class0 is None or self.centers_class1 is None:
            raise ValueError("Centers not set! Call set_centers() first.")

        # 确保centers在正确的设备上 (动态同步到当前batch的device)
        if self.centers_class0.device != device:
            self.centers_class0 = self.centers_class0.to(device)
            self.centers_class1 = self.centers_class1.to(device)
            if self.abnormal_indices_class0 is not None:
                self.abnormal_indices_class0 = self.abnormal_indices_class0.to(device)
            if self.abnormal_indices_class1 is not None:
                self.abnormal_indices_class1 = self.abnormal_indices_class1.to(device)
        
        # ===== Step 1: 特征转换 =====
        if mode == 'cartesian':
            # L2归一化
            h_hat = torch.nn.functional.normalize(features, p=2, dim=-1)
            theta = self.transformer.cartesian_to_angular(h_hat)

            # 应用 Sin/Cos 周期性修复 (与聚类时保持一致: 511 → 512 维)
            from .angular_transform import fix_last_dim_periodicity
            theta = fix_last_dim_periodicity(theta)  # [N, 512]

            # ⭐ Option B: 应用去中心化 (与聚类时 compute_centers.py 保持一致)
            # 仅对前510维去中心化，cos/sin列(最后2列)保持不变
            if hasattr(Config, 'USE_DECENTERED') and Config.USE_DECENTERED:
                import numpy as np
                pi_over_2 = torch.tensor(np.pi / 2, dtype=theta.dtype, device=theta.device)
                theta[:, :-2] = theta[:, :-2] - pi_over_2

            # FDR 降维: 应用与聚类时相同的维度选择 (如果需要)
            if hasattr(self, 'selected_dims') and self.selected_dims is not None:
                theta = theta[:, self.selected_dims]  # [N, p] where p=128
        elif mode == 'angular':
            theta = features
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # ===== Step 2: 选择使用的中心 =====
        if self.use_abnormal_only and self.abnormal_indices_class0 is not None:
            # 安全检查: 如果abnormal_indices为空，回退到使用所有centers
            if len(self.abnormal_indices_class0) > 0:
                abn_c0 = self.centers_class0[self.abnormal_indices_class0]  # [n_abn0, D]
            else:
                abn_c0 = self.centers_class0  # 回退: 使用所有centers
                # print(f"[WARN] Class 0 has 0 abnormal centers, using all {len(abn_c0)} centers")

            if len(self.abnormal_indices_class1) > 0:
                abn_c1 = self.centers_class1[self.abnormal_indices_class1]  # [n_abn1, D]
            else:
                abn_c1 = self.centers_class1  # 回退: 使用所有centers
                # print(f"[WARN] Class 1 has 0 abnormal centers, using all {len(abn_c1)} centers")
        else:
            abn_c0 = self.centers_class0
            abn_c1 = self.centers_class1
        
        # ===== Step 3: 计算相似度 =====
        # 到Class 0的所有abnormal中心的相似度
        sim_to_all_c0 = self.transformer.compute_angular_similarity(theta, abn_c0)  # [N, n_abn0]
        max_sim_c0 = sim_to_all_c0.max(dim=1)[0]  # [N]
        
        # 到Class 1的所有abnormal中心的相似度
        sim_to_all_c1 = self.transformer.compute_angular_similarity(theta, abn_c1)  # [N, n_abn1]
        max_sim_c1 = sim_to_all_c1.max(dim=1)[0]  # [N]
        
        # ===== Step 4: Softmax归一化 → 软标签 =====
        scores = torch.stack([max_sim_c0, max_sim_c1], dim=-1)  # [N, 2]
        pseudo_labels = torch.softmax(scores / self.temperature, dim=-1)  # [N, 2]
        
        # ===== Step 5: MIL约束 =====
        # MIL约束: 阴性bag中所有instance都应该是阴性的
        if bag_label == 0:
            forced_negative = torch.zeros_like(pseudo_labels)
            forced_negative[:, 0] = 1.0  # 强制全部为Class 0
            pseudo_labels = forced_negative
            
            self._stats['mil_forced_negative'] += N
        
        # ===== Step 6: 置信度计算和过滤 =====
        confidence = pseudo_labels.max(dim=1)[0]  # [N]
        valid_mask = confidence >= self.confidence_threshold  # [N]
        
        # 更新统计
        self._stats['total_samples'] += N
        self._stats['high_confidence_count'] += valid_mask.sum().item()
        if bag_label == 1:
            positive_preds = (pseudo_labels.argmax(dim=1) == 1).sum().item()
            self._stats['positive_pseudo_labels'] += positive_preds
        
        return {
            'pseudo_labels': pseudo_labels,
            'confidence': confidence,
            'valid_mask': valid_mask,
            'sim_to_class0': max_sim_c0,
            'sim_to_class1': max_sim_c1
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        stats = self._stats.copy()
        if stats['total_samples'] > 0:
            stats['high_confidence_ratio'] = stats['high_confidence_count'] / stats['total_samples']
            stats['positive_ratio'] = stats['positive_pseudo_labels'] / max(stats['total_samples'], 1)
        return stats
    
    def reset_statistics(self):
        """重置统计"""
        self._stats = {
            'total_samples': 0,
            'positive_pseudo_labels': 0,
            'high_confidence_count': 0,
            'mil_forced_negative': 0
        }


# 为了在模块中使用Config (避免循环导入)
try:
    from config import Config
except ImportError:
    pass


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config
    
    print("=" * 70)
    print("Testing Angular Pseudo-Label Generator")
    print("=" * 70)
    
    generator = AngularPseudoLabelGenerator(
        temperature=0.2,
        confidence_threshold=0.8,
        use_abnormal_only=True,
        abnormal_threshold=0.9
    )
    
    # 创建模拟聚类中心
    D_minus_1 = 511  # 角坐标维度
    
    # Class 0 centers (3个)
    centers_c0 = torch.randn(3, D_minus_1)
    centers_c0[..., :D_minus_1-1] = torch.clamp(centers_c0[..., :D_minus_1-1], 0.01, np.pi-0.01)
    centers_c0[..., -1] = torch.clamp(centers_c0[..., -1], -(np.pi-0.01), np.pi-0.01)
    
    # Class 1 centers (3个)
    centers_c1 = torch.randn(3, D_minus_1)
    centers_c1[..., :D_minus_1-1] = torch.clamp(centers_c1[..., :D_minus_1-1], 0.01, np.pi-0.01)
    centers_c1[..., -1] = torch.clamp(centers_c1[..., -1], -(np.pi-0.01), np.pi-0.01)
    
    # 设置中心
    generator.set_centers(centers_c0, centers_c1)
    
    # 测试正样本bag
    print("\n[Test 1] Positive Bag (Tumor)")
    N_pos = 50
    h_pos = torch.randn(N_pos, 512)
    h_pos = torch.nn.functional.normalize(h_pos, p=2, dim=-1)
    
    result_pos = generator.generate(h_pos, bag_label=1, mode='cartesian')
    print(f"  Instances: {N_pos}")
    print(f"  Pseudo-labels shape: {result_pos['pseudo_labels'].shape}")
    print(f"  Confidence range: [{result_pos['confidence'].min():.4f}, {result_pos['confidence'].max():.4f}]")
    print(f"  High-confidence samples: {result_pos['valid_mask'].sum().item()} / {N_pos}")
    print(f"  Avg confidence: {result_pos['confidence'].mean():.4f}")
    
    # 测试负样本bag
    print("\n[Test 2] Negative Bag (Normal) - MIL constraint")
    N_neg = 30
    h_neg = torch.randn(N_neg, 512)
    h_neg = torch.nn.functional.normalize(h_neg, p=2, dim=-1)
    
    result_neg = generator.generate(h_neg, bag_label=0, mode='cartesian')
    print(f"  Instances: {N_neg}")
    print(f"  All forced to negative? {(result_neg['pseudo_labels'].argmax(dim=1) == 0).all().item()}")
    print(f"  Confidence (should be all 1.0): {result_neg['confidence'].unique()}")
    
    # 打印统计
    print("\n[Statistics]")
    stats = generator.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
