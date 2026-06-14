"""
Angular Coordinate Transformer

实现球面笛卡尔坐标与角坐标之间的双向转换,
以及角坐标空间中的相似度计算。

核心公式:
    正向变换 T: S^{D-1} -> Θ ⊂ R^{D-1}
        θ₁ = arccos(x₁)
        θₖ = arccos(xₖ / (∏ⱼ₁ᵏ⁻¹ sinθⱼ))   k=2,...,D-2
        θ_{D-1} = atan2(x_D, x_{D-1})          [-π, π]
    
    反向变换 T^{-1}: Θ -> S^{D-1}
        x₁ = cos(θ₁)
        xₖ = (∏ⱼ₁ᵏ⁻¹ sinθⱼ) · cos(θₖ)      k=2,...,D-1
        x_D = ∏ⱼ₁ᴰ⁻¹ sinθⱼ
    
    角坐标相似度 (O(D), 无需反向变换):
        Sim(θ, φ) = cos(θ₁)cos(φ₁) + Σₖ[∏ⱼ<ₖ sin(θⱼ)sin(φⱼ)]·cos(θₖ-φₖ)

数值稳定性:
    - 使用float64中间计算
    - clamp到[-1+ε, 1-ε]避免arccos梯度爆炸
    - 添加小常数防止除零

参考:
    - Xiao (2026): Lossless Embedding Compression via Spherical Coordinates
    - Cai et al. (2013): On the Distribution of Spherical Angles
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, Optional


class AngularCoordinateTransformer(nn.Module):
    """
    球坐标变换器
    
    提供以下功能:
    1. 笛卡尔坐标 ↔ 角坐标转换
    2. 角坐标相似度计算 (O(D))
    3. 维度选择/降维 (可选)
    4. Jacobian因子计算
    
    Args:
        eps: 数值稳定性小常数
        use_float64: 是否使用float64中间计算 (推荐True)
    """
    
    def __init__(self, eps: float = 1e-8, use_float64: bool = True):
        super().__init__()
        self.eps = eps
        self.use_float64 = use_float64
        
    def cartesian_to_angular(self, h_hat: torch.Tensor) -> torch.Tensor:
        """
        正向变换: 笛卡尔坐标 → 角坐标
        
        将L2归一化的特征向量从超球面S^{D-1}转换到角坐标空间Θ⊂R^{D-1}
        
        Args:
            h_hat: [..., D] L2归一化特征向量 (||h_hat||₂ = 1)
        
        Returns:
            theta: [..., D-1] 角坐标向量
                - theta[:, 0:D-2] ∈ [0, π]
                - theta[:, D-1] ∈ [-π, π]
        """
        if self.use_float64:
            original_dtype = h_hat.dtype
            h_hat = h_hat.double()
        else:
            original_dtype = h_hat.dtype
            
        # 确保输入已归一化
        h_hat = torch.nn.functional.normalize(h_hat, p=2, dim=-1)
        
        D = h_hat.shape[-1]
        batch_shape = h_hat.shape[:-1]
        
        # 预分配输出tensor
        theta = torch.zeros(*batch_shape, D - 1, dtype=h_hat.dtype, device=h_hat.device)
        
        # 连乘积 ∏ sin(θⱼ), 初始化为1
        sin_product = torch.ones(*batch_shape, dtype=h_hat.dtype, device=h_hat.device)
        
        # 计算前D-2个角度: θ₁, ..., θ_{D-2}
        for i in range(D - 2):
            x_i = h_hat[..., i]
            
            # cos(θᵢ) = xᵢ / (∏ⱼ<ᵢ sin(θⱼ))
            denom = sin_product + self.eps
            cos_theta_i = x_i / denom
            
            # clamp到安全范围避免arccos数值问题
            cos_theta_i = torch.clamp(cos_theta_i, -(1 - self.eps), (1 - self.eps))
            
            # θᵢ = arccos(cos(θᵢ))
            theta[..., i] = torch.arccos(cos_theta_i)
            
            # 更新连乘积: ∏ⱼ≤ᵢ sin(θⱼ)
            sin_product = sin_product * torch.sin(theta[..., i])
        
        # 最后一个角度: θ_{D-1} = atan2(x_D, x_{D-1}), 范围[-π, π]
        theta[..., D - 2] = torch.atan2(h_hat[..., D - 1], h_hat[..., D - 2])
        
        # 转回原始精度
        if self.use_float64:
            theta = theta.to(original_dtype)
            
        return theta
    
    def angular_to_cartesian(self, theta: torch.Tensor) -> torch.Tensor:
        """
        反向变换: 角坐标 → 笛卡尔坐标
        
        Args:
            theta: [..., D-1] 角坐标向量
                - theta[..., 0:D-2] ∈ [0, π]
                - theta[..., D-1] ∈ [-π, π]
        
        Returns:
            h_hat: [..., D] L2归一化特征向量 (自动归一化)
        """
        if self.use_float64:
            original_dtype = theta.dtype
            theta = theta.double()
        else:
            original_dtype = theta.dtype
            
        D_minus_1 = theta.shape[-1]
        D = D_minus_1 + 1
        batch_shape = theta.shape[:-1]
        
        # 预分配输出
        h_hat_list = []
        
        # 连乘积 ∏ sin(θⱼ), 初始化为1
        sin_product = torch.ones(*batch_shape, dtype=theta.dtype, device=theta.device)
        
        # 计算前D-1个分量
        for i in range(D - 1):
            # xᵢ = (∏ⱼ<ᵢ sin(θⱼ)) · cos(θᵢ)
            x_i = sin_product * torch.cos(theta[..., i])
            h_hat_list.append(x_i)
            
            # 更新连乘积
            sin_product = sin_product * torch.sin(theta[..., i])
        
        # 最后一个分量: x_D = ∏ⱼ₁ᴰ⁻¹ sin(θⱼ)
        x_D = sin_product
        h_hat_list.append(x_D)
        
        # 堆叠成 [..., D]
        h_hat = torch.stack(h_hat_list, dim=-1)
        
        # L2归一化 (确保在单位超球面上)
        h_hat = torch.nn.functional.normalize(h_hat, p=2, dim=-1)
        
        if self.use_float64:
            h_hat = h_hat.to(original_dtype)
            
        return h_hat
    
    def compute_angular_similarity(
        self, 
        theta_i: torch.Tensor, 
        theta_j: torch.Tensor,
        return_cosine: bool = False
    ) -> torch.Tensor:
        """
        计算角坐标相似度 (等价于原始余弦相似度!)
        
        使用O(D)反向递推算法,无需反向变换到笛卡尔坐标。
        
        公式:
        Sim(θ, φ) = cos(θ₁)cos(φ₁) 
                    + Σ_{k=1}^{D-2} [(∏_{l=0}^{k-1} sin(θ_l)sin(φ_l))] · cos(θ_k - φ_k)
        
        Args:
            theta_i: [N, D-1] 或 [..., D-1] 第一组角坐标
            theta_j: [M, D-1] 或 [..., D-1] 第二组角坐标
            return_cosine: 是否返回cosine值 (而非角度距离)
        
        Returns:
            similarity: [N, M] 或 [...] 相似度矩阵
                - 范围 [-1, 1], 等价于原始内积 h_i^T h_j
        """
        # ⚠️ 子空间警告 (新增!)
        D_input = theta_i.shape[-1]
        if hasattr(self, 'dimension') and D_input != self.dimension:
            import warnings
            warnings.warn(
                f"⚠️ Input dimension ({D_input}) != expected ({self.dimension}). "
                f"This is a SUBSPACE of angular coordinates! "
                f"The angular similarity formula assumes complete nested structure. "
                f"Consider using Euclidean distance or cosine similarity instead.",
                UserWarning,
                stacklevel=2
            )
        
        # 处理batch情况
        if theta_i.dim() == 2 and theta_j.dim() == 2:
            N = theta_i.shape[0]
            M = theta_j.shape[0]
            D_minus_1 = theta_i.shape[-1]
            
            # 扩展维度以便广播: [N, 1, D-1] 和 [1, M, D-1]
            theta_i_exp = theta_i.unsqueeze(1)   # [N, 1, D-1]
            theta_j_exp = theta_j.unsqueeze(0)   # [1, M, D-1]
            
            # 第一项: cos(θ₁)·cos(φ₁)
            sim = torch.cos(theta_i_exp[..., 0]) * torch.cos(theta_j_exp[..., 0])
            
            # 初始化连乘积
            sin_prod_i = torch.ones(N, M, device=theta_i.device, dtype=theta_i.dtype)
            sin_prod_j = torch.ones(N, M, device=theta_j.device, dtype=theta_j.dtype)
            
            # 累加第2项到第D-1项
            for k in range(1, D_minus_1):
                # 更新连乘积: ∏_{l=0}^{k-1} sin(θ_l) 和 sin(φ_l)
                sin_prod_i = sin_prod_i * torch.sin(theta_i_exp[..., k-1])
                sin_prod_j = sin_prod_j * torch.sin(theta_j_exp[..., k-1])
                
                # cos(θ_k)·cos(φ_k) 乘积项
                cos_term = torch.cos(theta_i_exp[..., k]) * torch.cos(theta_j_exp[..., k])
                
                # 累加到相似度
                sim = sim + sin_prod_i * sin_prod_j * cos_term
            
            return sim
            
        else:
            # 单样本或共享维度情况
            D_minus_1 = theta_i.shape[-1]
            
            # 第一项
            sim = torch.cos(theta_i[..., 0]) * torch.cos(theta_j[..., 0])
            
            # 初始化连乘积
            sin_prod_i = torch.ones_like(theta_i[..., 0])
            sin_prod_j = torch.ones_like(theta_j[..., 0])
            
            for k in range(1, D_minus_1):
                sin_prod_i = sin_prod_i * torch.sin(theta_i[..., k-1])
                sin_prod_j = sin_prod_j * torch.sin(theta_j[..., k-1])
                
                cos_term = torch.cos(theta_i[..., k]) * torch.cos(theta_j[..., k])
                sim = sim + sin_prod_i * sin_prod_j * cos_term
            
            return sim
    
    def compute_angular_distance(
        self,
        theta_i: torch.Tensor,
        theta_j: torch.Tensor
    ) -> torch.Tensor:
        """
        计算角坐标测地线距离 (球面弧长)
        
        d_Θ(θ, φ) = arccos(Sim(θ, φ))
        
        范围: [0, π]
        
        Args:
            theta_i: [N, D-1] 角坐标
            theta_j: [M, D-1] 角坐标
        
        Returns:
            distance: [N, M] 测地线距离矩阵
        """
        # ⚠️ 子空间警告 (新增!)
        D_input = theta_i.shape[-1]
        if hasattr(self, 'dimension') and D_input != self.dimension:
            import warnings
            warnings.warn(
                f"⚠️ Input dimension ({D_input}) != expected ({self.dimension}). "
                f"This is a SUBSPACE of angular coordinates! "
                f"The geodesic distance formula may not be valid. "
                f"Consider using Euclidean distance instead.",
                UserWarning,
                stacklevel=2
            )
        
        sim = self.compute_angular_similarity(theta_i, theta_j)
        
        # clamp到安全范围避免arccos域错误
        sim_clamped = torch.clamp(sim, -(1 - self.eps), (1 - self.eps))
        
        distance = torch.arccos(sim_clamped)
        
        return distance
    
    def compute_jacobian_factor(
        self,
        theta: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Jacobian因子 J_D(θ)
        
        J_D(θ) = ∏_{j=1}^{D-2} sin^{D-1-j}(θ_j)
        
        这是角度集中定理的来源,用于修正Θ空间的密度估计。
        
        Args:
            theta: [N, D-1] 角坐标
        
        Returns:
            jacobian: [N] Jacobian因子 (每个样本一个标量)
        """
        D_minus_1 = theta.shape[-1]
        D = D_minus_1 + 1
        
        jacobian = torch.ones(theta.shape[0], device=theta.device, dtype=theta.dtype)
        
        for j in range(D_minus_1 - 1):  # j = 1, ..., D-2
            power = D - 1 - j           # D-j-1
            jacobian = jacobian * torch.pow(torch.sin(theta[..., j]), power)
        
        return jacobian
    
    def select_dimensions(
        self,
        theta_all: torch.Tensor,
        strategy: str = 'hybrid',
        m_fixed: int = 32,
        m_adaptive: int = 16,
        var_threshold: float = 0.05,
        cumulative_pct: float = 0.90
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        维度选择 (降维)
        
        基于角度集中定理: Var(θᵢ) ∝ 1/(D-i)
        低索引角度方差大(有信息), 高索引方差小(噪声)
        
        策略:
        - 'fixed': 固定取前m个维度
        - 'variance_threshold': 方差阈值法
        - 'cumulative': 累积贡献率法
        - 'hybrid': 固定 + 自适应混合 (推荐)
        
        Args:
            theta_all: [N, D-1] 所有样本的角坐标
            strategy: 选择策略
            m_fixed: 固定部分维度数
            m_adaptive: 自适应部分维度数
            var_threshold: 方差阈值比例
            cumulative_pct: 累积贡献率目标
        
        Returns:
            theta_reduced: [N, m] 降维后的角坐标
            selected_dims: [m] 选中的维度索引
        """
        N, D_minus_1 = theta_all.shape
        
        if strategy == 'fixed':
            m = min(m_fixed, D_minus_1)
            selected_dims = torch.arange(m, device=theta_all.device, dtype=torch.long)
            
        elif strategy == 'variance_threshold':
            variances = theta_all.var(dim=0)
            max_var = variances.max()
            mask = variances > (max_var * var_threshold)
            selected_dims = torch.where(mask)[0]
            
        elif strategy == 'cumulative':
            variances = theta_all.var(dim=0)
            total_var = variances.sum()
            sorted_idx = torch.argsort(variances, descending=True)
            sorted_vars = variances[sorted_idx]
            cumsum = torch.cumsum(sorted_vars, dim=0)
            n_needed = (cumsum / total_var >= cumulative_pct).nonzero()[0][0].item() + 1
            selected_dims = sorted_idx[:n_needed]
            
            # 强制包含最后一个维度
            if D_minus_1 - 1 not in selected_dims:
                selected_dims = torch.cat([selected_dims, torch.tensor([D_minus_1 - 1])])
                
        elif strategy == 'hybrid':
            # 固定部分: 前m_fixed个低索引维度
            fixed_dims = list(range(min(m_fixed, D_minus_1 - 1)))
            
            # 自适应部分: 从剩余维度中按方差选top m_adaptive个
            if D_minus_1 > m_fixed + 1:
                candidate_range = range(m_fixed, D_minus_1 - 1)
                candidate_vars = theta_all[:, list(candidate_range)].var(dim=0)
                top_adaptive = torch.argsort(candidate_vars, descending=True)[:m_adaptive]
                adaptive_dims = [candidate_range[i.item()] for i in top_adaptive]
            else:
                adaptive_dims = []
            
            # 强制包含最后一个维度 (均匀分布, 有信息)
            last_dim = [D_minus_1 - 1]
            
            # 合并去重
            all_dims = sorted(list(set(fixed_dims + adaptive_dims + last_dim)))
            selected_dims = torch.tensor(all_dims, device=theta_all.device, dtype=torch.long)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 投影到选中的子空间
        theta_reduced = theta_all[:, selected_dims]
        
        print(f"[AngularTransform] Dimension selection:")
        print(f"  Strategy: {strategy}")
        print(f"  Original dim: {D_minus_1}")
        print(f"  Reduced dim: {len(selected_dims)}")
        print(f"  Selected dims: {selected_dims.tolist()}")
        
        return theta_reduced, selected_dims


def fix_last_dim_periodicity(theta_all):
    """
    修复最后一维 θ_{D-1} ∈ [-π, π] 的周期性断点问题

    将周期性的角度表示转换为非周期的二维笛卡尔坐标 (cos θ, sin θ)，
    从而正确处理 ±π 边界处的连续性问题。

    Args:
        theta_all: np.ndarray or torch.Tensor
            角坐标矩阵，形状为 (N, D-1)，其中 D-1 ≥ 2
            最后一维 theta[:, -1] ∈ [-π, π] 是周期变量

    Returns:
        theta_fixed: 与输入相同类型 (np.ndarray 或 torch.Tensor)
            修复后的矩阵，形状为 (N, D) 其中 D = (D-1) + 1
            结构: [前(D-2)维 | cos(θ_{D-1}) | sin(θ_{D-1})]

    Example:
        >>> theta = np.array([[1.0, 2.0, -3.14159], [1.0, 2.0, 3.14159]])
        >>> fixed = fix_last_dim_periodicity(theta)
        >>> fixed.shape
        (2, 4)  # 从3维变为4维
        >>> # 原本距离=6.28的两个点，现在cos/sin值相近

    Raises:
        ValueError: 如果输入维度 < 2
        TypeError: 如果输入类型不是 np.ndarray 或 torch.Tensor
    """
    if isinstance(theta_all, torch.Tensor):
        return _fix_last_dim_periodicity_torch(theta_all)
    elif isinstance(theta_all, np.ndarray):
        return _fix_last_dim_periodicity_numpy(theta_all)
    else:
        raise TypeError(f"Unsupported type: {type(theta_all)}. Expected np.ndarray or torch.Tensor")


def _fix_last_dim_periodicity_numpy(theta_np):
    """NumPy版本的周期性修复实现"""
    if theta_np.ndim != 2:
        raise ValueError(f"Expected 2D array, got {theta_np.ndim}D")

    if theta_np.shape[1] < 2:
        raise ValueError(f"Expected at least 2 dimensions, got {theta_np.shape[1]}")

    last_dim = theta_np[:, -1]  # (N,) ∈ [-π, π]

    # 转换为二维笛卡尔坐标
    cos_last = np.cos(last_dim)   # (N,)
    sin_last = np.sin(last_dim)   # (N,)

    # 组合: 前(D-2)维不变 + (cos, sin)
    theta_fixed = np.concatenate([
        theta_np[:, :-1],           # (N, D-2)
        cos_last.reshape(-1, 1),     # (N, 1)
        sin_last.reshape(-1, 1)      # (N, 1)
    ], axis=1)

    return theta_fixed  # (N, D)


def _fix_last_dim_periodicity_torch(theta_tensor):
    """PyTorch版本的周期性修复实现"""
    if theta_tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {theta_tensor.dim()}D")

    if theta_tensor.shape[1] < 2:
        raise ValueError(f"Expected at least 2 dimensions, got {theta_tensor.shape[1]}")

    last_dim = theta_tensor[:, -1]  # (N,) ∈ [-π, π]

    # 转换为二维笛卡尔坐标
    cos_last = torch.cos(last_dim)   # (N,)
    sin_last = torch.sin(last_dim)   # (N,)

    # 组合: 前(D-2)维不变 + (cos, sin)
    theta_fixed = torch.cat([
        theta_tensor[:, :-1],              # (N, D-2)
        cos_last.unsqueeze(1),             # (N, 1)
        sin_last.unsqueeze(1)              # (N, 1)
    ], dim=1)

    return theta_fixed  # (N, D)


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Angular Coordinate Transformer")
    print("=" * 70)
    
    transformer = AngularCoordinateTransformer(eps=1e-8, use_float64=True)
    
    # 生成测试数据: D=512维单位向量
    N = 100
    D = 512
    
    print(f"\n[Test Data] N={N}, D={D}")
    
    # 创建随机单位向量
    h_random = torch.randn(N, D)
    h_hat = torch.nn.functional.normalize(h_random, p=2, dim=-1)
    
    # ===== Test 1: 正向变换 =====
    print("\n" + "-" * 50)
    print("Test 1: Cartesian → Angular")
    theta = transformer.cartesian_to_angular(h_hat)
    print(f"  Input shape: {h_hat.shape}")
    print(f"  Output shape: {theta.shape}")
    print(f"  θ range (first D-2): [{theta[:, :-1].min():.4f}, {theta[:, :-1].max():.4f}]")
    print(f"  θ_{D-1} range (last): [{theta[:, -1].min():.4f}, {theta[:, -1].max():.4f}]")
    
    # ===== Test 2: 反向变换 =====
    print("\n" + "-" * 50)
    print("Test 2: Angular → Cartesian")
    h_reconstructed = transformer.angular_to_cartesian(theta)
    reconstruction_error = (h_hat - h_reconstructed).norm(dim=1).mean()
    print(f"  Reconstructed shape: {h_reconstructed.shape}")
    print(f"  L2 reconstruction error: {reconstruction_error:.2e}")
    print(f"  Max error (should be < 1e-7): {(h_hat - h_reconstructed).abs().max():.2e}")
    
    # ===== Test 3: 相似度计算 =====
    print("\n" + "-" * 50)
    print("Test 3: Angular Similarity")
    
    # 使用前10个样本测试
    theta_sub = theta[:10]
    sim_matrix = transformer.compute_angular_similarity(theta_sub, theta_sub)
    
    # 对比原始余弦相似度
    cosine_sim = torch.mm(h_hat[:10], h_hat[:10].T)
    
    diff = (sim_matrix - cosine_sim).abs().max()
    print(f"  Similarity matrix shape: {sim_matrix.shape}")
    print(f"  Max difference from cosine sim: {diff:.2e}")
    print(f"  Diagonal values (should be ~1.0): {sim_matrix.diag()[:5].tolist()}")
    
    # ===== Test 4: 测地线距离 =====
    print("\n" + "-" * 50)
    print("Test 4: Geodesic Distance")
    dist_matrix = transformer.compute_angular_distance(theta_sub, theta_sub)
    print(f"  Distance matrix shape: {dist_matrix.shape}")
    print(f"  Distance range: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]")
    print(f"  Diagonal distances (should be ~0): {dist_matrix.diag()[:5].tolist()}")
    
    # ===== Test 5: 维度选择 =====
    print("\n" + "-" * 50)
    print("Test 5: Dimension Selection (Hybrid)")
    theta_reduced, selected_dims = transformer.select_dimensions(
        theta, strategy='hybrid', m_fixed=32, m_adaptive=32
    )
    print(f"  Reduced shape: {theta_reduced.shape}")
    print(f"  Reduction ratio: {theta_reduced.shape[1] / D * 100:.1f}%")
    
    # ===== Test 6: 周期性修复 =====
    print("\n" + "-" * 50)
    print("Test 6: Fix Last Dimension Periodicity")
    
    # 测试NumPy版本
    print("\n  [NumPy Version]")
    test_data_np = np.array([
        [0.5, 0.5, -3.14159],   # 接近-π
        [0.5, 0.5,  3.14159],   # 接近+π (物理上与上面相同)
        [0.5, 0.5,  0.0],       # 0弧度
        [0.5, 0.5,  1.5708],    # π/2
    ])
    
    result_np = fix_last_dim_periodicity(test_data_np)
    print(f"    Input shape:  {test_data_np.shape}")
    print(f"    Output shape: {result_np.shape}")
    
    # 验证边界点连续性
    pt_near_neg_pi = result_np[0, -2:]  # (cos(-π), sin(-π)) ≈ (-1, 0)
    pt_near_pos_pi = result_np[1, -2:]  # (cos(π), sin(π)) ≈ (-1, 0)
    distance_np = np.linalg.norm(pt_near_neg_pi - pt_near_pos_pi)
    print(f"    Distance between ±π points: {distance_np:.6f} (should be ~0)")
    
    # 验证单位圆性质
    for i in range(len(test_data_np)):
        cos_val, sin_val = result_np[i, -2], result_np[i, -1]
        norm = np.sqrt(cos_val**2 + sin_val**2)
        assert abs(norm - 1.0) < 1e-6, f"Unit circle violated: {norm}"
    print("    ✓ All (cos, sin) pairs lie on unit circle")
    
    # 测试PyTorch版本
    print("\n  [PyTorch Version]")
    test_data_torch = torch.tensor(test_data_np)
    result_torch = fix_last_dim_periodicity(test_data_torch)
    print(f"    Input shape:  {test_data_torch.shape}")
    print(f"    Output shape: {result_torch.shape}")
    
    # 验证边界点连续性
    pt_near_neg_pi_torch = result_torch[0, -2:]
    pt_near_pos_pi_torch = result_torch[1, -2:]
    distance_torch = torch.norm(pt_near_neg_pi_torch - pt_near_pos_pi_torch).item()
    print(f"    Distance between ±π points: {distance_torch:.6f} (should be ~0)")
    
    # 验证单位圆性质
    for i in range(test_data_torch.shape[0]):
        cos_val, sin_val = result_torch[i, -2], result_torch[i, -1]
        norm = torch.sqrt(cos_val**2 + sin_val**2).item()
        assert abs(norm - 1.0) < 1e-6, f"Unit circle violated: {norm}"
    print("    ✓ All (cos, sin) pairs lie on unit circle")
    
    # 测试高维数据 (N=10, D-1=511 -> D=512)
    print("\n  [High-Dimensional Test]")
    theta_high_dim = theta[:10]  # 取前10个样本
    result_high_dim = fix_last_dim_periodicity(theta_high_dim)
    print(f"    Input shape:  {theta_high_dim.shape} (D-1=511)")
    print(f"    Output shape: {result_high_dim.shape} (D=512)")
    print(f"    Dimension increase: +{result_high_dim.shape[1] - theta_high_dim.shape[1]}")
    
    print("\n  ✓ Periodicity fix tests passed!")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
