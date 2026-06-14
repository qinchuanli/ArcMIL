"""
AFC (Angular Fréchet Center) Clusterer

Offline clustering and center-refinement module used in the current public ArcMIL release,
使用AFC(角坐标Fréchet中心)替代简单算术平均,
确保在超球面原生几何中找到正确的聚类中心。

核心算法:
    E-step: 基于测地线距离的软分配
        r_{ik} ∝ exp(-κ · d_Θ²(θ_i, ω_k))
    
    M-step (两阶段):
        Phase 1 - 粗估计:
            ω_k^(0) = clip((1/|C_k|) Σ θ_i)  简单平均
        
        Phase 2 - AFC精炼:
            ω_k^(t+1) = clip(ω_k^t - η_t · g_t)
            
            梯度 (Theorem 4.1):
                g_t = -2 Σ_i r_{ik} · [d_Θ(θ_i, ω_k)/√(1-Sim_θ²)] · ∇_ω S
            
            步长 (Barzilai-Borwein):
                η_t = <Δg, Δω> / ||Δg||²
    
    收敛准则:
        ||ω^{t+1} - ω^t||₂ < ε  (通常10-15步收敛)

数学基础:
    - Fréchet Mean: min_ω Σ d_Θ²(θ_i, ω)
    - 测地线距离: d_Θ(θ, φ) = arccos(Sim_θ(θ, φ))
    - 梯度公式来自测地线距离的链式法则

参考:
    - current ArcMIL manuscript center-refinement stage
    - Xiao (2026): Spherical Coordinates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict

from .angular_transform import AngularCoordinateTransformer


class AFCClusterer(nn.Module):
    """
    角坐标AFC聚类器 (支持双模式: 全维角坐标 / 降维子空间)
    
    在Θ空间中执行vMF混合模型聚类,
    返回K个AFC精炼的聚类中心。
    
    ⚠️ 重要: 当使用降维子空间 (p < D-1) 时,
    必须设置 use_euclidean_distance=True,
    因为子空间不再是有效的超球面坐标!
    
    Args:
        n_clusters: 聚类数 K
        max_iterations: EM最大迭代次数
        afc_max_iter: AFC最大迭代次数 (每个M-step)
        tolerance: 收敛阈值
        kappa: vMF集中参数 κ (控制分配锐度)
        init_method: 初始化方法 ('kmeans++', 'random', 'uniform')
        refine_steps: AFC精炼步数
        use_bb_step_size: 是否使用BB自适应步长
        use_euclidean_distance: bool (默认False)
            False: 使用角坐标测地线距离 (适用于完整D-1维角坐标)
            True: 使用欧氏距离 (适用于降维后的p维子空间, p << D-1)
    """
    
    def __init__(
        self,
        n_clusters: int = 6,
        max_iterations: int = 20,
        afc_max_iter: int = 15,
        tolerance: float = 1e-6,
        kappa: float = 50.0,
        init_method: str = 'kmeans++',
        refine_steps: int = 12,
        use_bb_step_size: bool = True,
        eps: float = 1e-8,
        use_euclidean_distance: bool = False
    ):
        super().__init__()
        
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.afc_max_iter = afc_max_iter
        self.tolerance = tolerance
        self.kappa = kappa
        self.init_method = init_method
        self.refine_steps = refine_steps
        self.use_bb_step_size = use_bb_step_size
        self.eps = eps
        
        # ⭐ 新增: 测度模式标志
        self.use_euclidean = use_euclidean_distance
        
        # 角坐标变换器 (内部使用float64保证精度)
        # 仅在全维模式下使用
        if not self.use_euclidean:
            self.transformer = AngularCoordinateTransformer(
                eps=eps, 
                use_float64=True
            )
        
        print(f"[AFCClusterer] Initialized with:")
        print(f"  K={n_clusters}, κ={kappa}, max_iter={max_iterations}")
        print(f"  Metric mode: {'EUCLIDEAN (subspace)' if self.use_euclidean else 'ANGULAR (full space)'}")
        
    def _initialize_centers(self, theta: torch.Tensor) -> torch.Tensor:
        """
        初始化聚类中心 (增强版: 支持子空间模式)
        
        Args:
            theta: [N, D] 数据点 (D可以是D-1维角坐标或p维子空间)
        
        Returns:
            centers: [K, D] 初始中心
        """
        N, D = theta.shape
        K = self.n_clusters
        
        # ⭐ 新增: 子空间多样性预检查 (防止kmeans++退化)
        if self.use_euclidean and self.init_method == 'kmeans++':
            sample_size = min(1000, N)
            sample_indices = torch.randperm(N)[:sample_size]
            sample_theta = theta[sample_indices]
            
            # 计算样本间距离统计
            pairwise_dist = torch.cdist(
                sample_theta[::max(1, sample_size//100)], 
                sample_theta[::max(1, sample_size//100)], 
                p=2
            )
            
            mean_dist = pairwise_dist.mean()
            std_dist = pairwise_dist.std()
            
            if std_dist < 1e-6 * (mean_dist + 1e-8):
                print(f"    [Warning] Data points nearly identical!")
                print(f"      Mean distance: {mean_dist:.6f}, Std: {std_dist:.2e}")
                print(f"      [Fallback] Using random initialization with perturbation")
                
                # 回退到带扰动随机初始化
                indices = torch.randperm(N)[:K]
                centers = theta[indices].clone()
                perturbation = torch.randn_like(centers) * 0.01 * (mean_dist + 1e-6)
                centers = centers + perturbation
                
                return centers
        
        if self.init_method == 'random':
            indices = torch.randperm(N)[:K]
            centers = theta[indices].clone()
            
        elif self.init_method == 'kmeans++':
            centers = []
            
            first_idx = torch.randint(N, (1,)).item()
            centers.append(theta[first_idx].clone())
            
            for k in range(1, K):
                current_centers = torch.stack(centers)
                
                # ⭐ 根据模式选择距离计算方式
                if self.use_euclidean:
                    dists = torch.cdist(theta, current_centers, p=2)
                else:
                    dists = self.transformer.compute_angular_distance(theta, current_centers)
                
                min_dists = dists.min(dim=1)[0]
                min_dists = torch.clamp(min_dists, min=0.0)
                
                if torch.isnan(min_dists).any() or torch.isinf(min_dists).any():
                    print(f"    [Warning] Invalid distances found, using uniform sampling")
                    next_idx = torch.randint(0, len(theta), (1,)).item()
                elif min_dists.sum() < 1e-10:
                    print(f"    [Warning] Sum of distances too small, using uniform sampling")
                    next_idx = torch.randint(0, len(theta), (1,)).item()
                else:
                    probs = min_dists / min_dists.sum()
                    next_idx = torch.multinomial(probs.unsqueeze(0), 1).item()
                    
                centers.append(theta[next_idx].clone())
            
            centers = torch.stack(centers)
            
        elif self.init_method == 'uniform':
            centers = torch.zeros(K, D, dtype=theta.dtype, device=theta.device)
            
            for k in range(K):
                if not self.use_euclidean:
                    # 角坐标范围: 前 D-1 维 ∈ [0, π], 最后1维 ∈ [-π, π]
                    centers[k, :D-1] = torch.rand(D-1) * np.pi
                    centers[k, -1] = torch.rand(1) * 2 * np.pi - np.pi
                else:
                    # 子空间: 使用数据范围初始化
                    data_min = theta.min(dim=0)[0]
                    data_max = theta.max(dim=0)[0]
                    centers[k] = data_min + torch.rand(D) * (data_max - data_min)
                    
        else:
            raise ValueError(f"Unknown init method: {self.init_method}")
        
        return centers
    
    def _compute_soft_assignments(
        self,
        theta: torch.Tensor,
        centers: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        E-step: 计算软分配矩阵 (支持双模式)
        
        基于vMF分布的概率密度:
            r_{ik} ∝ exp(κ · Sim_θ(θ_i, ω_k))
              ≈ exp(-κ · d²(θ_i, ω_k))  (小角度近似)
        
        ⭐ 模式选择:
            - use_euclidean=False: 使用角坐标测地线距离 d_Θ
            - use_euclidean=True: 使用欧氏距离 ||θ - ω||₂
        
        Args:
            theta: [N, D] 数据点
            centers: [K, D] 聚类中心
        
        Returns:
            responsibilities: [N, K] 软分配矩阵 (每行和为1)
            distances: [N, K] 距离矩阵
        """
        N = theta.shape[0]
        K = centers.shape[0]
        
        # ⭐ 根据模式选择距离计算方式
        if self.use_euclidean:
            # 欧氏距离模式 (R^p空间)
            distances = torch.cdist(theta, centers, p=2)  # [N, K]
            
            # 数值验证
            assert not torch.isnan(distances).any(), "NaN in Euclidean distances"
            assert not torch.isinf(distances).any(), "Inf in Euclidean distances"
            assert distances.min() >= -1e-6, f"Negative distance: {distances.min():.6f}"
            
        else:
            # 角坐标测地线距离模式 (Θ空间)
            distances = self.transformer.compute_angular_distance(theta, centers)
        
        # vMF似然度: log p(θ | ω_k) ∝ κ·Sim_θ(θ, ω_k)
        # 使用负距离平方作为能量 (等价于高斯近似)
        log_responsibilities = -self.kappa * distances.pow(2)
        
        # Log-sum-exp归一化 (数值稳定)
        max_log_resp = log_responsibilities.max(dim=1, keepdim=True)[0]
        log_responsibilities_shifted = log_responsibilities - max_log_resp
        
        responsibilities = torch.exp(log_responsibilities_shifted)
        responsibilities = responsibilities / (responsibilities.sum(dim=1, keepdim=True) + self.eps)
        
        return responsibilities, distances
    
    def _afc_refine_center(
        self,
        theta: torch.Tensor,
        weights: torch.Tensor,
        center_init: torch.Tensor
    ) -> torch.Tensor:
        """
        M-step: AFC精炼单个聚类中心
        
        两阶段法:
        Phase 1: 简单平均初始化
        Phase 2: 梯度下降精炼 (BB步长)
        
        Args:
            theta: [N, D-1] 所有数据点
            weights: [N] 该中心的权重 (r_{ik})
            center_init: [D-1] 初始中心
        
        Returns:
            omega: [D-1] AFC精炼后的中心
        """
        D_minus_1 = theta.shape[-1]
        
        # ===== Phase 1: 粗估计 =====
        # 加权算术平均
        weight_sum = weights.sum() + self.eps
        omega = (weights.unsqueeze(-1) * theta).sum(dim=0) / weight_sum
        
        # 投影到合法范围
        omega = self._clip_to_valid_range(omega)
        
        # ===== Phase 2: AFC梯度下降精炼 (使用解析梯度!) =====
        prev_gradient_norm = None
        prev_omega = None
        step_size = 0.1  # 初始步长
        
        for t in range(self.afc_max_iter):
            # 计算当前中心到所有点的相似度和距离
            sim_vec = self.transformer.compute_angular_similarity(theta, omega.unsqueeze(0))  # [N, 1]
            dist_vec = self.transformer.compute_angular_distance(theta, omega.unsqueeze(0))   # [N, 1]
            
            # clamp避免数值问题
            sim_clamped = torch.clamp(sim_vec.squeeze(), -(1 - self.eps), (1 - self.eps))
            
            # ===== 计算解析梯度 (Theorem 4.1 完整形式) =====
            # 目标: min_ω Σ w_i · d_Θ²(θ_i, ω) = min_ω Σ w_i · arccos²(S(θ_i, ω))
            # 
            # 梯度公式:
            # g = -2 Σ_i w_i · [d_Θ(θ_i, ω) / √(1-S²(θ_i, ω))] · ∇_ω S(θ_i, ω)
            #
            # 其中 ∇_ω S(θ, ω) 是角坐标相似度的解析梯度:
            #   ∂S/∂ω₁ = -sin(ω₁)·cos(θ₁) 
            #             - Σ_{k=2}^{D-2} [sin(ω₁)·(∏_{l=2}^{k-1} sin ω_l)·(∏_{l=1}^{k-1} sin θ_l)]·cos(θ_k-ω_k)
            #   
            #   ∂S/∂ω_j (j≥2) = -(∏_{l=1}^{j-1} sin ω_l)·(∏_{l=1}^{j-1} sin θ_l)·sin(θ_j-ω_j)
            
            # 计算每个样本的解析梯度 ∇_ω S(θ_i, ω)  [N, D-1]
            grad_sim = self._compute_similarity_gradient_analytical(theta, omega)
            
            # 安全因子: d / √(1-s²)  [N]
            safe_factor = dist_vec.squeeze() / torch.sqrt(
                torch.clamp(1.0 - sim_clamped.pow(2), min=self.eps)
            )
            
            # 完整梯度: g = -2 Σ w_i · factor_i · grad_sim_i
            gradient = -2.0 * (
                weights.unsqueeze(-1) * safe_factor.unsqueeze(-1) * grad_sim
            ).sum(dim=0)  # [D-1]
            
            gradient_norm = gradient.norm()
            
            if gradient_norm < self.tolerance:
                break
            
            # Barzilai-Borwein自适应步长
            if self.use_bb_step_size and t > 0 and prev_gradient_norm is not None and prev_omega is not None:
                delta_omega = omega - prev_omega
                delta_grad = gradient - prev_gradient_norm * (gradient / gradient_norm)  # 近似
                
                dot_gg = delta_grad.dot(delta_grad)
                if dot_gg > self.eps:
                    bb_step = abs(delta_omega.dot(delta_grad)) / dot_gg
                    # 限制步长范围
                    step_size = max(min(bb_step, 1.0), 1e-5)
            
            # 更新中心
            new_omega = omega - step_size * gradient / (gradient_norm + self.eps)
            
            # 投影到合法范围
            new_omega = self._clip_to_valid_range(new_omega)
            
            # 检查收敛
            change = (new_omega - omega).norm()
            
            # 保存历史
            prev_omega = omega.clone()
            prev_gradient_norm = gradient_norm
            omega = new_omega
            
            if change < self.tolerance:
                break
        
        return omega
    
    def _compute_similarity_gradient_analytical(
        self, 
        theta: torch.Tensor, 
        omega: torch.Tensor
    ) -> torch.Tensor:
        """
        计算角坐标相似度 S(θ, ω) 对 ω 的解析梯度 ∇_ω S
        
        基于论文 Theorem 4.1 和角坐标相似度的完整公式:
        
        S(θ, ω) = cos(θ₁)cos(ω₁) + Σ_{k=1}^{D-2} [(∏_{l=0}^{k-1} sin θ_l sin ω_l)] · cos(θ_k - ω_k)
        
        解析梯度公式:
            ∂S/∂ω₁ = -sin(ω₁)·cos(θ₁)
                      - Σ_{k=2}^{D-2} [sin(ω₁)·(∏_{l=2}^{k-1} sin ω_l)·(∏_{l=1}^{k-1} sin θ_l)]·cos(θ_k-ω_k)
            
            ∂S/∂ω_j (j≥2) = -(∏_{l=1}^{j-1} sin ω_l)·(∏_{l=1}^{j-1} sin θ_l)·sin(θ_j - ω_j)
        
        Args:
            theta: [N, D-1] 数据点角坐标
            omega: [D-1] 中心角坐标
        
        Returns:
            grad: [N, D-1] 每个样本的解析梯度
        """
        N = theta.shape[0]
        D_minus_1 = theta.shape[1]
        device = theta.device
        dtype = theta.dtype
        
        # 初始化梯度矩阵
        grad = torch.zeros(N, D_minus_1, device=device, dtype=dtype)
        
        # ===== 预计算连乘积 (提高效率) =====
        # sin_prod_theta[k] = ∏_{l=1}^{k} sin(θ_l), 长度 D-2 (索引1到D-2)
        # sin_prod_omega[k] = ∏_{l=1}^{k} sin(ω_l), 长度 D-2
        
        sin_theta = torch.sin(theta[:, 1:D_minus_1])   # [N, D-2] (跳过第一个维度)
        sin_omega = torch.sin(omega[1:D_minus_1])       # [D-2]
        
        # 计算累积乘积
        cumprod_sin_theta = torch.ones(N, D_minus_1, device=device, dtype=dtype)
        cumprod_sin_omega = torch.ones(D_minus_1, device=device, dtype=dtype)
        
        for k in range(D_minus_1 - 1):  # k = 0, 1, ..., D-3 (对应维度1到D-2)
            if k == 0:
                cumprod_sin_theta[:, k+1] = sin_theta[:, k]
                cumprod_sin_omega[k+1] = sin_omega[k]
            else:
                cumprod_sin_theta[:, k+1] = cumprod_sin_theta[:, k] * sin_theta[:, k]
                cumprod_sin_omega[k+1] = cumprod_sin_omega[k] * sin_omega[k]
        
        # ===== 计算 ∂S/∂ω₁ =====
        # 第一项: -sin(ω₁)·cos(θ₁)
        grad[:, 0] = -torch.sin(omega[0]) * torch.cos(theta[:, 0])
        
        # 第二项: -Σ_{k=2}^{D-2} [sin(ω₁)·(∏_{l=2}^{k-1} sin ω_l)·(∏_{l=1}^{k-1} sin θ_l)]·cos(θ_k-ω_k)
        for k in range(1, D_minus_1 - 1):  # k = 1, ..., D-3 (对应角坐标的第2到第D-2个维度)
            if k >= 2:
                omega_factor = torch.sin(omega[0]) * cumprod_sin_omega[k-1]  # sin(ω₁) * ∏_{l=2}^{k-1} sin(ω_l)
            else:
                omega_factor = torch.sin(omega[0])  # 当k=1时, 空积为1
            
            theta_factor = cumprod_sin_theta[:, k-1]  # ∏_{l=1}^{k-1} sin(θ_l)
            
            diff = theta[:, k] - omega[k]  # θ_k - ω_k
            
            grad[:, 0] -= (omega_factor * theta_factor * torch.cos(diff))
        
        # ===== 计算 ∂S/∂ω_j (j ≥ 2) =====
        for j in range(1, D_minus_1):
            if j < D_minus_1 - 1:
                # 对于前 D-2 个维度 (j = 1, ..., D-3)
                # ∂S/∂ω_j = -(∏_{l=1}^{j-1} sin ω_l)·(∏_{l=1}^{j-1} sin θ_l)·sin(θ_j - ω_j)
                
                omega_cumprod = cumprod_sin_omega[j-1]      # ∏_{l=1}^{j-1} sin(ω_l)
                theta_cumprod = cumprod_sin_theta[:, j-1]    # ∏_{l=1}^{j-1} sin(θ_l)
                diff = theta[:, j] - omega[j]                 # θ_j - ω_j
                
                grad[:, j] = -omega_cumprod * theta_cumprod * torch.sin(diff)
                
            else:
                # 对于最后一个维度 (j = D-2, 即θ_{D-1})
                # S的最后一项不包含ω_{D-1}, 所以梯度为0
                # (因为最后一项是 cos(θ_{D-1} - ω_{D-1}) 对 ω_{D-1}求导是 sin(...),
                # 但前面的连乘积已经包含了所有维度)
                # 实际上, 根据完整公式, 最后一项确实有贡献
                omega_cumprod = cumprod_sin_omega[j-1]
                theta_cumprod = cumprod_sin_theta[:, j-1]
                diff = theta[:, j] - omega[j]
                
                grad[:, j] = -omega_cumprod * theta_cumprod * torch.sin(diff)
        
        return grad
    
    def _clip_to_valid_range(self, theta: torch.Tensor) -> torch.Tensor:
        """
        将角坐标投影到合法范围
        
        前 D-2 个维度: [ε, π-ε]
        最后一个维度: [-π+ε, π-ε]
        
        Args:
            theta: [D-1] 角坐标向量
        
        Returns:
            clipped_theta: [D-1] 合法范围的角坐标
        """
        D_minus_1 = theta.shape[-1]
        
        result = theta.clone()
        
        # 前 D-2 维: [ε, π-ε]
        if D_minus_1 > 1:
            result[..., :D_minus_1-1] = torch.clamp(result[..., :D_minus_1-1], 
                                                     self.eps, np.pi - self.eps)
        
        # 最后一维: [-π+ε, π-ε]
        result[..., -1] = torch.clamp(result[..., -1],
                                      -(np.pi - self.eps),
                                       np.pi - self.eps)
        
        return result
    
    def fit(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        执行完整的AFC聚类
        
        Args:
            theta: [N, D-1] 角坐标数据 (已转换好的)
        
        Returns:
            dict {
                'centers': [K, D-1] 聚类中心,
                'labels': [N] 硬标签 (最近中心),
                'responsibilities': [N, K] 软分配矩阵,
                'inertia': float 总惯性 (加权和平方距离),
                'n_iterations': int 实际迭代次数
            }
        """
        print(f"\n[AFC Clustering] Starting clustering with K={self.n_clusters}")
        print(f"  Data shape: {theta.shape}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  AFC refine steps: {self.refine_steps}")
        
        N, D_minus_1 = theta.shape
        K = self.n_clusters
        
        # 初始化中心
        centers = self._initialize_centers(theta)
        
        print(f"  Initialization method: {self.init_method}")
        
        # EM迭代
        prev_inertia = float('inf')
        
        for iteration in range(self.max_iterations):
            # ===== E-step: 软分配 =====
            responsibilities, distances = self._compute_soft_assignments(theta, centers)
            
            # 计算惯性 (目标函数值)
            inertia = (responsibilities * distances.pow(2)).sum().item()
            
            # 打印进度
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"  Iter {iteration+1:3d}: Inertia={inertia:.4f}")
            
            # 收敛检查
            change = abs(prev_inertia - inertia)
            if change < self.tolerance and iteration > 0:
                print(f"  Converged at iteration {iteration+1}")
                break
            
            prev_inertia = inertia
            
            # ===== M-step: 更新每个中心 =====
            new_centers = torch.zeros_like(centers)

            for k in range(K):
                weights = responsibilities[:, k]

                if self.use_euclidean:
                    # 欧式模式: 使用简单加权平均 (不需要AFC精炼)
                    weight_sum = weights.sum() + self.eps
                    new_centers[k] = (weights.unsqueeze(-1) * theta).sum(dim=0) / weight_sum
                else:
                    # 角坐标模式: 使用AFC精炼
                    new_centers[k] = self._afc_refine_center(
                        theta,
                        weights,
                        centers[k]
                    )
            
            centers = new_centers
        
        # 最终E-step获取标签
        final_responsibilities, final_distances = self._compute_soft_assignments(theta, centers)
        labels = final_responsibilities.argmax(dim=1)
        final_inertia = (final_responsibilities * final_distances.pow(2)).sum().item()
        
        print(f"\n[AFC Clustering] Completed!")
        print(f"  Final inertia: {final_inertia:.4f}")
        print(f"  Total iterations: {iteration + 1}")
        
        # 统计每个簇的大小
        unique_labels, counts = labels.unique(return_counts=True)
        print(f"  Cluster sizes: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        
        return {
            'centers': centers,
            'labels': labels,
            'responsibilities': final_responsibilities,
            'distances': final_distances,
            'inertia': final_inertia,
            'n_iterations': iteration + 1
        }
    
    def predict(self, theta: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        预测新样本的聚类标签
        
        Args:
            theta: [N, D-1] 新数据点
            centers: [K, D-1] 已学习的中心
        
        Returns:
            labels: [N] 预测标签
        """
        _, distances = self._compute_soft_assignments(theta, centers)
        labels = distances.argmin(dim=1)
        return labels


if __name__ == '__main__':
    print("=" * 70)
    print("Testing AFC Clusterer")
    print("=" * 70)
    
    # 创建测试数据
    transformer = AngularCoordinateTransformer(use_float64=True)
    
    # 生成模拟数据: 3个簇
    N_per_cluster = 100
    D = 512
    
    h_data = []
    true_labels = []
    
    for cluster_id in range(3):
        # 创建簇中心
        center = torch.randn(D)
        center = center / center.norm()
        
        # 围绕中心生成点 (加入噪声)
        for i in range(N_per_cluster):
            noise = torch.randn(D) * 0.1
            point = center + noise
            point = point / point.norm()
            h_data.append(point)
            true_labels.append(cluster_id)
    
    h_data = torch.stack(h_data)
    true_labels = torch.tensor(true_labels)
    
    print(f"\n[Test Data]")
    print(f"  Shape: {h_data.shape}")
    print(f"  Clusters: 3")
    print(f"  Points per cluster: {N_per_cluster}")
    
    # 转换到角坐标
    theta_data = transformer.cartesian_to_angular(h_data)
    print(f"\n[Transformed to Angular Space]")
    print(f"  Theta shape: {theta_data.shape}")
    
    # 执行AFC聚类
    clusterer = AFCClusterer(
        n_clusters=3,
        max_iterations=30,
        afc_max_iter=15,
        kappa=50.0,
        init_method='kmeans++'
    )
    
    results = clusterer.fit(theta_data)
    
    # 评估聚类质量
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    ari = adjusted_rand_score(true_labels.numpy(), results['labels'].numpy())
    nmi = normalized_mutual_info_score(true_labels.numpy(), results['labels'].numpy())
    
    print(f"\n[Clustering Quality]")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
    
    print("\n" + "=" * 70)
