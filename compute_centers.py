#!/usr/bin/env python3
"""
compute_centers.py - Offline center computation for the current ArcMIL release

严格保证所有特征在L2归一化的超球面S^{D-1}上(||h||₂=1),
然后转换到角坐标空间Θ进行AFC聚类。

完整流程:
    1. 加载训练集特征 (按类别分开,仅用Normal/Primary样本计算各自类别的中心)
    2. L2归一化 → 确保径向r=1 (单位超球面)
    3. 角坐标变换 T: S^{D-1} → Θ ⊂ R^{D-1}
    4. 按类别独立执行AFC聚类 (vMF混合模型EM算法)
    5. Abnormal Center Detection (异常中心筛选)
    6. 完整保存:
       - centers.npy: [K, D-1] 角坐标空间的AFC精炼中心
       - abnormal_indices.npy: 异常中心索引
       - metadata.json: 完整元数据(参数/质量统计/校验和)
       - checksum.md5: 文件完整性校验

使用方法:
    # Camelyon16实验
    python compute_centers.py --dataset camelyon16 --kn 6 --kt 6 --gpu 0
    
    # TCGA-NSCLC实验
    python compute_centers.py --dataset tcga_nsclc --kn 6 --kt 8 --gpu 1
    
    # MOC实验
    python compute_centers.py --dataset moc --kn 4 --kt 4 --gpu 2

数学基础:
    - 输入特征 h ∈ R^D
    - L2归一化: ĥ = h / ||h||₂ ∈ S^{D-1}  (径向r=1)
    - 角坐标变换: θ = T(ĥ) ∈ Θ ⊂ R^{D-1}
    - AFC聚类: min_ω Σ d_Θ²(θ_i, ω)  (测地线距离, 非欧氏距离)

参考:
    - current ArcMIL manuscript-aligned center computation stage
"""

import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np
import torch
from datetime import datetime

# 添加项目根目录和code目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.loader import get_dataset
from models.angular_transform import AngularCoordinateTransformer
from models.afc_clusterer import AFCClusterer
from scripts.discriminative_angle_selector import run_discriminative_selection


def get_git_hash():
    """获取当前Git commit hash"""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return "unknown"


def load_and_normalize_features(dataset, device):
    """
    加载特征并严格L2归一化到单位超球面
    
    关键保证:
    - ||h_i||₂ = 1 for all i (径向r=1)
    - 所有点在S^{D-1}上
    
    Args:
        dataset: PyTorch Dataset
        device: torch.device
    
    Returns:
        h_hat_list: list of [N_i, D] 归一化后的特征张量列表
    """
    print(f"  Loading and normalizing features...")
    
    h_hat_list = []
    total_samples = 0
    
    for idx in range(len(dataset)):
        features, label, name = dataset[idx]
        
        # 转为float32并移到设备
        features = features.float().to(device)
        
        # ===== 关键步骤: L2归一化确保径向=1 =====
        h_hat = torch.nn.functional.normalize(features, p=2, dim=-1)
        
        # 验证归一化 (数值精度检查)
        norms = h_hat.norm(dim=-1)
        max_deviation = (norms - 1.0).abs().max().item()
        
        if max_deviation > 1e-5:
            print(f"    ⚠ Warning: {name} normalization deviation={max_deviation:.2e}")
        
        h_hat_list.append(h_hat.cpu())  # 移回CPU节省显存
        total_samples += h_hat.shape[0]
        
        if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
            print(f"    Processed {idx+1}/{len(dataset)} samples | "
                  f"Total features: {total_samples}")
    
    print(f"  ✓ Loaded {total_samples} feature vectors (all ||h||₂=1)")
    
    return h_hat_list


def transform_to_angular_space(h_hat_list, transformer, device, args, batch_size=1000):
    """
    将笛卡尔坐标批量转换为角坐标 (支持双方案 + 周期性修复)
    
    Args:
        h_hat_list: list of [N_i, D] L2归一化特征
        transformer: AngularCoordinateTransformer实例
        device: torch.device
        args: 命令行参数 (包含use_decentered_angular, fix_periodicity)
        batch_size: int 批次大小
    
    Returns:
        theta_final: [N_total, D-1] 或 [N_total, D] 所有样本的角坐标
            如果 fix_periodicity=True，维度从 D-1 变为 D（最后一维展开为 cos/sin）
        metric_mode: str 度量模式 ('cosine' 或 'angular')
    """
    print(f"\n  Transforming to angular space Θ...")
    
    theta_list = []
    
    for i, h_hat in enumerate(h_hat_list):
        # 分批处理避免OOM
        n_samples = h_hat.shape[0]
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = h_hat[start:end].to(device)
            
            # 转换到角坐标空间
            theta_batch = transformer.cartesian_to_angular(batch)
            
            theta_list.append(theta_batch.cpu())
            
            if (i == 0 and start == 0) or ((start // batch_size) % 10 == 0):
                print(f"    Batch [{start}:{end}/{n_samples}] done")
    
    # 合并所有批次
    theta_all = torch.cat(theta_list, dim=0)  # [N_total, D-1]
    original_dim = theta_all.shape[1]
    
    # ⭐ 步骤1: 应用周期性修复（在去中心化之前）
    if args.fix_periodicity:
        from models.angular_transform import fix_last_dim_periodicity
        
        print(f"\n  [Periodicity Fix]")
        print(f"    Status: ENABLED")
        print(f"    Original dims: {original_dim}")
        
        # 转换为numpy进行处理
        if isinstance(theta_all, torch.Tensor):
            theta_np = theta_all.cpu().numpy()
            theta_fixed = fix_last_dim_periodicity(theta_np)
            theta_all = torch.from_numpy(theta_fixed).to(device)
        else:
            theta_all = fix_last_dim_periodicity(theta_all)
        
        print(f"    Fixed dims: {theta_all.shape[1]}")
        print(f"    Method: Sin/Cos encoding for θ_{original_dim-1}")
    
    # ⭐ 步骤2: 双方案切换（去中心化）- 使用选项B
    if args.use_decentered_angular:
        # 🔧 修复: 大数据集时将去中心化操作移到CPU以避免CUDA OOM
        theta_all = theta_all.cpu()  # 移到CPU
        pi_over_2 = torch.tensor(np.pi / 2, dtype=theta_all.dtype)

        if args.fix_periodicity:
            # ⭐ 选项B: 仅对前510维去中心化，cos/sin列保持不变
            # 理由: cos/sin编码具有单位圆性质，不应被平移破坏
            theta_all[:, :-2] = theta_all[:, :-2] - pi_over_2
            mode_name = "DECENTERED (θ' = θ - π/2, except cos/sin cols)"
            metric_mode = 'cosine'
        else:
            # 原始行为：无修复时对所有维度去中心化
            theta_all = theta_all - pi_over_2
            mode_name = "DECENTERED (θ' = θ - π/2)"
            metric_mode = 'cosine'

        theta_all = theta_all.to(device)  # 移回GPU (小数据集)
    else:
        # 方案A: 原始角坐标模式（使用修复后的相似度公式）
        theta_final = theta_all
        mode_name = "ORIGINAL ANGULAR (fixed formula)"
        metric_mode = 'angular'
    
    theta_final = theta_all

    # 🔧 智能内存管理: 大数据集保留在CPU，只在需要时移到GPU
    LARGE_DATASET_THRESHOLD = 200000  # 超过20万样本视为大数据集
    if theta_final.shape[0] > LARGE_DATASET_THRESHOLD:
        print(f"\n  [Memory Optimization]")
        print(f"    Dataset size: {theta_final.shape[0]:,} samples (> {LARGE_DATASET_THRESHOLD:,} threshold)")
        print(f"    Strategy: Keep on CPU for preprocessing, move to GPU only for clustering")
        # 保持theta_final在CPU上（已经是cpu tensor如果走了decentering路径）
        # 如果没走decenter但数据量大，强制移到cpu
        if theta_final.device.type != 'cpu':
            theta_final = theta_final.cpu()
            print(f"    Action: Moved large tensor to CPU to avoid CUDA OOM")
    else:
        # 小数据集：确保在GPU上
        if theta_final.device.type != 'cuda':
            theta_final = theta_final.to(device)
            print(f"\n  [Memory] Small dataset: Kept on GPU ({device})")
    
    # 输出最终状态
    if not args.fix_periodicity:
        print(f"\n  [Periodicity Fix]")
        print(f"    Status: DISABLED (using original {theta_final.shape[1]}-dim angular coords)")
    
    print(f"  ✓ Angular mode: {mode_name}")
    print(f"  ✓ Metric mode: {metric_mode}")
    print(f"  ✓ Final shape: {theta_final.shape}")
    
    return theta_final, metric_mode


def perform_kmeans_clustering(theta_data, n_clusters, class_name, args, use_reduced=False):
    """
    使用标准KMeans进行聚类 (与欧式坐标版本保持一致)
    
    这是基于文献证据的修复方案:
    - AFC设计用于单簇精炼，不适合直接做多簇聚类
    - 标准KMeans在归一化特征上效果良好 (参考vMF-ABMIL-pseudolabel)
    
    Args:
        theta_data: [N, D] 数据 (torch.Tensor或numpy数组)
        n_clusters: int 聚类数K
        class_name: str 类别名称
        args: 命令行参数
        use_reduced: bool 是否为降维子空间模式 (仅用于日志)
    
    Returns:
        dict {
            'centers': [K, D] KMeans中心,
            'labels': [N] 簇标签,
            'inertia': float 惯性,
            'cluster_sizes': dict 各簇大小,
            'metric_mode': str 'kmeans',
            'method': str 'kmeans'
        }
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    
    print(f"\n{'='*60}")
    print(f" KMeans Clustering: {class_name} (K={n_clusters})")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 转换为numpy
    if isinstance(theta_data, torch.Tensor):
        theta_np = theta_data.cpu().numpy()
    else:
        theta_np = theta_data.copy()
    
    print(f"  Data shape: {theta_np.shape}")
    print(f"  Metric mode: {'EUCLIDEAN (subspace)' if use_reduced else 'ANGULAR (full space)'}")
    print(f"  Method: Standard sklearn.KMeans (n_init=10)")
    
    # L2归一化 (与欧式坐标版本一致)
    print(f"  L2 normalizing data...")
    theta_normalized = normalize(theta_np, norm='l2', axis=1)
    
    # 标准KMeans (与vMF-ABMIL-pseudolabel/models/vmf.py一致)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,  # 多次初始化避免局部最优
        max_iter=300,
        tol=1e-4
    )
    
    kmeans.fit(theta_normalized)
    
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    
    # 计算簇大小统计
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
    
    elapsed = time.time() - start_time
    
    print(f"  ✓ Clustering completed in {elapsed:.2f}s")
    print(f"  Inertia: {inertia:.4f}")
    print(f"  Cluster sizes: {cluster_sizes}")
    print(f"  Centers shape: {centers.shape}")
    
    return {
        'centers': torch.from_numpy(centers).float(),
        'labels': labels,
        'inertia': inertia,
        'cluster_sizes': cluster_sizes,
        'metric_mode': 'kmeans',
        'method': 'kmeans',
        'time': elapsed
    }


def perform_afc_clustering(theta_data, n_clusters, class_name, args, use_reduced=False):
    """
    对单个类别执行完整的AFC聚类 (支持双模式)
    
    流程:
    1. K-means++初始化
    2. EM迭代 (E-step + M-step with AFC refinement)
    3. 返回聚类结果
    
    ⭐ 模式选择:
        - use_reduced=False: 使用角坐标测地线距离 (完整D-1维角空间)
        - use_reduced=True: 使用欧氏距离 (降维后的p维子空间)
    
    Args:
        theta_data: [N, D] 数据 (D可以是D-1或p)
        n_clusters: int 聚类数K
        class_name: str 类别名称 (用于日志)
        args: 命令行参数
        use_reduced: bool 是否为降维子空间模式
    
    Returns:
        dict {
            'centers': [K, D] AFC精炼的中心,
            'labels': [N] 硬标签,
            'inertia': float 总惯性,
            'n_iterations': int 迭代次数,
            'cluster_sizes': dict 各簇大小,
            'metric_mode': str ('euclidean' 或 'angular')
        }
    """
    print(f"\n{'='*60}")
    print(f" AFC Clustering: {class_name} (K={n_clusters})")
    print(f"{'='*60}")
    print(f"  Data shape: {theta_data.shape}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  AFC refine steps: {args.afc_iter}")
    print(f"  κ (concentration): {args.kappa}")
    print(f"  Metric mode: {'EUCLIDEAN (subspace)' if use_reduced else 'ANGULAR (full space)'}")
    
    # 初始化AFC聚类器 (⭐ 传入use_euclidean_distance参数)
    clusterer = AFCClusterer(
        n_clusters=n_clusters,
        max_iterations=args.max_iter,
        afc_max_iter=args.afc_iter,
        tolerance=1e-6,
        kappa=args.kappa,
        init_method='kmeans++',
        use_bb_step_size=True,
        eps=1e-8,
        use_euclidean_distance=use_reduced  # ⭐ 关键参数!
    )
    
    # 执行聚类
    results = clusterer.fit(theta_data)
    
    # 打印结果摘要
    print(f"\n  Results:")
    print(f"    Final inertia: {results['inertia']:.6f}")
    print(f"    Converged at iteration: {results['n_iterations']}")
    print(f"    Centers shape: {results['centers'].shape}")
    
    unique_labels, counts = results['labels'].unique(return_counts=True)
    cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist()))
    print(f"    Cluster sizes: {cluster_sizes}")
    
    # 计算平均簇内距离 (质量指标) - ⭐ 根据模式选择距离度量
    avg_intra_dist = 0.0
    for k in range(n_clusters):
        mask = results['labels'] == k
        if mask.sum() > 0:
            cluster_points = theta_data[mask]
            center = results['centers'][k]
            
            if use_reduced:
                # 欧氏距离模式
                dists = torch.norm(cluster_points - center.unsqueeze(0), dim=1).mean()
            else:
                # 角坐标测地线距离模式
                dists = clusterer.transformer.compute_angular_distance(
                    cluster_points, center.unsqueeze(0)
                ).mean()
            
            avg_intra_dist += dists.item()
    avg_intra_dist /= n_clusters
    print(f"    Avg intra-cluster distance: {avg_intra_dist:.6f}")
    
    # 添加额外统计到results
    results['avg_intra_distance'] = avg_intra_dist
    results['cluster_sizes_dict'] = cluster_sizes
    results['metric_mode'] = 'euclidean' if use_reduced else 'angular'  # ⭐ 新增
    
    return results


def detect_abnormal_centers(centers_c0, centers_c1, transformer, threshold, use_reduced=False, metric_mode='cosine'):
    """
    检测异常中心 (支持双方案)

    Args:
        centers_c0: [K0, p] 或 [K0, D-1] Class 0的中心（角坐标或子空间）
        centers_c1: [K1, p] 或 [K1, D-1] Class 1的中心
        transformer: AngularCoordinateTransformer 实例
        threshold: 相似度阈值 τ_abn
        use_reduced: 是否为降维子空间模式
        metric_mode: str 度量模式 ('cosine' 或 'angular')

    Returns:
        abnormal_idx_c0: 异常中心索引列表 (Class 0)
        abnormal_idx_c1: 异常中心索引列表 (Class 1)
        cross_sim: [K0, K1] 跨类相似度矩阵
    """
    import torch.nn.functional as F

    print("\n" + "=" * 60)
    print(" Abnormal Center Detection")
    print("=" * 60)
    print(f"  Threshold (τ_abn): {threshold}")
    print(f"  Class 0 centers: {len(centers_c0)}")
    print(f"  Class 1 centers: {len(centers_c1)}")
    print(f"  Metric mode: {metric_mode}")

    if use_reduced or metric_mode == 'cosine':
        # ===== 模式1: 降维子空间 或 去中心化角坐标 (使用余弦相似度) =====
        print(f"\n  ✅ Using COSINE SIMILARITY in R^{centers_c0.shape[1]} space")

        c0_tensor = torch.from_numpy(centers_c0).float()
        c1_tensor = torch.from_numpy(centers_c1).float()

        # 标准余弦相似度 (值域 [-1, 1])
        cross_sim = F.cosine_similarity(
            c0_tensor.unsqueeze(1),
            c1_tensor.unsqueeze(0),
            dim=2
        ).numpy()

        assert cross_sim.min() >= -1.0 and cross_sim.max() <= 1.0 + 1e-6, \
            f"Cosine similarity out of range: [{cross_sim.min():.4f}, {cross_sim.max():.4f}]"

    else:
        # ===== 模式2: 原始角坐标空间 (Θ ⊂ R^{D-1}) 使用修复后的角坐标相似度 =====
        # ⭐ 关键修复: 先反投影到笛卡尔空间，再用标准余弦相似度!
        print(f"\n  ✅ Using PROJECTED COSINE SIMILARITY (Θ → S^{D-1} → cosine)")

        c0_tensor = torch.from_numpy(centers_c0).float()
        c1_tensor = torch.from_numpy(centers_c1).float()

        # 步骤1: 反投影角坐标 → 笛卡尔坐标 (回到超球面)
        with torch.no_grad():
            cart_c0 = transformer.angular_to_cartesian(c0_tensor)  # [K0, D]
            cart_c1 = transformer.angular_to_cartesian(c1_tensor)  # [K1, D]

            # 确保L2归一化 (应在单位超球面上)
            cart_c0 = F.normalize(cart_c0, p=2, dim=1)
            cart_c1 = F.normalize(cart_c1, p=2, dim=1)

        # 步骤2: 标准余弦相似度 (= 笛卡尔内积，因为已归一化)
        cross_sim = torch.matmul(cart_c0, cart_c1.T).numpy()  # [K0, K1]

        # 验证范围
        assert cross_sim.min() >= -1.0 - 1e-6 and cross_sim.max() <= 1.0 + 1e-6, \
            f"Projected cosine similarity out of range: [{cross_sim.min():.4f}, {cross_sim.max():.4f}]"

    # 输出统计信息
    print(f"  Cross-similarity matrix shape: {cross_sim.shape}")
    print(f"  Similarity range: [{cross_sim.min():.4f}, {cross_sim.max():.4f}]")
    print(f"  Expected range: [-1.0000, 1.0000]")

    # 检测异常中心
    abnormal_idx_c0 = []
    abnormal_idx_c1 = []

    # Class 0: 检查每个中心对Class 1的最大相似度
    max_sim_c0 = cross_sim.max(axis=1)  # [K0]
    for idx in range(len(centers_c0)):
        if max_sim_c0[idx] < threshold:
            abnormal_idx_c0.append(idx)

    # Class 1: 检查每个中心对Class 0的最大相似度
    max_sim_c1 = cross_sim.max(axis=0)  # [K1]
    for idx in range(len(centers_c1)):
        if max_sim_c1[idx] < threshold:
            abnormal_idx_c1.append(idx)

    # 输出结果
    print(f"\n  Class 0 ({'Normal' if True else 'Tumor'}):")
    print(f"    Abnormal: {len(abnormal_idx_c0)} / {len(centers_c0)}")
    if len(abnormal_idx_c0) > 0:
        print(f"    Indices: {abnormal_idx_c0}")
        print(f"    Max similarities: {[f'{max_sim_c0[i]:.4f}' for i in abnormal_idx_c0]}")

    print(f"\n  Class 1 ({'Tumor' if True else 'Normal'}):")
    print(f"    Abnormal: {len(abnormal_idx_c1)} / {len(centers_c1)}")
    if len(abnormal_idx_c1) > 0:
        print(f"    Indices: {abnormal_idx_c1}")
        print(f"    Max similarities: {[f'{max_sim_c1[i]:.4f}' for i in abnormal_idx_c1]}")

    if len(abnormal_idx_c0) == 0 and len(abnormal_idx_c1) == 0:
        print(f"\n  ⚠️ No abnormal centers detected!")
        print(f"     All inter-class similarities ≥ {threshold}")
        print(f"     Consider lowering threshold or checking data quality")

    return abnormal_idx_c0, abnormal_idx_c1, cross_sim


def save_class_centers(
    centers, 
    abnormal_indices, 
    cross_sim_row,
    clustering_results,
    class_id,
    class_name,
    n_clusters,
    output_dir,
    args
):
    """
    保存单个类别的聚类中心 (带完整元数据)
    
    保存内容:
    1. centers.npy - [K, D-1] 角坐标空间的AFC中心
    2. abnormal_indices.npy - 异常中心索引数组
    3. metadata.json - 完整元数据字典
    4. checksum.md5 - MD5校验和
    
    Args:
        centers: [K, D-1] numpy array 聚类中心
        abnormal_indices: numpy array 异常中心索引
        cross_sim_row: [K1] 该类各中心与另一类的最大相似度
        clustering_results: dict perform_afc_clustering()返回的结果
        class_id: int 类别ID (0或1)
        class_name: str 类别名称
        n_clusters: int 聚类数K
        output_dir: str 输出目录路径
        args: 命令行参数
    """
    # 创建子目录
    class_dir = os.path.join(output_dir, class_name, f'k={n_clusters}')
    os.makedirs(class_dir, exist_ok=True)
    
    # ===== 1. 保存聚类中心 (NumPy格式) =====
    centers_path = os.path.join(class_dir, 'centers.npy')
    np.save(centers_path, centers)
    print(f"  ✓ Saved centers: {centers_path}")
    print(f"    Shape: {centers.shape}, Dtype: {centers.dtype}")
    
    # ===== 2. 保存异常中心索引 =====
    abn_path = os.path.join(class_dir, 'abnormal_indices.npy')
    np.save(abn_path, abnormal_indices)
    print(f"  ✓ Saved abnormal indices: {abn_path}")
    print(f"    Count: {len(abnormal_indices)} / {len(centers)}")
    
    # ===== 3. 生成并保存元数据 (JSON格式) =====
    metadata = {
        # 基本信息
        "dataset": Config.DATASET_NAME,
        "class": {
            "id": int(class_id),
            "name": class_name,
        },
        "clustering": {
            "n_clusters": int(n_clusters),
            "coordinate_type": "angular",  # 明确标注: 角坐标空间!
            "original_dim": int(Config.IN_DIM),  # 原始笛卡尔维度D
            "angular_dim": int(centers.shape[1]),  # 角坐标维度D-1
            "shape": list(centers.shape),
            "dtype": str(centers.dtype),
        },
        
        # 聚类参数 (完全可复现!)
        "parameters": {
            "method": "AFC (Angular Fréchet Center)",
            "max_iterations": int(args.max_iter),
            "afc_max_iterations": int(args.afc_iter),
            "kappa": float(args.kappa),
            "init_method": "kmeans++",
            "tolerance": 1e-6,
            "use_bb_step_size": True,
            "abnormal_threshold": float(args.abnormal_threshold),
        },
        
        # 聚类质量指标
        "quality_metrics": {
            "inertia": float(clustering_results['inertia']),
            "n_iterations": int(clustering_results.get('n_iterations', 0)),  # KMeans可能没有此字段
            "avg_intra_cluster_distance": float(clustering_results.get('avg_intra_distance', 0)),
            "cluster_sizes": clustering_results.get('cluster_sizes_dict', clustering_results.get('cluster_sizes', {})),
        },
        
        # 异常中心检测结果
        "abnormal_detection": {
            "threshold": float(args.abnormal_threshold),
            "n_abnormal": int(len(abnormal_indices)),
            "abnormal_indices": abnormal_indices if isinstance(abnormal_indices, list) else abnormal_indices.tolist(),
            "max_similarity_to_other_class": (
                cross_sim_row[abnormal_indices].round(6).tolist() 
                if len(abnormal_indices) > 0 else []
            ),
            "is_abnormal": [bool(i in set(abnormal_indices)) 
                           for i in range(len(centers))],
        },
        
        # 数据统计
        "data_statistics": {
            "n_samples_used": int(clustering_results.get('n_samples', 0)),
            "feature_dim_original": int(Config.IN_DIM),
            "feature_dim_angular": int(centers.shape[1]),
            "normalization": "L2 (||h||₂=1 on S^{D-1})",  # 强调径向=1!
        },
        
        # 时间戳 & 版本控制
        "timestamp": datetime.now().isoformat(),
        "computed_by": "compute_centers.py v1.0",
        "environment": {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": get_git_hash(),
        }
    }
    
    metadata_path = os.path.join(class_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved metadata: {metadata_path}")
    
    # ===== 4. 计算并保存MD5校验和 =====
    md5_hash = hashlib.md5(np.load(centers_path).tobytes()).hexdigest()
    checksum_path = os.path.join(class_dir, 'checksum.md5')
    with open(checksum_path, 'w') as f:
        f.write(f"{md5_hash}  centers.npy\n")
        f.write(f"# Generated by compute_centers.py\n")
        f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
    print(f"  ✓ Saved checksum: {checksum_path} (MD5={md5_hash[:12]}...)")
    
    return metadata


def main():
    """主函数: 完整的中心计算流程"""
    start_time = time.time()
    
    # ===== Step 0: 参数解析 & 配置 =====
    parser = argparse.ArgumentParser(
        description='Compute AFC clustering centers on unit hypersphere S^{D-1}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['camelyon16', 'tcga_nsclc', 'moc'],
                       help='Dataset name')
    parser.add_argument('--kn', type=int, default=6,
                       help='Number of clusters for Class 0 (Normal/Primary/LUAD)')
    parser.add_argument('--kt', type=int, default=6,
                       help='Number of clusters for Class 1 (Tumor/Metastatic/LUSC)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--max_iter', type=int, default=30,
                       help='Max EM iterations')
    parser.add_argument('--afc_iter', type=int, default=15,
                       help='Max AFC refinement iterations per M-step')
    parser.add_argument('--kappa', type=float, default=50.0,
                       help='vMF concentration parameter κ')
    parser.add_argument('--abnormal_threshold', type=float, default=0.9,
                       help='Abnormal center detection threshold τ_abn')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for transformation (avoid OOM)')
    parser.add_argument('--use_disc_selection', action='store_true',
                       help='Enable discriminative angle selection (Corrected FDR)')
    parser.add_argument('--no_disc_selection', action='store_true',
                       help='Disable discriminative angle selection')
    parser.add_argument('--top_p', type=int, default=None,
                       help='Number of dimensions to keep after selection (default: from Config)')
    parser.add_argument('--use_decentered_angular', action='store_true', default=True,
                       help='Use decentered angular coordinates (theta\' = theta - pi/2) [default: True]')
    parser.add_argument('--no-decentered', action='store_true', default=False,
                       help='Use original angular coordinates (disable decentering)')
    parser.add_argument('--fix-periodicity', action='store_true', default=True,
                       help='Fix last dimension periodicity with Sin/Cos encoding [default: True]')
    parser.add_argument('--no-fix-periodicity', action='store_true', default=False,
                       help='Disable periodicity fix (use original 511-dim angular coords)')
    
    args = parser.parse_args()
    
    # 处理去中心化参数: --no-decentered 会覆盖为False
    if args.no_decentered:
        args.use_decentered_angular = False
    
    # 处理周期性修复参数: --no-fix-periodicity 会覆盖为False
    if args.no_fix_periodicity:
        args.fix_periodicity = False
    
    # 应用配置
    Config.set_dataset(args.dataset, args.kn, args.kt)
    
    # 命令行参数覆盖Config (如果提供)
    if args.use_disc_selection:
        Config.USE_DISCRIMINATIVE_SELECTION = True
    elif args.no_disc_selection:
        Config.USE_DISCRIMINATIVE_SELECTION = False
    
    if args.top_p is not None:
        Config.TOP_P = args.top_p
    
    # 设备配置
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    
    # 打印配置信息
    print("\n" + "=" * 70)
    print(" AFC Clustering Center Computation")
    print(" (Strictly on Unit Hypersphere S^{D-1}, r=1)")
    print("=" * 70)
    print(f"\n[Configuration]")
    print(f"  Dataset: {Config.DATASET_NAME}")
    print(f"  Classes: {Config.CLASS0_NAME}(0) vs {Config.CLASS1_NAME}(1)")
    print(f"  K_{Config.CLASS0_NAME}: {Config.VMF_K_NORMAL}")
    print(f"  K_{Config.CLASS1_NAME}: {Config.VMF_K_TUMOR}")
    print(f"  Feature dim (Cartesian): {Config.IN_DIM}")
    print(f"  Feature dim (Angular): {Config.IN_DIM - 1}")
    print(f"  Device: {device}")
    print(f"  Output dir: {Config.CENTERS_DIR}")
    print(f"  κ (concentration): {args.kappa}")
    print(f"  τ_abn (threshold): {args.abnormal_threshold}")
    decentered_label = "θ'=θ-π/2 (cosine)"
    original_label = "original θ (angular)"
    print(f"  Use decentered angular: {args.use_decentered_angular} "
          f"({decentered_label if args.use_decentered_angular else original_label})")
    print(f"  Fix periodicity: {args.fix_periodicity}")
    if args.fix_periodicity:
        print(f"    → Angular dims: {Config.IN_DIM - 1} → {Config.IN_DIM} (Sin/Cos for last dim)")
    else:
        print(f"    → Angular dims: {Config.IN_DIM - 1} (original)")
    
    if Config.USE_DISCRIMINATIVE_SELECTION and Config.TOP_P is not None:
        print(f"\n[Discriminative Angle Selection]")
        print(f"  ✓ ENABLED: Corrected FDR selection")
        print(f"  Original dimension: D-1 = {Config.IN_DIM - 1}")
        print(f"  Target dimension: top-p = {Config.TOP_P}")
        print(f"  Epsilon (ε): {Config.DISC_EPSILON}")
        print(f"  Min AUC retention: {Config.MIN_AUC_RETENTION:.0%}")
    else:
        print(f"\n[Discriminative Angle Selection]")
        print(f"  ✗ DISABLED: Using full-dimensional space ({Config.IN_DIM - 1} dims)")
    
    # 初始化角坐标变换器 (float64保证精度)
    transformer = AngularCoordinateTransformer(eps=1e-8, use_float64=True)
    
    # ===== Step 1: 加载Class 0数据 =====
    print(f"\n{'='*70}")
    print(f"[Step 1] Loading {Config.CLASS0_NAME} (Class 0) training data...")
    print(f"{'='*70}")
    
    dataset_c0 = get_dataset(mode='normal_only')
    print(f"  Found {len(dataset_c0)} samples")
    
    # 加载并L2归一化 (保证径向=1!)
    h_hat_list_c0 = load_and_normalize_features(dataset_c0, device)
    
    # 转换到角坐标空间
    theta_c0, metric_mode = transform_to_angular_space(h_hat_list_c0, transformer, device, args)
    
    # 更新统计信息
    clustering_results_c0 = {'n_samples': theta_c0.shape[0]}
    
    # ===== Step 2: 加载Class 1数据 =====
    print(f"\n{'='*70}")
    print(f"[Step 2] Loading {Config.CLASS1_NAME} (Class 1) training data...")
    print(f"{'='*70}")

    dataset_c1 = get_dataset(mode='tumor_only')
    print(f"  Found {len(dataset_c1)} samples")
    
    # 加载并L2归一化
    h_hat_list_c1 = load_and_normalize_features(dataset_c1, device)
    
    # 转换到角坐标空间
    theta_c1, _ = transform_to_angular_space(h_hat_list_c1, transformer, device, args)
    
    clustering_results_c1 = {'n_samples': theta_c1.shape[0]}
    
    # ===== Step 2.5: 判别性角选择 (Corrected FDR) ⭐ =====
    if Config.USE_DISCRIMINATIVE_SELECTION and Config.TOP_P is not None:
        print(f"\n{'='*70}")
        print(f"[Step 2.5] Discriminative Angle Selection (Corrected FDR)")
        print(f"{'='*70}")
        print(f"  Original dimension: D-1 = {theta_c0.shape[1]}")
        print(f"  Target dimension: top-p = {Config.TOP_P}")
        
        disc_start_time = time.time()
        
        # 合并两类数据用于选角
        all_theta = torch.cat([theta_c0, theta_c1], dim=0).cpu().numpy()  # [N_total, D-1]
        all_labels = np.concatenate([
            np.zeros(theta_c0.shape[0]),
            np.ones(theta_c1.shape[0])
        ])
        
        print(f"  Total samples for selection: {all_theta.shape[0]}")
        print(f"    Class 0: {theta_c0.shape[0]}")
        print(f"    Class 1: {theta_c1.shape[0]}")
        
        # 运行判别性选角算法
        selection_result = run_discriminative_selection(
            theta=all_theta,
            labels=all_labels,
            top_p=Config.TOP_P,
            output_dir=Config.CENTERS_DIR,
            save_intermediate=Config.SAVE_DISC_INTERMEDIATE,
            epsilon=Config.DISC_EPSILON,
            verbose=True
        )
        
        # 提取选中的维度索引
        selected_dims = selection_result['selected_dims']
        
        # 应用降维
        print(f"\n  Applying dimensionality reduction: {theta_c0.shape[1]} → {Config.TOP_P}")
        theta_c0 = torch.from_numpy(theta_c0.cpu().numpy()[:, selected_dims])  # [N0, p]
        theta_c1 = torch.from_numpy(theta_c1.cpu().numpy()[:, selected_dims])  # [N1, p]

        # 🔧 修复: FDR降维后释放原始大张量 (512维→128维，节省75%内存)
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  [Memory] Released original tensors after FDR reduction (512→128 dims)")
        
        # 设置降维模式标志
        use_reduced = (Config.TOP_P < all_theta.shape[1])  # 当 p < D-1 时为True
        print(f"\n  [Metric Mode] Reduced subspace mode: {use_reduced} (p={Config.TOP_P}, D-1={all_theta.shape[1]})")
        
        disc_elapsed = time.time() - disc_start_time
        
        # 验证结果
        validation = selection_result['validation']
        print(f"\n  [Selection Quality Report]")
        print(f"    Recommendation: {validation['recommendation']}")
        print(f"    AUC Retention: {validation['auc_retention_ratio']:.2%}")
        print(f"    Dimension Distribution:")
        print(f"      Low index: {validation['dim_distribution']['low']} ({validation['dim_distribution']['low']/Config.TOP_P*100:.1f}%)")
        print(f"      Mid index: {validation['dim_distribution']['mid']} ({validation['dim_distribution']['mid']/Config.TOP_P*100:.1f}%)")
        print(f"      High index: {validation['dim_distribution']['high']} ({validation['dim_distribution']['high']/Config.TOP_P*100:.1f}%)")
        print(f"    Selection time: {disc_elapsed:.2f}s")
        
        if validation['recommendation'] == 'FAIL':
            print(f"\n  ⚠ WARNING: Selection quality check failed!")
            print(f"  Consider adjusting TOP_P or MIN_AUC_RETENTION")
        
        # 更新统计信息
        clustering_results_c0['disc_selection'] = {
            'used': True,
            'original_dim': int(all_theta.shape[1]),
            'reduced_dim': int(Config.TOP_P),
            'selected_dims': selected_dims.tolist(),
            'auc_retention': float(validation['auc_retention_ratio']),
            'selection_time': float(disc_elapsed)
        }
        clustering_results_c1['disc_selection'] = clustering_results_c0['disc_selection']
    
    else:
        print(f"\n[Info] Discriminative angle selection disabled (using full-dimensional space)")
        clustering_results_c0['disc_selection'] = {'used': False}
        clustering_results_c1['disc_selection'] = {'used': False}
        use_reduced = False  # 全维模式：不使用降维度量
    
    # ===== Step 3: KMeans聚类 Class 0 =====
    print(f"\n{'='*70}")
    print(f"[Step 3] KMeans Clustering for {Config.CLASS0_NAME}")
    print(f"{'='*70}")
    
    results_c0 = perform_kmeans_clustering(
        theta_c0.to(device), 
        Config.VMF_K_NORMAL, 
        Config.CLASS0_NAME, 
        args,
        use_reduced=use_reduced
    )
    clustering_results_c0.update(results_c0)
    
    # ===== Step 4: KMeans聚类 Class 1 =====
    print(f"\n{'='*70}")
    print(f"[Step 4] KMeans Clustering for {Config.CLASS1_NAME}")
    print(f"{'='*70}")
    
    results_c1 = perform_kmeans_clustering(
        theta_c1.to(device), 
        Config.VMF_K_TUMOR, 
        Config.CLASS1_NAME, 
        args,
        use_reduced=use_reduced
    )
    clustering_results_c1.update(results_c1)
    
    # ===== Step 5: 异常中心检测 =====
    abnormal_idx_c0, abnormal_idx_c1, cross_sim = detect_abnormal_centers(
        results_c0['centers'].cpu().numpy(),
        results_c1['centers'].cpu().numpy(),
        transformer,
        args.abnormal_threshold,
        use_reduced=use_reduced,
        metric_mode=metric_mode  # ⭐ 新增参数!
    )
    
    # ===== Step 6: 保存结果 (带完整元数据) =====
    print(f"\n{'='*70}")
    print(f"[Step 6] Saving all results with metadata...")
    print(f"{'='*70}")
    
    # 保存Class 0
    # cross_sim 可能是 numpy array (reduced mode) 或 torch tensor (full mode)
    if hasattr(cross_sim, 'numpy'):
        cross_sim_row = cross_sim.max(axis=1).numpy()
        cross_sim_col = cross_sim.max(axis=0).numpy()
    else:
        cross_sim_row = cross_sim.max(axis=1)
        cross_sim_col = cross_sim.max(axis=0)

    metadata_c0 = save_class_centers(
        centers=results_c0['centers'].cpu().numpy(),
        abnormal_indices=abnormal_idx_c0,
        cross_sim_row=cross_sim_row,  # Class 0各中心对Class 1的最大相似度
        clustering_results=clustering_results_c0,
        class_id=0,
        class_name=Config.CLASS0_NAME,
        n_clusters=Config.VMF_K_NORMAL,
        output_dir=Config.CENTERS_DIR,
        args=args
    )

    # 保存Class 1
    metadata_c1 = save_class_centers(
        centers=results_c1['centers'].cpu().numpy(),
        abnormal_indices=abnormal_idx_c1,
        cross_sim_row=cross_sim_col,  # Class 1各中心对Class 0的最大相似度
        clustering_results=clustering_results_c1,
        class_id=1,
        class_name=Config.CLASS1_NAME,
        n_clusters=Config.VMF_K_TUMOR,
        output_dir=Config.CENTERS_DIR,
        args=args
    )
    
    # ===== Step 7: 最终总结 =====
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✓ AFC CENTER COMPUTATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    
    print(f"\n[Output Summary]")
    print(f"  Directory: {Config.CENTERS_DIR}")
    print(f"\n  {Config.CLASS0_NAME} (Class 0):")
    print(f"    Centers: {Config.CENTERS_DIR}/{Config.CLASS0_NAME}/k={Config.VMF_K_NORMAL}/centers.npy")
    print(f"    Shape: {results_c0['centers'].shape}")
    print(f"    Abnormal: {len(abnormal_idx_c0)} / {Config.VMF_K_NORMAL}")
    print(f"    Inertia: {results_c0['inertia']:.6f}")
    
    print(f"\n  {Config.CLASS1_NAME} (Class 1):")
    print(f"    Centers: {Config.CENTERS_DIR}/{Config.CLASS1_NAME}/k={Config.VMF_K_TUMOR}/centers.npy")
    print(f"    Shape: {results_c1['centers'].shape}")
    print(f"    Abnormal: {len(abnormal_idx_c1)} / {Config.VMF_K_TUMOR}")
    print(f"    Inertia: {results_c1['inertia']:.6f}")
    
    print(f"\n[Statistics]")
    print(f"  Total computation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} min)")
    print(f"  Class 0 samples used: {theta_c0.shape[0]}")
    print(f"  Class 1 samples used: {theta_c1.shape[0]}")
    print(f"  Feature dimension: {theta_c0.shape[1]} (angular space)")
    
    if clustering_results_c0.get('disc_selection', {}).get('used', False):
        disc_info = clustering_results_c0['disc_selection']
        print(f"\n[Discriminative Angle Selection Results]")
        print(f"  ✓ Selection applied: {disc_info['original_dim']} → {disc_info['reduced_dim']} dims")
        print(f"  AUC retention: {disc_info['auc_retention']:.2%}")
        print(f"  Selection time: {disc_info['selection_time']:.2f}s")

    D = args.feat_dim_cartesian if hasattr(args, 'feat_dim_cartesian') else theta_c0.shape[1] + 1
    print(f"  All features normalized: ||h||₂ = 1 (unit hypersphere S^{D-1})")
    
    print(f"\n[Next Steps]")
    print(f"  Run training:")
    print(f"    python main.py --dataset {args.dataset} --kn {args.kn} --kt {args.kt} --gpu {args.gpu}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
