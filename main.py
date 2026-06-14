"""
ArcMIL Main Training Script

完整的训练、验证、测试流程,包含:
- N折交叉验证
- Early Stopping
- Checkpoint管理
- 多阈值评估
- Ensemble测试

使用方法:
    # Camelyon16实验
    python main.py --dataset camelyon16 --kn 6 --kt 6 --gpu 0
    
    # TCGA-NSCLC实验
    python main.py --dataset tcga_nsclc --kn 6 --kt 8 --gpu 1
    
    # MOC实验
    python main.py --dataset moc --kn 4 --kt 4 --gpu 2
    
    # 调参实验
    python main.py --dataset camelyon16 --conf 0.9 --temp 0.15 --lambda_inst 3.0

参考:
    - ArcMIL manuscript experimental pipeline
"""

import os
import sys
import time
import argparse
import shutil  # 新增: 用于原子文件操作
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.loader import get_dataset
from models.hpmil import create_model
from utils.metrics import compute_metrics
from utils.experiment_logger import ExperimentLogger  # 新增!


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ArcMIL Training')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='camelyon16',
                       choices=['camelyon16', 'tcga_nsclc', 'moc'],
                       help='Dataset name')
    parser.add_argument('--kn', type=int, default=None,
                       help='Number of clusters for Class 0')
    parser.add_argument('--kt', type=int, default=None,
                       help='Number of clusters for Class 1')
    
    # 训练超参数
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (default=1 for MIL)')
    
    # 伪标签参数
    parser.add_argument('--conf', type=float, default=None,
                       help='Confidence threshold')
    parser.add_argument('--temp', type=float, default=None,
                       help='Softmax temperature')
    parser.add_argument('--lambda_inst', type=float, default=None,
                       help='Instance loss weight')
    parser.add_argument('--dp', type=float, default=None,
                       help='Dropout rate (e.g., 0.75 for heavy regularization)')
    parser.add_argument('--abnormal_threshold', type=float, default=None,
                       help='Abnormal center detection threshold')
    parser.add_argument('--use_abnormal_centers', action='store_true', default=True,
                       help='Whether to use abnormal center filtering (default: True)')
    parser.add_argument('--no_abnormal_centers', action='store_true',
                       help='Disable abnormal center filtering (for ablation study)')

    # 设备参数
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    # 路径参数
    parser.add_argument('--centers_path', type=str, default=None,
                       help='Path to pre-computed centers (optional)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    
    return parser.parse_args()


def apply_args_to_config(args):
    """将命令行参数应用到Config"""
    if args.dataset:
        Config.set_dataset(args.dataset, args.kn, args.kt)
    
    if args.lr is not None:
        Config.LR = args.lr
    if args.epochs is not None:
        Config.EPOCHS = args.epochs
    if args.conf is not None:
        Config.CONFIDENCE_THRESHOLD = args.conf
    if args.temp is not None:
        Config.TEMPERATURE = args.temp
    if args.lambda_inst is not None:
        Config.LAMBDA_INSTANCE = args.lambda_inst
    if args.dp is not None:
        Config.DROPOUT = args.dp
    if args.abnormal_threshold is not None:
        Config.ABNORMAL_THRESHOLD = args.abnormal_threshold

    # 处理异常中心过滤开关
    if hasattr(args, 'no_abnormal_centers') and args.no_abnormal_centers:
        Config.USE_ABNORMAL_CENTERS = False
    else:
        Config.USE_ABNORMAL_CENTERS = args.use_abnormal_centers

    if args.gpu is not None:
        Config.DEVICE = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    if args.seed is not None:
        Config.SEED = args.seed
    if args.experiment_name is not None:
        Config.EXPERIMENT_NAME = args.experiment_name

    # 重新更新路径（确保checkpoint目录包含完整的超参数信息，避免并发覆盖）
    Config._update_paths()


def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_centers(args) -> tuple:
    """
    加载预计算的聚类中心
    
    Returns:
        centers_class0, centers_class1, abnormal_idx0, abnormal_idx1
    """
    print("\n" + "=" * 70)
    print("Loading Pre-computed Centers")
    print("=" * 70)
    
    # 尝试从指定路径加载
    if args.centers_path and os.path.exists(args.centers_path):
        print(f"Loading from custom path: {args.centers_path}")
        centers_dir = args.centers_path
    else:
        # 从默认路径加载
        centers_dir = Config.CENTERS_DIR
        
        if not os.path.exists(centers_dir):
            raise FileNotFoundError(
                f"Centers directory not found: {centers_dir}\n"
                "Please run compute_centers.py first to generate cluster centers."
            )
    
    class0_dir = os.path.join(centers_dir, Config.CLASS0_NAME)
    class1_dir = os.path.join(centers_dir, Config.CLASS1_NAME)
    
    k0_str = str(Config.VMF_K_NORMAL)
    k1_str = str(Config.VMF_K_TUMOR)
    
    c0_path = os.path.join(class0_dir, f'k={k0_str}', 'centers.npy')
    c1_path = os.path.join(class1_dir, f'k={k1_str}', 'centers.npy')
    
    if not os.path.exists(c0_path):
        raise FileNotFoundError(f"Class 0 centers not found: {c0_path}")
    if not os.path.exists(c1_path):
        raise FileNotFoundError(f"Class 1 centers not found: {c1_path}")
    
    centers_class0 = np.load(c0_path)
    centers_class1 = np.load(c1_path)
    
    print(f"Loaded Class 0 ({Config.CLASS0_NAME}) centers:")
    print(f"  Path: {c0_path}")
    print(f"  Shape: {centers_class0.shape}")
    
    print(f"\nLoaded Class 1 ({Config.CLASS1_NAME}) centers:")
    print(f"  Path: {c1_path}")
    print(f"  Shape: {centers_class1.shape}")
    
    # 加载abnormal indices (如果存在且启用异常中心过滤)
    if Config.USE_ABNORMAL_CENTERS:
        abn_c0_path = os.path.join(class0_dir, f'k={k0_str}', 'abnormal_indices.npy')
        abn_c1_path = os.path.join(class1_dir, f'k={k1_str}', 'abnormal_indices.npy')

        abnormal_idx0 = np.load(abn_c0_path) if os.path.exists(abn_c0_path) else None
        abnormal_idx1 = np.load(abn_c1_path) if os.path.exists(abn_c1_path) else None

        if abnormal_idx0 is not None:
            print(f"\nAbnormal indices loaded:")
            print(f"  Class 0: {len(abnormal_idx0)} / {len(centers_class0)}")
            print(f"  Class 1: {len(abnormal_idx1)} / {len(centers_class1)}")
        else:
            print(f"\nNo abnormal indices found, will compute automatically.")
    else:
        # 消融实验：不使用异常中心过滤
        print(f"\n[ABLATION] Abnormal center filtering DISABLED")
        abnormal_idx0 = None
        abnormal_idx1 = None
    
    return centers_class0, centers_class1, abnormal_idx0, abnormal_idx1


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """
    训练一个epoch
    
    Args:
        model: ArcMIL model
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch编号
    
    Returns:
        metrics_dict: 训练指标字典
    """
    model.train()
    total_loss = 0.0
    total_bag_loss = 0.0
    total_instance_loss = 0.0
    num_batches = 0
    all_bag_labels = []
    all_bag_probs = []
    
    for batch_idx, (features, bag_label, wsi_name) in enumerate(train_loader):
        features = features.squeeze(0).to(device)  # [N, D]
        bag_label = bag_label.to(device)           # [1]
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(features, bag_label=bag_label.item())
        
        # 计算损失
        loss, loss_dict = model.compute_loss(outputs, bag_label)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 累积统计
        total_loss += loss.item()
        total_bag_loss += loss_dict['bag_loss']
        total_instance_loss += loss_dict['instance_loss']
        num_batches += 1
        
        # 收集预测结果
        all_bag_labels.append(bag_label.item())
        all_bag_probs.append(outputs['bag_prob'].squeeze().item())
        
    
    # 计算平均指标
    avg_loss = total_loss / max(num_batches, 1)
    avg_bag_loss = total_bag_loss / max(num_batches, 1)
    avg_instance_loss = total_instance_loss / max(num_batches, 1)
    
    try:
        train_auc = roc_auc_score(all_bag_labels, all_bag_probs)
    except:
        train_auc = 0.0

    # 计算 ACC (使用阈值 0.5)
    train_preds = [1 if p >= 0.5 else 0 for p in all_bag_probs]
    train_acc = sum(1 for t, p in zip(all_bag_labels, train_preds) if t == p) / max(len(all_bag_labels), 1)

    metrics = {
        'avg_loss': avg_loss,
        'avg_bag_loss': avg_bag_loss,
        'avg_instance_loss': avg_instance_loss,
        'train_auc': train_auc,
        'train_acc': train_acc,
        'num_batches': num_batches
    }
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> dict:
    """
    验证模型
    
    Args:
        model: ArcMIL model
        val_loader: 验证数据加载器
        device: 设备
    
    Returns:
        metrics_dict: 验证指标字典
    """
    model.eval()
    total_loss = 0.0
    all_bag_labels = []
    all_bag_probs = []
    
    for features, bag_label, wsi_name in val_loader:
        features = features.squeeze(0).to(device)
        bag_label = bag_label.to(device)
        
        outputs = model(features, bag_label=bag_label.item())
        loss, _ = model.compute_loss(outputs, bag_label)
        
        total_loss += loss.item()
        all_bag_labels.append(bag_label.item())
        all_bag_probs.append(outputs['bag_prob'].squeeze().item())
    
    avg_loss = total_loss / len(val_loader)
    
    try:
        val_auc = roc_auc_score(all_bag_labels, all_bag_probs)
    except:
        val_auc = 0.0

    # 计算 ACC (使用阈值 0.5)
    val_preds = [1 if p >= 0.5 else 0 for p in all_bag_probs]
    val_acc = sum(1 for t, p in zip(all_bag_labels, val_preds) if t == p) / max(len(all_bag_labels), 1)

    metrics = {
        'val_loss': avg_loss,
        'val_auc': val_auc,
        'val_acc': val_acc,
        'labels': all_bag_labels,
        'probs': all_bag_probs
    }
    
    return metrics


@torch.no_grad()
def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    thresholds: list = [0.35, 0.4, 0.45, 0.5]
) -> dict:
    """
    测试模型 (多阈值评估)
    
    Args:
        model: ArcMIL model
        test_loader: 测试数据加载器
        device: 设备
        thresholds: 候选分类阈值列表
    
    Returns:
        results_dict: 测试结果
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_names = []
    
    for features, label, name in test_loader:
        features = features.squeeze(0).to(device)
        label = label.to(device)
        
        outputs = model(features, bag_label=label.item())
        prob = outputs['bag_prob'].squeeze().item()
        
        all_labels.append(label.item())
        all_probs.append(prob)
        all_names.append(name[0] if isinstance(name, (list, tuple)) else name)
    
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    
    # 计算AUC
    auc = roc_auc_score(labels, probs)
    
    # 多阈值评估
    best_metrics = None
    best_f1 = 0.0
    best_thresh = 0.5
    
    threshold_results = {}
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        metrics = compute_metrics(labels, preds, probs)
        threshold_results[thresh] = metrics
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_metrics = metrics
            best_thresh = thresh
    
    results = {
        'auc': auc,
        'best_threshold': best_thresh,
        'best_metrics': best_metrics,
        'threshold_results': threshold_results,
        'labels': labels,
        'probs': probs,
        'names': all_names
    }
    
    return results


def save_checkpoint(model, optimizer, scheduler, fold, epoch, metrics, is_best=False):
    """
    保存完整的模型checkpoint (支持断点续训和实验复现)
    
    包含所有必要信息:
    - 模型权重 (model_state_dict)
    - 优化器状态 (optimizer_state_dict) - 支持断点续训!
    - 调度器状态 (scheduler_state_dict)
    - 完整配置快照 (config) - 支持实验复现!
    - 随机状态 (rng_state) - 保证可重复性!
    - 性能指标 (metrics)
    
    Args:
        model: nn.Module 模型实例
        optimizer: optim.Optimizer 优化器
        scheduler: _LRScheduler 学习率调度器
        fold: int 当前折数
        epoch: int 当前epoch
        metrics: dict 性能指标字典
        is_best: bool 是否是最佳模型
    """
    # 收集完整的配置信息
    config_snapshot = {
        'dataset': Config.DATASET_NAME,
        'kn': Config.VMF_K_NORMAL,
        'kt': Config.VMF_K_TUMOR,
        
        # 模型架构参数
        'in_dim': Config.IN_DIM,
        'hidden_dim': Config.HIDDEN_DIM,
        'dropout': Config.DROPOUT,
        'use_projection': Config.USE_PROJECTION,
        
        # 训练超参数
        'lr': Config.LR,
        'weight_decay': Config.WEIGHT_DECAY,
        'epochs': Config.EPOCHS,
        'batch_size': Config.BATCH_SIZE,
        
        # 伪标签参数
        'lambda_instance': Config.LAMBDA_INSTANCE,
        'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
        'temperature': Config.TEMPERATURE,
        'abnormal_threshold': Config.ABNORMAL_THRESHOLD,
        
        # 其他参数
        'seed': Config.SEED,
        'n_folds': Config.N_FOLDS,
        'device': str(Config.DEVICE),
    }
    
    # 收集随机状态 (保证可重复性)
    rng_state = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    
    # 构建完整checkpoint
    checkpoint = {
        # 元数据
        'meta': {
            'fold': fold,
            'epoch': epoch,
            'timestamp': time.strftime('%Y-%m-%d_%H:%M:%S'),
            'pytorch_version': torch.__version__,
            'git_commit': get_git_hash() if 'get_git_hash' in dir() else None,
        },
        
        # ===== 核心状态 (断点续训必需!) =====
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        
        # ===== 复现实验必需! =====
        'config': config_snapshot,
        'rng_state': rng_state,
        
        # ===== 性能指标 =====
        'metrics': metrics,
    }
    
    # 保存到文件
    path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_fold{fold}_ep{epoch}.pth')
    torch.save(checkpoint, path)
    
    # 如果是最佳模型,额外保存一份
    if is_best:
        best_path = os.path.join(Config.CHECKPOINT_DIR, f'best_model_fold{fold}.pth')
        
        # 原子写入: 先写临时文件,再rename (防止写入中断导致损坏)
        tmp_path = best_path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, best_path)  # 原子操作
        
        print(f"  ✓ Saved: fold{fold}_epoch{epoch}.pth (AUC={metrics.get('val_auc', 0):.4f})")
    
    return path


def load_checkpoint(model, optimizer=None, scheduler=None, fold=0):
    """
    加载完整的模型checkpoint
    
    如果提供了optimizer和scheduler,将恢复其状态以支持断点续训。
    
    Args:
        model: nn.Module 模型实例
        optimizer: optim.Optimizer 可选的优化器 (用于断点续训)
        scheduler: _LRScheduler 可选的调度器
        fold: int 折数
    
    Returns:
        dict: checkpoint信息 (包含epoch/metrics/config等)
    """
    path = os.path.join(Config.CHECKPOINT_DIR, f'best_model_fold{fold}.pth')
    
    if not os.path.exists(path):
        print(f"  ✗ No checkpoint found at {path}")
        return None
    
    # 加载checkpoint
    checkpoint = torch.load(path, map_location=Config.DEVICE)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态 (如果提供)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Restored optimizer state")
    
    # 加载调度器状态 (如果提供)
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  ✓ Restored scheduler state")
    
    # 打印加载信息
    meta = checkpoint.get('meta', {})
    metrics = checkpoint.get('metrics', {})
    
    print(f"  ✓ Loaded checkpoint from Fold {fold}:")
    print(f"    Path: {path}")
    print(f"    Epoch: {meta.get('epoch', 'N/A')}")
    print(f"    Timestamp: {meta.get('timestamp', 'N/A')}")
    print(f"    Val AUC: {metrics.get('val_auc', 'N/A'):.4f}"
          if isinstance(metrics.get('val_auc'), (int, float)) else f"    Val AUC: {metrics.get('val_auc', 'N/A')}")
    print(f"    Val ACC: {metrics.get('val_acc', 'N/A'):.4f}"
          if isinstance(metrics.get('val_acc'), (int, float)) else f"    Val ACC: {metrics.get('val_acc', 'N/A')}")
    print(f"    Val Loss: {metrics.get('val_loss', 'N/A'):.4f}"
          if isinstance(metrics.get('val_loss'), (int, float)) else "")
    
    # 返回完整checkpoint信息供后续使用
    return checkpoint


def get_git_hash():
    """获取当前Git commit hash (用于实验复现)"""
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


def main():
    """
    主函数: 完整的训练-验证-测试流程
    """
    start_time = time.time()
    
    # ===== Step 1: 解析参数 & 配置 =====
    args = parse_args()
    apply_args_to_config(args)
    set_seed(Config.SEED)
    
    # 打印配置
    Config.print_config()
    
    device = torch.device(Config.DEVICE)
    print(f"\nUsing device: {device}")
    
    # ===== Step 2: 加载聚类中心 =====
    # 如果 λ=0 (无实例监督), 不需要加载centers
    if Config.LAMBDA_INSTANCE > 0:
        try:
            centers_c0, centers_c1, abn_idx0, abn_idx1 = load_centers(args)
        except Exception as e:
            print(f"\nError loading centers: {e}")
            print("Please run compute_centers.py first to generate cluster centers.")
            sys.exit(1)
    else:
        print("\n[INFO] λ=0, skipping centers loading (no instance supervision)")
        centers_c0, centers_c1, abn_idx0, abn_idx1 = None, None, None, None
    
    # ===== Step 3: N折交叉验证训练 =====
    print("\n" + "=" * 70)
    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation Training")
    print("=" * 70)
    
    all_fold_results = []
    
    for fold in range(Config.N_FOLDS):
        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/{Config.N_FOLDS}")
        print(f"{'='*70}")
        
        # 创建DataLoaders
        train_dataset = get_dataset(mode='train', fold=fold)
        val_dataset = get_dataset(mode='val', fold=fold)
        test_dataset = get_dataset(mode='test', fold=fold)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        # 创建模型
        model = create_model(Config).to(device)

        # 设置聚类中心到伪标签生成器 (关键: 必须设置到pseudo_label_gen!)
        # 如果 λ=0, 不需要设置centers (无实例监督)
        if Config.LAMBDA_INSTANCE > 0 and centers_c0 is not None:
            model.pseudo_label_gen.centers_class0 = torch.from_numpy(centers_c0).float().to(device)
            model.pseudo_label_gen.centers_class1 = torch.from_numpy(centers_c1).float().to(device)

            # 同时保存到model (用于identify_abnormal_centers等方法)
            model.centers_class0 = model.pseudo_label_gen.centers_class0
            model.centers_class1 = model.pseudo_label_gen.centers_class1

            if abn_idx0 is not None:
                model.abnormal_indices_class0 = torch.from_numpy(abn_idx0).long().to(device)
                model.pseudo_label_gen.abnormal_indices_class0 = model.abnormal_indices_class0  # ✅ 关键!
            else:
                model.abnormal_indices_class0 = None
                model.pseudo_label_gen.abnormal_indices_class0 = None
            if abn_idx1 is not None:
                model.abnormal_indices_class1 = torch.from_numpy(abn_idx1).long().to(device)
                model.pseudo_label_gen.abnormal_indices_class1 = model.abnormal_indices_class1  # ✅ 关键!
            else:
                model.abnormal_indices_class1 = None
                model.pseudo_label_gen.abnormal_indices_class1 = None
        else:
            print("[INFO] Skipping centers setup (λ=0 or no centers provided)")
            model.centers_class0 = None
            model.centers_class1 = None
            model.abnormal_indices_class0 = None
            model.abnormal_indices_class1 = None

        # 手动加载 selected_dims (必须同时设置到pseudo_label_gen!)
        import json as _json
        centers_dir = getattr(Config, 'CENTERS_DIR', None)
        if centers_dir:
            selected_dims_path = os.path.join(centers_dir, 'discriminative_selection', 'selected_dims_top128.json')
            if os.path.exists(selected_dims_path):
                with open(selected_dims_path, 'r') as f:
                    dims_data = _json.load(f)
                    selected_dims = dims_data.get('selected_dimensions', dims_data.get('selected_dims', None))
                    if selected_dims is not None:
                        selected_dims = list(map(int, selected_dims))
                        model.selected_dims = selected_dims
                        model.pseudo_label_gen.selected_dims = selected_dims  # ✅ 关键!
                        print(f"[Main] Loaded {len(selected_dims)} selected dimensions from FDR")

        # 打印centers信息 (仅在λ>0时)
        if Config.LAMBDA_INSTANCE > 0 and model.centers_class0 is not None:
            print(f"\n[PseudoLabel] Centers set manually:")
            print(f"  Class 0 ({Config.CLASS0_NAME}): {len(model.centers_class0)} centers, shape: {model.centers_class0.shape}")
            print(f"  Class 1 ({Config.CLASS1_NAME}): {len(model.centers_class1)} centers, shape: {model.centers_class1.shape}")
        else:
            print(f"\n[PseudoLabel] No centers loaded (λ=0 or no instance supervision)")
        
        # 优化器
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.LR,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Early Stopping变量
        best_val_loss = float('inf')
        best_val_auc = 0.0
        patience_counter = 0
        
        # ===== 初始化日志系统 (每折一个独立的logger) =====
        logger = ExperimentLogger(
            log_dir=Config.LOG_DIR,
            experiment_name=f'fold{fold}_{Config.EXPERIMENT_NAME}'
        )
        logger.log_config(Config)
        
        # 训练循环
        print(f"\nStarting training for {Config.EPOCHS} epochs...")
        
        # 获取当前学习率 (用于日志)
        current_lr = Config.LR
        
        for epoch in range(Config.EPOCHS):
            # Train
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
            
            # Validate
            val_metrics = validate(model, val_loader, device)
            
            # 更新学习率
            scheduler.step(val_metrics['val_loss'])
            
            # 获取当前学习率 (用于日志)
            current_lr = optimizer.param_groups[0]['lr']

            # ===== Epoch 级别日志输出（简洁风格）=====
            print(f"Fold {fold} | Epoch {epoch+1}/{Config.EPOCHS}")
            print(f"  Train Loss: {train_metrics['avg_loss']:.4f} (bag: {train_metrics['avg_bag_loss']:.4f}, ins: {train_metrics['avg_instance_loss']:.4f})")
            print(f"  Train AUC: {train_metrics['train_auc']:.4f} | Acc: {train_metrics.get('train_acc', 0):.4f} | LR: {current_lr:.4f}")
            print(f"  Val   Loss: {val_metrics['val_loss']:.4f} | AUC: {val_metrics['val_auc']:.4f} | Acc: {val_metrics.get('val_acc', 0):.4f}")

            # Checkpoint & Early Stopping
            is_best = val_metrics['val_auc'] > best_val_auc
            
            if is_best:
                best_val_auc = val_metrics['val_auc']
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                # 保存完整Checkpoint (包含optimizer/scheduler/config/rng!)
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    fold=fold,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=True
                )
                
                print(f"  [Save] New Best Val AUC: {best_val_auc:.4f} (Acc: {val_metrics.get('val_acc', 0):.4f})")
            else:
                patience_counter += 1
                
                if patience_counter >= Config.EARLY_STOPPING:
                    print(f"\n  ✗ Early stopping triggered after {patience_counter} epochs "
                          f"(no improvement)")
                    break
        
        # ===== 测试最佳模型 =====
        print(f"\n{'-'*50}")
        print(f"Testing Best Model (Fold {fold+1})")
        print(f"{'-'*50}")
        
        # 加载最佳模型 (恢复完整状态!)
        load_checkpoint(model, optimizer=optimizer, scheduler=scheduler, fold=fold)
        
        # 测试
        test_results = test_model(model, test_loader, device)
        
        # 使用Logger记录详细测试结果
        logger.log_test_results(fold, test_results)
        
        # 打印摘要
        print(f"\nTest Results (Fold {fold+1}):")
        print(f"  AUC: {test_results['auc']:.4f}")
        print(f"  Best Threshold: {test_results['best_threshold']}")
        if test_results['best_metrics']:
            m = test_results['best_metrics']
            print(f"  Accuracy: {m['accuracy']:.4f}")
            print(f"  F1-Score: {m['f1']:.4f}")
            print(f"  Sensitivity: {m['sensitivity']:.4f}")
            print(f"  Specificity: {m['specificity']:.4f}")
        
        all_fold_results.append(test_results)
        
        # 关闭当前折的logger
        logger.log_training_summary(
            epochs=epoch+1,
            best_auc=best_val_auc,
            time_sec=time.time()-start_time,
            early_stopped=(patience_counter >= Config.EARLY_STOPPING)
        )
        logger.close()
    
    # ===== Step 4: Ensemble测试 =====
    print(f"\n{'='*70}")
    print("ENSEMBLE RESULTS (All Folds)")
    print(f"{'='*70}")
    
    # 收集所有折的预测概率
    ensemble_probs = None
    true_labels = None
    
    for fold, results in enumerate(all_fold_results):
        if true_labels is None:
            true_labels = results['labels']
            ensemble_probs = results['probs'].copy()
        else:
            ensemble_probs += results['probs']
    
    # 平均概率
    ensemble_probs /= len(all_fold_results)

    # 计算Ensemble指标
    ensemble_auc = roc_auc_score(true_labels, ensemble_probs)

    # ===== Cross-Validation Summary =====
    print(f"\n>>> Cross-Validation Summary <<<")
    fold_aucs = [r['auc'] for r in all_fold_results]
    fold_accs = [r['best_metrics']['accuracy'] if r.get('best_metrics') else 0 for r in all_fold_results]
    fold_f1s = [r['best_metrics']['f1'] if r.get('best_metrics') else 0 for r in all_fold_results]
    fold_precs = [r['best_metrics']['precision'] if r.get('best_metrics') else 0 for r in all_fold_results]
    fold_recalls = [r['best_metrics']['sensitivity'] if r.get('best_metrics') else 0 for r in all_fold_results]

    import numpy as np
    print(f"AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"ACC: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"F1:  {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    print(f"PRECISION: {np.mean(fold_precs):.4f} +/- {np.std(fold_precs):.4f}")
    print(f"RECALL:    {np.mean(fold_recalls):.4f} +/- {np.std(fold_recalls):.4f}")

    # ===== 各折测试详情 =====
    print(f"\n[STAGE 3] Final Testing...")
    for fold, results in enumerate(all_fold_results):
        best_acc = results['best_metrics']['accuracy'] if results.get('best_metrics') else 0
        print(f"Evaluating Fold {fold} model...")
        print(f"  Fold {fold} Test AUC: {results['auc']:.4f} | Acc: {best_acc:.4f}")

    # ===== Ensemble Result + 多阈值分析表格 =====
    print(f"\n>>> Ensemble Result <<<\n")
    print(f"Threshold Analysis:")
    print(f"{'-'*90}")
    print(f"{'Threshold':<10} {'Acc':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'TN':<5} {'FP':<5} {'FN':<5} {'TP':<5}")
    print(f"{'-'*90}")

    best_ensemble_metrics = None
    best_ensemble_f1 = 0.0
    best_thresh = 0.5

    for thresh in [0.35, 0.4, 0.45, 0.5]:
        preds = (ensemble_probs >= thresh).astype(int)
        metrics = compute_metrics(true_labels, preds, ensemble_probs)

        cm = metrics.get('confusion_matrix', {})
        tn, fp, fn, tp = cm.get('TN', 0), cm.get('FP', 0), cm.get('FN', 0), cm.get('TP', 0)

        print(f"{thresh:<10.2f} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} "
              f"{metrics['precision']:<12.4f} {metrics['sensitivity']:<10.4f} "
              f"{tn:<5} {fp:<5} {fn:<5} {tp:<5}")

        if metrics['f1'] > best_ensemble_f1:
            best_ensemble_f1 = metrics['f1']
            best_ensemble_metrics = metrics
            best_thresh = thresh

    print(f"{'-'*90}\n")
    print(f"Best Threshold: {best_thresh}")
    print(f"  AUC: {ensemble_auc:.4f}")
    print(f"  Acc: {best_ensemble_metrics['accuracy']:.4f}")
    print(f"  F1 : {best_ensemble_metrics['f1']:.4f}")
    print(f"  Precision: {best_ensemble_metrics['precision']:.4f}")
    print(f"  Recall: {best_ensemble_metrics['sensitivity']:.4f}")

    cm = best_ensemble_metrics.get('confusion_matrix', {})
    print(f"  Confusion Matrix: TN={cm.get('TN', 0)}, FP={cm.get('FP', 0)}, FN={cm.get('FN', 0)}, TP={cm.get('TP', 0)}")

    # 总耗时
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Total Training Time: {elapsed_time/60:.2f} minutes")


if __name__ == '__main__':
    main()
