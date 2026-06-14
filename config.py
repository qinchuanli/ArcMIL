"""
ArcMIL Configuration

Configuration file for the current public ArcMIL release
支持多数据集、可扩展的超参数管理
"""

import os


class Config:
    """
    全局配置类
    
    设计模式: 单例 + 类方法动态配置
    支持数据集切换时自动更新所有路径
    """
    
    # ===== 基础路径 =====
    BASE_DIR = '/userhome/home/lijunfei'
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # ===== 数据集路径 (Camelyon16) =====
    CAMELYON16_DIR = os.path.join(BASE_DIR, 'data_features', 'camelyon16_CONCH')
    CAMELYON16_CSV = os.path.join(CAMELYON16_DIR, 'camelyon16_labels.csv')
    
    # ===== 数据集路径 (TCGA-NSCLC) =====
    TCGA_NSCLC_DIR = os.path.join(BASE_DIR, 'data_features', 'TCGA_NSCLC_CONCH')
    TCGA_NSCLC_CSV = os.path.join(TCGA_NSCLC_DIR, 'tcga_nsclc_labels.csv')
    
    # ===== 数据集路径 (MOC) =====
    MOC_DIR = os.path.join(BASE_DIR, 'data_features', 'MOC')
    MOC_TRAIN_DIR = os.path.join(MOC_DIR, 'features_conch_v15')
    
    # ===== 当前数据集配置 (默认) =====
    DATASET_NAME = 'camelyon16'
    
    # ===== 特征维度 =====
    IN_DIM = 512
    
    # ===== 标签映射 =====
    LABEL_MAP = {'Normal': 0, 'Tumor': 1}
    CLASS0_NAME = 'Normal'
    CLASS1_NAME = 'Tumor'
    
    # ===== 是否使用投影层 (MOC需要768->512) =====
    USE_PROJECTION = False
    PROJECTION_DIM = 512
    
    # ===== clustering-related parameters =====
    VMF_K_NORMAL = 6          # 正常类聚类数
    VMF_K_TUMOR = 6           # 肿瘤类聚类数
    
    # ===== 角坐标参数 =====
    USE_ANGULAR_TRANSFORM = True   # 是否使用角坐标变换
    ANGULAR_REDUCE_DIM = None      # 降维后的维度 (None=不降维)
    ANGULAR_SELECT_STRATEGY = 'hybrid'  # 维度选择策略: fixed/variance/cumulative/hybrid
    USE_DECENTERED: bool = True    # 使用去中心化角坐标 (θ' = θ - π/2, Option B: 仅前510维)
    
    # ===== 判别性角选择参数 (Corrected FDR) =====
    USE_DISCRIMINATIVE_SELECTION: bool = True   # 是否启用判别性角选择
    TOP_P: int = 128                          # 保留角维数 (None=不降维)
    DISC_EPSILON: float = 1e-8                 # FDR分母防零常数
    MIN_AUC_RETENTION: float = 0.90             # 最低AUC保留率阈值
    SAVE_DISC_INTERMEDIATE: bool = True         # 是否保存中间结果
    
    # ===== offline center refinement parameters =====
    AFC_MAX_ITERATIONS = 15       # maximum center-refinement iterations
    AFC_TOLERANCE = 1e-6         # convergence tolerance
    AFC_INIT_STEP_SIZE = 0.1     # initial step size
    AFC_REFINE_STEPS = 12        # refinement steps in the released pipeline
    
    # ===== 异常中心检测参数 =====
    ABNORMAL_THRESHOLD = 0.9     # 异常中心判定阈值 (余弦相似度)
    USE_ABNORMAL_CENTERS = True  # 是否使用异常中心过滤 (消融实验可设为False)
    USE_ABNORMAL_ONLY = True     # 是否只使用异常中心生成伪标签
    
    # ===== 模型架构参数 =====
    HIDDEN_DIM = 256             # 隐藏层维度
    NUM_CLASSES = 1              # 分类数 (二分类)
    DROPOUT = 0.5               # Dropout率
    
    # ===== 伪标签参数 =====
    LAMBDA_INSTANCE = 2.0        # 实例损失权重
    CONFIDENCE_THRESHOLD = 0.8   # 置信度阈值
    TEMPERATURE = 0.2            # Softmax温度参数
    
    # ===== 训练超参数 =====
    SEED = 42                   # 随机种子
    N_FOLDS = 5                 # 交叉验证折数
    LR = 2e-4                   # 学习率
    WEIGHT_DECAY = 1e-5         # 权重衰减
    EPOCHS = 50                 # 训练轮数
    EARLY_STOPPING = 10         # 早停耐心值
    BATCH_SIZE = 1              # MIL标准batch_size=1
    
    # ===== 设备配置 =====
    DEVICE = 'cuda:2' if __import__('torch').cuda.is_available() else 'cpu'
    
    # ===== 实验名称 =====
    EXPERIMENT_NAME = 'hpmil_angular'
    
    @classmethod
    def _update_paths(cls):
        """根据当前配置更新输出路径"""

        # 聚类中心目录
        cls.CENTERS_DIR = os.path.join(
            cls.CODE_DIR,
            'centers_angular',
            cls.DATASET_NAME,
            f'k{cls.VMF_K_NORMAL}_{cls.VMF_K_TUMOR}'
        )

        # 实验标识：包含超参数，避免并发时checkpoint互相覆盖
        param_tag = f't{int(cls.TEMPERATURE*100):03d}_c{int(cls.CONFIDENCE_THRESHOLD*100):03d}_dp{int(cls.DROPOUT*100):03d}'

        # Checkpoint目录（含超参数隔离）
        cls.CHECKPOINT_DIR = os.path.join(
            cls.CODE_DIR,
            'checkpoints',
            cls.DATASET_NAME,
            f'kn{cls.VMF_K_NORMAL}_kt{cls.VMF_K_TUMOR}',
            f'{cls.EXPERIMENT_NAME}_{param_tag}'
        )

        # 日志目录（不含超参数，保持原有结构兼容analyze_results）
        cls.LOG_DIR = os.path.join(
            cls.CODE_DIR,
            'logs',
            cls.DATASET_NAME,
            f'kn{cls.VMF_K_NORMAL}_kt{cls.VMF_K_TUMOR}'
        )
        
        # 创建目录
        os.makedirs(cls.CENTERS_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
    
    @classmethod
    def set_dataset(cls, dataset_name: str, kn: int = None, kt: int = None):
        """
        切换数据集并自动更新配置
        
        Args:
            dataset_name: 数据集名称 ('camelyon16', 'tcga_nsclc', 'moc')
            kn: 正常类聚类数 (可选)
            kt: 肿瘤类聚类数 (可选)
        """
        cls.DATASET_NAME = dataset_name
        
        if kn is not None:
            cls.VMF_K_NORMAL = kn
        if kt is not None:
            cls.VMF_K_TUMOR = kt
        
        if dataset_name == 'camelyon16':
            cls.DATA_FEATURE_DIR = cls.CAMELYON16_DIR
            cls.DATA_CSV_PATH = cls.CAMELYON16_CSV
            cls.LABEL_MAP = {'Normal': 0, 'Tumor': 1}
            cls.CLASS0_NAME = 'Normal'
            cls.CLASS1_NAME = 'Tumor'
            cls.IN_DIM = 512
            cls.USE_PROJECTION = False
            
        elif dataset_name == 'tcga_nsclc':
            cls.DATA_FEATURE_DIR = cls.TCGA_NSCLC_DIR
            cls.DATA_CSV_PATH = cls.TCGA_NSCLC_CSV
            cls.LABEL_MAP = {'LUAD': 0, 'LUSC': 1}
            cls.CLASS0_NAME = 'LUAD'
            cls.CLASS1_NAME = 'LUSC'
            cls.IN_DIM = 512
            cls.USE_PROJECTION = False
            
        elif dataset_name == 'moc':
            cls.DATA_FEATURE_DIR = cls.MOC_TRAIN_DIR
            cls.DATA_CSV_PATH = None
            cls.LABEL_MAP = {'Primary': 0, 'Metastatic': 1}
            cls.CLASS0_NAME = 'Primary'
            cls.CLASS1_NAME = 'Metastatic'
            cls.IN_DIM = 768
            cls.USE_PROJECTION = True
            cls.PROJECTION_DIM = 512
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Supported: ['camelyon16', 'tcga_nsclc', 'moc']")
        
        cls._update_paths()
    
    @classmethod
    def print_config(cls):
        """打印当前配置信息（精简版）"""
        print(f"[Config] dataset={cls.DATASET_NAME}, "
              f"kn={cls.VMF_K_NORMAL}, kt={cls.VMF_K_TUMOR}, "
              f"lr={cls.LR}, epochs={cls.EPOCHS}, "
              f"device={cls.DEVICE}, early_stopping={cls.EARLY_STOPPING}")


# 初始化默认配置
Config.set_dataset(Config.DATASET_NAME)

if __name__ == '__main__':
    Config.print_config()
