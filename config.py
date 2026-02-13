# config.py
# 本文件集中管理所有超参数，不包含模型逻辑
import torch

config = {
    # ---------- 数据相关 ----------
    "data": {
        "fmri_path": "./fMri_pcc.npy",               # fMRI PCC 数据路径
        "audio_path": "stft_data_fixed_10/all_stfts_64x640.npy",  # 语音特征路径
        "label_path": "./fMri_label.npy",            # 标签路径
        "n_brain_regions": 90,                        # 脑区数量
        "n_speech_features": 64,                      # 语音特征压缩后维度
        "speech_channels": 64,                         # AudioInfoCollect 输出通道数
        "speech_length": 640,                          # 语音特征时间长度
    },

    # ---------- 训练相关 ----------
    "training": {
        "batch_size": 16,
        "num_epochs": 51,
        "n_splits": 5,                                 # K折交叉验证折数
        "random_seed": 63,
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "lr": 0.0001,                                   # 学习率
        "weight_decay": 0.0005,                         # 权重衰减
    },

    # ---------- 模型结构相关 ----------
    "model": {
        "hidden_size": 128,                             # 隐藏层维度
        "num_classes": 2,                               # 分类数（MDD/HC）

        # ROI掩码模块专用超参数
        "roi_mask": {
            "roi_sparsity": 0.1,                        # 稀疏正则系数
            "smooth_lambda": 0.05,                       # 平滑正则系数
            "lambda_p": 0.5,                             # 间接效应权重
            "damping_factor": 0.85,                      # PageRank阻尼因子
            "grandag_hidden": 64,                         # GraNDAG隐藏层维度
            "grandag_layers": 3,                          # GraNDAG层数
            "grandag_lambda_a": 1.0,                      # GraNDAG拉格朗日乘子
            "grandag_rho_init": 1.0,                      # GraNDAG惩罚参数初值
            "grandag_rho_factor": 2.0,                    # GraNDAG惩罚参数更新倍数
        }
    }
}