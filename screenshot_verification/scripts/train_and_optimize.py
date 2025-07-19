#!/usr/bin/env python3
"""
模型训练与优化脚本
包括超参数调优、模型选择、训练监控和性能优化
"""
import os
import sys
import argparse
import logging
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import optuna
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import joblib

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_models import ModelTrainer, ScreenshotDataset, get_transforms
from core.model_enhancements import (
    create_vision_transformer, create_dual_stream_network,
    create_adversarial_detector, create_model_ensemble
)
from config.settings import settings


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.study = None
    
    def optimize_hyperparameters(self, model_type: str, n_trials: int = 50):
        """优化超参数"""
        self.logger.info(f"开始超参数优化: {model_type}, 试验次数: {n_trials}")
        
        # 创建Optuna研究
        study_name = f"{model_type}_optimization"
        storage = f"sqlite:///{self.output_dir}/optuna_study.db"
        
        self.study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        
        # 定义目标函数
        def objective(trial):
            return self._objective_function(trial, model_type)
        
        # 运行优化
        self.study.optimize(objective, n_trials=n_trials)
        
        # 保存最佳参数
        self._save_best_params(model_type)
        
        # 生成优化报告
        self._generate_optimization_report()
        
        self.logger.info("超参数优化完成")
        return self.study.best_params
    
    def _objective_function(self, trial, model_type: str) -> float:
        """目标函数：返回验证准确率"""
        # 定义超参数搜索空间
        params = self._define_search_space(trial, model_type)
        
        try:
            # 创建训练配置
            config = self._create_training_config(model_type, params)
            
            # 运行训练
            trainer = ModelTrainer(config)
            
            # 使用交叉验证
            cv_score = self._cross_validation_score(trainer, config, n_folds=3)
            
            return cv_score
            
        except Exception as e:
            self.logger.error(f"试验失败: {e}")
            return 0.0
    
    def _define_search_space(self, trial, model_type: str) -> Dict:
        """定义超参数搜索空间"""
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        }
        
        if model_type == 'vision_transformer':
            params.update({
                'patch_size': trial.suggest_categorical('patch_size', [8, 16, 32]),
                'embed_dim': trial.suggest_categorical('embed_dim', [512, 768, 1024]),
                'depth': trial.suggest_int('depth', 6, 12),
                'num_heads': trial.suggest_categorical('num_heads', [8, 12, 16]),
            })
        elif model_type == 'dual_stream':
            params.update({
                'fusion_dropout': trial.suggest_float('fusion_dropout', 0.3, 0.7),
                'attention_heads': trial.suggest_categorical('attention_heads', [4, 8, 16]),
            })
        
        return params
    
    def _create_training_config(self, model_type: str, params: Dict) -> Dict:
        """创建训练配置"""
        config = {
            'model_type': model_type,
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir / f"trial_{time.time()}"),
            'num_epochs': 20,  # 快速试验
            'num_workers': 4,
            'img_size': 224,
            **params
        }
        
        return config
    
    def _cross_validation_score(self, trainer, config: Dict, n_folds: int = 3) -> float:
        """交叉验证评分"""
        # 加载数据
        transform = get_transforms(img_size=config.get('img_size', 224), is_training=True)
        dataset = ScreenshotDataset(config['data_dir'], transform=transform)
        
        # 准备标签
        labels = []
        for _, label in dataset:
            labels.append(label)
        
        # 交叉验证
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            try:
                # 创建数据加载器
                train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
                val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
                
                train_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=config['batch_size'], sampler=train_sampler
                )
                val_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=config['batch_size'], sampler=val_sampler
                )
                
                # 训练一个epoch
                trainer.train_epoch(train_loader, 1)
                val_metrics = trainer.validate_epoch(val_loader, 1)
                
                scores.append(val_metrics['accuracy'])
                
            except Exception as e:
                self.logger.error(f"Fold {fold} 失败: {e}")
                scores.append(0.0)
        
        return np.mean(scores)
    
    def _save_best_params(self, model_type: str):
        """保存最佳参数"""
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        result = {
            'model_type': model_type,
            'best_params': best_params,
            'best_value': best_value,
            'optimization_history': [
                {'trial': trial.number, 'value': trial.value, 'params': trial.params}
                for trial in self.study.trials
            ]
        }
        
        with open(self.output_dir / f"{model_type}_best_params.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        self.logger.info(f"最佳参数已保存: {best_params}")
        self.logger.info(f"最佳准确率: {best_value:.4f}")
    
    def _generate_optimization_report(self):
        """生成优化报告"""
        # 优化历史
        trials_df = pd.DataFrame([
            {
                'trial': trial.number,
                'value': trial.value,
                'learning_rate': trial.params.get('learning_rate', 0),
                'batch_size': trial.params.get('batch_size', 0),
                'weight_decay': trial.params.get('weight_decay', 0),
            }
            for trial in self.study.trials
        ])
        
        # 绘制优化历史
        plt.figure(figsize=(15, 10))
        
        # 优化曲线
        plt.subplot(2, 3, 1)
        plt.plot(trials_df['trial'], trials_df['value'])
        plt.title('Optimization History')
        plt.xlabel('Trial')
        plt.ylabel('Accuracy')
        
        # 学习率分布
        plt.subplot(2, 3, 2)
        plt.scatter(trials_df['learning_rate'], trials_df['value'])
        plt.title('Learning Rate vs Accuracy')
        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.xscale('log')
        
        # 批次大小分布
        plt.subplot(2, 3, 3)
        plt.boxplot([trials_df[trials_df['batch_size'] == bs]['value'] for bs in [16, 32, 64]])
        plt.title('Batch Size vs Accuracy')
        plt.xlabel('Batch Size')
        plt.ylabel('Accuracy')
        
        # 权重衰减分布
        plt.subplot(2, 3, 4)
        plt.scatter(trials_df['weight_decay'], trials_df['value'])
        plt.title('Weight Decay vs Accuracy')
        plt.xlabel('Weight Decay')
        plt.ylabel('Accuracy')
        plt.xscale('log')
        
        # 参数重要性
        plt.subplot(2, 3, 5)
        importance = optuna.importance.get_param_importances(self.study)
        plt.bar(importance.keys(), importance.values())
        plt.title('Parameter Importance')
        plt.xticks(rotation=45)
        
        # 优化收敛
        plt.subplot(2, 3, 6)
        best_values = []
        for i in range(len(trials_df)):
            best_values.append(max(trials_df['value'][:i+1]))
        plt.plot(trials_df['trial'], best_values)
        plt.title('Best Value Convergence')
        plt.xlabel('Trial')
        plt.ylabel('Best Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimization_report.png', dpi=300, bbox_inches='tight')
        plt.close()


class ModelSelector:
    """模型选择器"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def select_best_model(self, model_types: List[str] = None) -> str:
        """选择最佳模型"""
        if model_types is None:
            model_types = ['efficientnet', 'vision_transformer', 'dual_stream']
        
        self.logger.info(f"开始模型选择: {model_types}")
        
        results = {}
        
        for model_type in model_types:
            try:
                # 使用最佳参数训练模型
                best_params = self._load_best_params(model_type)
                if best_params is None:
                    self.logger.warning(f"未找到 {model_type} 的最佳参数，使用默认参数")
                    best_params = self._get_default_params(model_type)
                
                # 训练模型
                config = self._create_full_training_config(model_type, best_params)
                trainer = ModelTrainer(config)
                
                # 评估模型
                score = self._evaluate_model(trainer, config)
                results[model_type] = score
                
                self.logger.info(f"{model_type}: {score:.4f}")
                
            except Exception as e:
                self.logger.error(f"模型 {model_type} 评估失败: {e}")
                results[model_type] = 0.0
        
        # 选择最佳模型
        best_model = max(results, key=results.get)
        best_score = results[best_model]
        
        self.logger.info(f"最佳模型: {best_model} (准确率: {best_score:.4f})")
        
        # 保存选择结果
        self._save_selection_results(results, best_model)
        
        return best_model
    
    def _load_best_params(self, model_type: str) -> Optional[Dict]:
        """加载最佳参数"""
        params_file = self.output_dir / f"{model_type}_best_params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                data = json.load(f)
                return data['best_params']
        return None
    
    def _get_default_params(self, model_type: str) -> Dict:
        """获取默认参数"""
        defaults = {
            'efficientnet': {
                'learning_rate': 1e-4,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'dropout': 0.3
            },
            'vision_transformer': {
                'learning_rate': 5e-5,
                'batch_size': 16,
                'weight_decay': 1e-4,
                'dropout': 0.1,
                'patch_size': 16,
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12
            },
            'dual_stream': {
                'learning_rate': 1e-4,
                'batch_size': 24,
                'weight_decay': 1e-4,
                'dropout': 0.3,
                'fusion_dropout': 0.5,
                'attention_heads': 8
            }
        }
        return defaults.get(model_type, {})
    
    def _create_full_training_config(self, model_type: str, params: Dict) -> Dict:
        """创建完整训练配置"""
        config = {
            'model_type': model_type,
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir / model_type),
            'num_epochs': 100,
            'num_workers': 4,
            'img_size': 224,
            'save_interval': 10,
            'log_interval': 100,
            **params
        }
        return config
    
    def _evaluate_model(self, trainer, config: Dict) -> float:
        """评估模型"""
        # 加载数据
        train_transform = get_transforms(img_size=config.get('img_size', 224), is_training=True)
        val_transform = get_transforms(img_size=config.get('img_size', 224), is_training=False)
        
        train_dataset = ScreenshotDataset(config['data_dir'], transform=train_transform, is_training=True)
        val_dataset = ScreenshotDataset(config['data_dir'], transform=val_transform, is_training=False)
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']
        )
        
        # 训练模型
        trainer.train(train_loader, val_loader)
        
        # 返回最佳验证准确率
        return trainer.best_accuracy / 100.0  # 转换为0-1范围
    
    def _save_selection_results(self, results: Dict, best_model: str):
        """保存选择结果"""
        selection_result = {
            'selection_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results': results,
            'best_model': best_model,
            'best_score': results[best_model]
        }
        
        with open(self.output_dir / 'model_selection_results.json', 'w') as f:
            json.dump(selection_result, f, indent=2)


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def optimize_performance(self):
        """优化模型性能"""
        self.logger.info("开始性能优化")
        
        # 1. 模型量化
        self._quantize_model()
        
        # 2. 模型剪枝
        self._prune_model()
        
        # 3. 模型蒸馏
        self._distill_model()
        
        # 4. 推理优化
        self._optimize_inference()
        
        self.logger.info("性能优化完成")
    
    def _quantize_model(self):
        """模型量化"""
        self.logger.info("执行模型量化")
        
        try:
            # 加载模型
            checkpoint = torch.load(self.model_path, map_location='cpu')
            model = self._create_model_from_checkpoint(checkpoint)
            
            # 量化模型
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            
            # 保存量化模型
            quantized_path = self.output_dir / 'quantized_model.pth'
            torch.save(quantized_model.state_dict(), quantized_path)
            
            self.logger.info(f"量化模型已保存: {quantized_path}")
            
        except Exception as e:
            self.logger.error(f"模型量化失败: {e}")
    
    def _prune_model(self):
        """模型剪枝"""
        self.logger.info("执行模型剪枝")
        
        try:
            # 加载模型
            checkpoint = torch.load(self.model_path, map_location='cpu')
            model = self._create_model_from_checkpoint(checkpoint)
            
            # 剪枝模型
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=0.3)
                elif isinstance(module, nn.Linear):
                    torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=0.2)
            
            # 保存剪枝模型
            pruned_path = self.output_dir / 'pruned_model.pth'
            torch.save(model.state_dict(), pruned_path)
            
            self.logger.info(f"剪枝模型已保存: {pruned_path}")
            
        except Exception as e:
            self.logger.error(f"模型剪枝失败: {e}")
    
    def _distill_model(self):
        """模型蒸馏"""
        self.logger.info("执行模型蒸馏")
        
        # 这里应该实现知识蒸馏逻辑
        # 需要一个教师模型和学生模型
        self.logger.info("模型蒸馏功能需要进一步实现")
    
    def _optimize_inference(self):
        """推理优化"""
        self.logger.info("执行推理优化")
        
        try:
            # 加载模型
            checkpoint = torch.load(self.model_path, map_location='cpu')
            model = self._create_model_from_checkpoint(checkpoint)
            model.eval()
            
            # 转换为TorchScript
            dummy_input = torch.randn(1, 3, 224, 224)
            traced_model = torch.jit.trace(model, dummy_input)
            
            # 保存优化模型
            optimized_path = self.output_dir / 'optimized_model.pt'
            traced_model.save(optimized_path)
            
            self.logger.info(f"优化模型已保存: {optimized_path}")
            
        except Exception as e:
            self.logger.error(f"推理优化失败: {e}")
    
    def _create_model_from_checkpoint(self, checkpoint: Dict):
        """从检查点创建模型"""
        model_config = checkpoint['config']
        model_type = model_config['model_type']
        
        if model_type == 'vision_transformer':
            model = create_vision_transformer(num_classes=model_config.get('num_classes', 2))
        elif model_type == 'dual_stream':
            model = create_dual_stream_network(num_classes=model_config.get('num_classes', 2))
        elif model_type == 'efficientnet':
            from torchvision import models
            model = models.efficientnet_b0(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, model_config.get('num_classes', 2))
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def main():
    parser = argparse.ArgumentParser(description='模型训练与优化工具')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='./outputs', 
                       help='输出目录')
    parser.add_argument('--mode', type=str, choices=['optimize', 'select', 'train', 'optimize_performance'], 
                       required=True, help='运行模式')
    parser.add_argument('--model_type', type=str, 
                       choices=['efficientnet', 'vision_transformer', 'dual_stream'], 
                       help='模型类型（仅用于optimize模式）')
    parser.add_argument('--n_trials', type=int, default=50, 
                       help='优化试验次数')
    parser.add_argument('--model_path', type=str, 
                       help='模型路径（仅用于optimize_performance模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'optimize':
        if not args.model_type:
            parser.error("optimize模式需要指定model_type")
        
        optimizer = HyperparameterOptimizer(args.data_dir, args.output_dir)
        best_params = optimizer.optimize_hyperparameters(args.model_type, args.n_trials)
        print(f"最佳参数: {best_params}")
    
    elif args.mode == 'select':
        selector = ModelSelector(args.data_dir, args.output_dir)
        best_model = selector.select_best_model()
        print(f"最佳模型: {best_model}")
    
    elif args.mode == 'train':
        # 使用最佳参数训练模型
        selector = ModelSelector(args.data_dir, args.output_dir)
        best_model = selector.select_best_model()
        
        # 加载最佳参数
        params_file = Path(args.output_dir) / f"{best_model}_best_params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                data = json.load(f)
                best_params = data['best_params']
        else:
            best_params = {}
        
        # 创建训练配置
        config = {
            'model_type': best_model,
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'num_epochs': 100,
            'num_workers': 4,
            'img_size': 224,
            **best_params
        }
        
        # 训练模型
        trainer = ModelTrainer(config)
        train_transform = get_transforms(img_size=224, is_training=True)
        val_transform = get_transforms(img_size=224, is_training=False)
        
        train_dataset = ScreenshotDataset(args.data_dir, transform=train_transform, is_training=True)
        val_dataset = ScreenshotDataset(args.data_dir, transform=val_transform, is_training=False)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.get('batch_size', 32), shuffle=True, num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=4
        )
        
        trainer.train(train_loader, val_loader)
        print(f"模型训练完成，最佳准确率: {trainer.best_accuracy:.2f}%")
    
    elif args.mode == 'optimize_performance':
        if not args.model_path:
            parser.error("optimize_performance模式需要指定model_path")
        
        optimizer = PerformanceOptimizer(args.model_path, args.output_dir)
        optimizer.optimize_performance()
        print("性能优化完成")


if __name__ == '__main__':
    main()