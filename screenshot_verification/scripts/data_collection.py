#!/usr/bin/env python3
"""
数据收集与准备脚本
包括真实截图收集、伪造截图生成、数据标注和数据集构建
"""
import os
import sys
import argparse
import logging
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from tqdm import tqdm
import shutil

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings


class ScreenshotCollector:
    """截图收集器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.authentic_dir = self.output_dir / "authentic"
        self.fake_dir = self.output_dir / "fake"
        
        # 创建目录
        self.authentic_dir.mkdir(parents=True, exist_ok=True)
        self.fake_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """设置日志"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'data_collection.log'),
                logging.StreamHandler()
            ]
        )
    
    def collect_authentic_screenshots(self, num_samples: int = 1000):
        """收集真实截图"""
        self.logger.info(f"开始收集 {num_samples} 张真实截图")
        
        # 1. 从公开数据集收集
        self._collect_from_datasets(num_samples // 2)
        
        # 2. 从网络收集
        self._collect_from_web(num_samples // 4)
        
        # 3. 从本地设备收集
        self._collect_from_local(num_samples // 4)
        
        self.logger.info("真实截图收集完成")
    
    def _collect_from_datasets(self, num_samples: int):
        """从公开数据集收集"""
        datasets = [
            "https://example.com/dataset1",
            "https://example.com/dataset2"
        ]
        
        for dataset_url in datasets:
            try:
                # 这里应该实现具体的数据集下载逻辑
                self.logger.info(f"从数据集 {dataset_url} 收集截图")
                # 实现下载和保存逻辑
            except Exception as e:
                self.logger.error(f"从数据集收集失败: {e}")
    
    def _collect_from_web(self, num_samples: int):
        """从网络收集截图"""
        # 使用Selenium收集网页截图
        try:
            driver = webdriver.Chrome()  # 需要安装ChromeDriver
            
            websites = [
                "https://www.google.com",
                "https://www.github.com",
                "https://www.stackoverflow.com",
                # 添加更多网站
            ]
            
            for i, website in enumerate(websites[:num_samples // len(websites)]):
                try:
                    driver.get(website)
                    time.sleep(2)  # 等待页面加载
                    
                    # 截图
                    screenshot_path = self.authentic_dir / f"web_{i:04d}.png"
                    driver.save_screenshot(str(screenshot_path))
                    
                    self.logger.info(f"收集网页截图: {screenshot_path}")
                    
                except Exception as e:
                    self.logger.error(f"收集网页截图失败: {e}")
            
            driver.quit()
            
        except Exception as e:
            self.logger.error(f"Web收集器初始化失败: {e}")
    
    def _collect_from_local(self, num_samples: int):
        """从本地设备收集截图"""
        # 这里应该实现从本地设备收集截图的逻辑
        # 可能需要用户手动提供或使用自动化工具
        self.logger.info("本地截图收集功能需要手动实现")


class FakeScreenshotGenerator:
    """伪造截图生成器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.fake_dir = self.output_dir / "fake"
        self.fake_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def generate_fake_screenshots(self, num_samples: int = 1000):
        """生成伪造截图"""
        self.logger.info(f"开始生成 {num_samples} 张伪造截图")
        
        # 1. 文本编辑伪造
        self._generate_text_editing_fakes(num_samples // 4)
        
        # 2. 图像拼接伪造
        self._generate_image_compositing_fakes(num_samples // 4)
        
        # 3. 滤镜和特效伪造
        self._generate_filter_fakes(num_samples // 4)
        
        # 4. 深度伪造
        self._generate_deepfake_fakes(num_samples // 4)
        
        self.logger.info("伪造截图生成完成")
    
    def _generate_text_editing_fakes(self, num_samples: int):
        """生成文本编辑伪造"""
        for i in range(num_samples):
            # 创建基础图像
            img = Image.new('RGB', (400, 800), color='white')
            draw = ImageDraw.Draw(img)
            
            # 添加文本
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # 原始文本
            original_text = "余额: $1,000.00"
            draw.text((50, 100), original_text, fill='black', font=font)
            
            # 伪造文本（随机修改）
            fake_amounts = ["$10,000.00", "$100,000.00", "$1,000,000.00"]
            fake_text = f"余额: {random.choice(fake_amounts)}"
            
            # 在稍微不同的位置绘制伪造文本
            draw.text((50, 150), fake_text, fill='red', font=font)
            
            # 保存
            fake_path = self.fake_dir / f"text_edit_{i:04d}.png"
            img.save(fake_path)
            
            self.logger.info(f"生成文本编辑伪造: {fake_path}")
    
    def _generate_image_compositing_fakes(self, num_samples: int):
        """生成图像拼接伪造"""
        for i in range(num_samples):
            # 创建基础图像
            base_img = Image.new('RGB', (400, 800), color='lightblue')
            
            # 添加一些UI元素
            draw = ImageDraw.Draw(base_img)
            
            # 添加按钮
            draw.rectangle([50, 200, 150, 250], fill='green', outline='black')
            draw.text((70, 215), "确认", fill='white')
            
            # 添加输入框
            draw.rectangle([50, 300, 350, 350], fill='white', outline='black')
            draw.text((60, 315), "输入金额", fill='gray')
            
            # 保存
            fake_path = self.fake_dir / f"composite_{i:04d}.png"
            base_img.save(fake_path)
            
            self.logger.info(f"生成图像拼接伪造: {fake_path}")
    
    def _generate_filter_fakes(self, num_samples: int):
        """生成滤镜和特效伪造"""
        for i in range(num_samples):
            # 创建基础图像
            img = Image.new('RGB', (400, 800), color='white')
            draw = ImageDraw.Draw(img)
            
            # 添加一些内容
            draw.text((50, 100), "转账成功", fill='black')
            draw.text((50, 150), "金额: $1,000", fill='black')
            
            # 转换为numpy数组进行滤镜处理
            img_array = np.array(img)
            
            # 应用随机滤镜效果
            effects = [
                lambda x: cv2.GaussianBlur(x, (5, 5), 0),
                lambda x: cv2.medianBlur(x, 5),
                lambda x: cv2.bilateralFilter(x, 9, 75, 75),
                lambda x: cv2.addWeighted(x, 1.2, x, 0, 10)
            ]
            
            effect = random.choice(effects)
            processed = effect(img_array)
            
            # 转换回PIL图像
            fake_img = Image.fromarray(processed)
            
            # 保存
            fake_path = self.fake_dir / f"filter_{i:04d}.png"
            fake_img.save(fake_path)
            
            self.logger.info(f"生成滤镜伪造: {fake_path}")
    
    def _generate_deepfake_fakes(self, num_samples: int):
        """生成深度伪造"""
        for i in range(num_samples):
            # 创建复杂的伪造图像
            img = Image.new('RGB', (400, 800), color='white')
            draw = ImageDraw.Draw(img)
            
            # 模拟银行界面
            # 背景
            draw.rectangle([0, 0, 400, 800], fill='#f0f0f0')
            
            # 头部
            draw.rectangle([0, 0, 400, 100], fill='#2c3e50')
            draw.text((150, 40), "银行APP", fill='white')
            
            # 内容区域
            draw.rectangle([20, 120, 380, 200], fill='white', outline='#bdc3c7')
            draw.text((40, 140), "账户余额", fill='#2c3e50')
            draw.text((40, 170), "$999,999.99", fill='#27ae60', font=ImageFont.load_default())
            
            # 添加一些不一致的元素（伪造特征）
            # 字体不一致
            draw.text((40, 250), "最近交易", fill='red')  # 使用不同颜色
            
            # 对齐问题
            draw.rectangle([40, 280, 360, 320], fill='#ecf0f1')
            draw.text((50, 290), "转账 -$1000", fill='black')
            draw.text((300, 290), "2024-01-01", fill='gray')
            
            # 保存
            fake_path = self.fake_dir / f"deepfake_{i:04d}.png"
            img.save(fake_path)
            
            self.logger.info(f"生成深度伪造: {fake_path}")


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
    
    def build_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        """构建训练、验证、测试数据集"""
        self.logger.info("开始构建数据集")
        
        # 检查数据
        authentic_files = list((self.data_dir / "authentic").glob("*.png"))
        fake_files = list((self.data_dir / "fake").glob("*.png"))
        
        self.logger.info(f"找到 {len(authentic_files)} 张真实截图")
        self.logger.info(f"找到 {len(fake_files)} 张伪造截图")
        
        # 创建数据集目录
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"
        
        for dir_path in [train_dir, val_dir, test_dir]:
            (dir_path / "authentic").mkdir(parents=True, exist_ok=True)
            (dir_path / "fake").mkdir(parents=True, exist_ok=True)
        
        # 分割数据
        self._split_and_copy_files(authentic_files, "authentic", train_ratio, val_ratio, test_ratio)
        self._split_and_copy_files(fake_files, "fake", train_ratio, val_ratio, test_ratio)
        
        # 生成数据集信息
        self._generate_dataset_info()
        
        self.logger.info("数据集构建完成")
    
    def _split_and_copy_files(self, files: List[Path], label: str, train_ratio: float, val_ratio: float, test_ratio: float):
        """分割并复制文件"""
        random.shuffle(files)
        
        n = len(files)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        # 复制文件
        for file_path in train_files:
            shutil.copy2(file_path, self.data_dir / "train" / label / file_path.name)
        
        for file_path in val_files:
            shutil.copy2(file_path, self.data_dir / "val" / label / file_path.name)
        
        for file_path in test_files:
            shutil.copy2(file_path, self.data_dir / "test" / label / file_path.name)
        
        self.logger.info(f"{label}: 训练集 {len(train_files)}, 验证集 {len(val_files)}, 测试集 {len(test_files)}")
    
    def _generate_dataset_info(self):
        """生成数据集信息"""
        info = {
            "dataset_info": {
                "total_authentic": len(list((self.data_dir / "authentic").glob("*.png"))),
                "total_fake": len(list((self.data_dir / "fake").glob("*.png"))),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "splits": {}
        }
        
        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / split
            authentic_count = len(list((split_dir / "authentic").glob("*.png")))
            fake_count = len(list((split_dir / "fake").glob("*.png")))
            
            info["splits"][split] = {
                "authentic": authentic_count,
                "fake": fake_count,
                "total": authentic_count + fake_count
            }
        
        # 保存信息
        with open(self.data_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        self.logger.info("数据集信息已保存")


class DataValidator:
    """数据验证器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
    
    def validate_dataset(self):
        """验证数据集质量"""
        self.logger.info("开始验证数据集")
        
        issues = []
        
        # 检查文件完整性
        for split in ["train", "val", "test"]:
            for label in ["authentic", "fake"]:
                split_dir = self.data_dir / split / label
                if not split_dir.exists():
                    issues.append(f"目录不存在: {split_dir}")
                    continue
                
                files = list(split_dir.glob("*.png"))
                if not files:
                    issues.append(f"没有文件: {split_dir}")
                    continue
                
                # 检查图像质量
                for file_path in files:
                    try:
                        img = Image.open(file_path)
                        if img.size[0] < 100 or img.size[1] < 100:
                            issues.append(f"图像太小: {file_path}")
                        if img.mode != 'RGB':
                            issues.append(f"图像格式错误: {file_path}")
                    except Exception as e:
                        issues.append(f"图像损坏: {file_path} - {e}")
        
        # 检查数据平衡
        for split in ["train", "val", "test"]:
            authentic_count = len(list((self.data_dir / split / "authentic").glob("*.png")))
            fake_count = len(list((self.data_dir / split / "fake").glob("*.png")))
            
            if authentic_count == 0 or fake_count == 0:
                issues.append(f"{split} 集缺少某个类别")
            elif abs(authentic_count - fake_count) / max(authentic_count, fake_count) > 0.3:
                issues.append(f"{split} 集数据不平衡: authentic={authentic_count}, fake={fake_count}")
        
        # 报告结果
        if issues:
            self.logger.warning("发现以下问题:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        else:
            self.logger.info("数据集验证通过")
        
        return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description='数据收集与准备工具')
    parser.add_argument('--output_dir', type=str, default='./data/screenshots', 
                       help='输出目录')
    parser.add_argument('--num_authentic', type=int, default=1000, 
                       help='真实截图数量')
    parser.add_argument('--num_fake', type=int, default=1000, 
                       help='伪造截图数量')
    parser.add_argument('--collect_only', action='store_true', 
                       help='仅收集数据，不生成伪造')
    parser.add_argument('--generate_only', action='store_true', 
                       help='仅生成伪造，不收集数据')
    parser.add_argument('--validate_only', action='store_true', 
                       help='仅验证数据集')
    
    args = parser.parse_args()
    
    if args.validate_only:
        # 仅验证
        validator = DataValidator(args.output_dir)
        validator.validate_dataset()
        return
    
    # 收集真实截图
    if not args.generate_only:
        collector = ScreenshotCollector(args.output_dir)
        collector.collect_authentic_screenshots(args.num_authentic)
    
    # 生成伪造截图
    if not args.collect_only:
        generator = FakeScreenshotGenerator(args.output_dir)
        generator.generate_fake_screenshots(args.num_fake)
    
    # 构建数据集
    builder = DatasetBuilder(args.output_dir)
    builder.build_dataset()
    
    # 验证数据集
    validator = DataValidator(args.output_dir)
    validator.validate_dataset()


if __name__ == '__main__':
    main()