"""
核心检测器模块
包含传统图像分析、元数据分析和AI深度学习检测器
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import imagehash
from PIL import Image
import exifread
import piexif
from scipy import ndimage
from scipy.stats import entropy
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """检测结果数据类"""
    is_authentic: bool
    confidence: float
    risk_factors: List[Dict[str, Any]]
    processing_time: float
    detector_name: str


@dataclass
class RiskFactor:
    """风险因子数据类"""
    type: str
    severity: str  # low, medium, high, critical
    confidence: float
    location: Optional[List[int]] = None
    description: str = ""


class BaseDetector(ABC):
    """检测器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def detect(self, image: np.ndarray, metadata: Dict[str, Any]) -> DetectionResult:
        """执行检测"""
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class TraditionalDetector(BaseDetector):
    """传统图像分析检测器"""
    
    def __init__(self):
        super().__init__("TraditionalDetector")
    
    def detect(self, image: np.ndarray, metadata: Dict[str, Any]) -> DetectionResult:
        """执行传统图像分析检测"""
        import time
        start_time = time.time()
        
        risk_factors = []
        total_confidence = 0.0
        checks_count = 0
        
        # 1. JPEG压缩痕迹检测
        compression_result = self._detect_compression_artifacts(image)
        if compression_result:
            risk_factors.append(compression_result)
            total_confidence += compression_result["confidence"]
            checks_count += 1
        
        # 2. 噪声模式检测
        noise_result = self._detect_noise_patterns(image)
        if noise_result:
            risk_factors.append(noise_result)
            total_confidence += noise_result["confidence"]
            checks_count += 1
        
        # 3. 边缘不一致性检测
        edge_result = self._detect_edge_inconsistency(image)
        if edge_result:
            risk_factors.append(edge_result)
            total_confidence += edge_result["confidence"]
            checks_count += 1
        
        # 4. 双重压缩检测
        double_compression_result = self._detect_double_compression(image)
        if double_compression_result:
            risk_factors.append(double_compression_result)
            total_confidence += double_compression_result["confidence"]
            checks_count += 1
        
        processing_time = time.time() - start_time
        
        # 计算平均置信度
        avg_confidence = total_confidence / max(checks_count, 1)
        
        # 判断真伪（置信度越高越可能是真实的）
        is_authentic = avg_confidence > 0.7
        
        return DetectionResult(
            is_authentic=is_authentic,
            confidence=avg_confidence,
            risk_factors=risk_factors,
            processing_time=processing_time,
            detector_name=self.name
        )
    
    def _detect_compression_artifacts(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """检测JPEG压缩痕迹"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 计算DCT系数
            dct = cv2.dct(np.float32(gray))
            
            # 分析DCT系数的分布
            dct_flat = dct.flatten()
            dct_hist, _ = np.histogram(dct_flat, bins=50)
            
            # 计算熵值
            dct_entropy = entropy(dct_hist + 1e-10)
            
            # 检测块效应
            block_artifacts = self._detect_block_artifacts(gray)
            
            # 综合评分
            compression_score = (dct_entropy / 10.0 + block_artifacts) / 2.0
            
            if compression_score > 0.6:
                return {
                    "type": "compression_artifact",
                    "severity": "medium",
                    "confidence": compression_score,
                    "description": f"检测到JPEG压缩痕迹，熵值: {dct_entropy:.3f}"
                }
            
        except Exception as e:
            self.logger.error(f"压缩痕迹检测失败: {e}")
        
        return None
    
    def _detect_block_artifacts(self, gray: np.ndarray) -> float:
        """检测块效应"""
        try:
            # 计算8x8块的方差
            block_size = 8
            h, w = gray.shape
            block_variances = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    var = np.var(block)
                    block_variances.append(var)
            
            # 计算块间方差的一致性
            block_var_std = np.std(block_variances)
            block_var_mean = np.mean(block_variances)
            
            # 归一化评分
            consistency_score = 1.0 - min(block_var_std / (block_var_mean + 1e-10), 1.0)
            
            return consistency_score
            
        except Exception as e:
            self.logger.error(f"块效应检测失败: {e}")
            return 0.0
    
    def _detect_noise_patterns(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """检测噪声模式"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 计算局部噪声
            kernel = np.ones((3, 3), np.float32) / 9
            smoothed = cv2.filter2D(gray, -1, kernel)
            noise = gray.astype(np.float32) - smoothed
            
            # 分析噪声分布
            noise_std = np.std(noise)
            noise_mean = np.mean(noise)
            
            # 检测噪声的不一致性
            noise_inconsistency = self._calculate_noise_inconsistency(noise)
            
            # 综合评分
            noise_score = (noise_inconsistency + (noise_std / 50.0)) / 2.0
            
            if noise_score > 0.5:
                return {
                    "type": "noise_pattern",
                    "severity": "low",
                    "confidence": noise_score,
                    "description": f"检测到异常噪声模式，标准差: {noise_std:.3f}"
                }
            
        except Exception as e:
            self.logger.error(f"噪声模式检测失败: {e}")
        
        return None
    
    def _calculate_noise_inconsistency(self, noise: np.ndarray) -> float:
        """计算噪声不一致性"""
        try:
            # 将图像分成多个区域
            h, w = noise.shape
            regions = []
            
            for i in range(0, h, h//4):
                for j in range(0, w, w//4):
                    region = noise[i:i+h//4, j:j+w//4]
                    if region.size > 0:
                        regions.append(np.std(region))
            
            # 计算区域间噪声标准差的一致性
            region_std = np.std(regions)
            region_mean = np.mean(regions)
            
            # 归一化评分
            consistency = 1.0 - min(region_std / (region_mean + 1e-10), 1.0)
            
            return consistency
            
        except Exception as e:
            self.logger.error(f"噪声不一致性计算失败: {e}")
            return 0.0
    
    def _detect_edge_inconsistency(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """检测边缘不一致性"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 计算边缘
            edges = cv2.Canny(gray, 50, 150)
            
            # 分析边缘分布
            edge_density = np.sum(edges > 0) / edges.size
            
            # 检测不自然的边缘模式
            edge_inconsistency = self._analyze_edge_patterns(edges)
            
            # 综合评分
            edge_score = (edge_inconsistency + edge_density) / 2.0
            
            if edge_score > 0.6:
                return {
                    "type": "edge_inconsistency",
                    "severity": "medium",
                    "confidence": edge_score,
                    "description": f"检测到边缘不一致性，边缘密度: {edge_density:.3f}"
                }
            
        except Exception as e:
            self.logger.error(f"边缘不一致性检测失败: {e}")
        
        return None
    
    def _analyze_edge_patterns(self, edges: np.ndarray) -> float:
        """分析边缘模式"""
        try:
            # 计算边缘的方向分布
            h, w = edges.shape
            directions = []
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if edges[i, j] > 0:
                        # 计算局部梯度方向
                        gx = edges[i, j+1] - edges[i, j-1]
                        gy = edges[i+1, j] - edges[i-1, j]
                        if gx != 0 or gy != 0:
                            angle = np.arctan2(gy, gx)
                            directions.append(angle)
            
            if len(directions) > 0:
                # 计算方向分布的一致性
                direction_hist, _ = np.histogram(directions, bins=18)
                direction_entropy = entropy(direction_hist + 1e-10)
                
                # 归一化评分
                consistency = 1.0 - min(direction_entropy / 10.0, 1.0)
                return consistency
            
        except Exception as e:
            self.logger.error(f"边缘模式分析失败: {e}")
        
        return 0.0
    
    def _detect_double_compression(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """检测双重压缩"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 计算DCT系数
            dct = cv2.dct(np.float32(gray))
            
            # 分析DCT系数的分布特征
            dct_flat = dct.flatten()
            
            # 计算高频成分的比例
            high_freq_ratio = np.sum(np.abs(dct_flat) > np.std(dct_flat)) / len(dct_flat)
            
            # 检测双重压缩特征
            double_compression_score = high_freq_ratio
            
            if double_compression_score > 0.7:
                return {
                    "type": "double_compression",
                    "severity": "high",
                    "confidence": double_compression_score,
                    "description": f"检测到可能的双重压缩，高频成分比例: {high_freq_ratio:.3f}"
                }
            
        except Exception as e:
            self.logger.error(f"双重压缩检测失败: {e}")
        
        return None


class MetadataDetector(BaseDetector):
    """元数据分析检测器"""
    
    def __init__(self):
        super().__init__("MetadataDetector")
    
    def detect(self, image: np.ndarray, metadata: Dict[str, Any]) -> DetectionResult:
        """执行元数据分析检测"""
        import time
        start_time = time.time()
        
        risk_factors = []
        total_confidence = 0.0
        checks_count = 0
        
        # 1. EXIF信息检测
        exif_result = self._analyze_exif_metadata(image, metadata)
        if exif_result:
            risk_factors.append(exif_result)
            total_confidence += exif_result["confidence"]
            checks_count += 1
        
        # 2. 文件属性检测
        file_result = self._analyze_file_attributes(image, metadata)
        if file_result:
            risk_factors.append(file_result)
            total_confidence += file_result["confidence"]
            checks_count += 1
        
        # 3. 图像指纹检测
        fingerprint_result = self._analyze_image_fingerprint(image)
        if fingerprint_result:
            risk_factors.append(fingerprint_result)
            total_confidence += fingerprint_result["confidence"]
            checks_count += 1
        
        processing_time = time.time() - start_time
        
        # 计算平均置信度
        avg_confidence = total_confidence / max(checks_count, 1)
        
        # 判断真伪
        is_authentic = avg_confidence > 0.6
        
        return DetectionResult(
            is_authentic=is_authentic,
            confidence=avg_confidence,
            risk_factors=risk_factors,
            processing_time=processing_time,
            detector_name=self.name
        )
    
    def _analyze_exif_metadata(self, image: np.ndarray, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """分析EXIF元数据"""
        try:
            # 将numpy数组转换为PIL图像
            pil_image = Image.fromarray(image)
            
            # 检查EXIF数据
            exif_data = pil_image.getexif()
            
            if exif_data:
                # 分析EXIF信息
                suspicious_flags = []
                
                # 检查设备信息
                if 271 in exif_data:  # Make
                    make = exif_data[271]
                    if make and "screenshot" in make.lower():
                        suspicious_flags.append("设备信息异常")
                
                # 检查软件信息
                if 305 in exif_data:  # Software
                    software = exif_data[305]
                    if software and any(keyword in software.lower() for keyword in ["photoshop", "gimp", "paint"]):
                        suspicious_flags.append("检测到图像编辑软件")
                
                # 检查时间信息
                if 36867 in exif_data:  # DateTimeOriginal
                    original_time = exif_data[36867]
                    if original_time:
                        # 这里可以添加时间合理性检查
                        pass
                
                if suspicious_flags:
                    return {
                        "type": "exif_metadata",
                        "severity": "medium",
                        "confidence": 0.7,
                        "description": f"EXIF元数据异常: {', '.join(suspicious_flags)}"
                    }
            else:
                # 没有EXIF数据可能是正常的截图
                return {
                    "type": "exif_metadata",
                    "severity": "low",
                    "confidence": 0.3,
                    "description": "缺少EXIF元数据"
                }
            
        except Exception as e:
            self.logger.error(f"EXIF元数据分析失败: {e}")
        
        return None
    
    def _analyze_file_attributes(self, image: np.ndarray, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """分析文件属性"""
        try:
            # 分析图像的基本属性
            h, w = image.shape[:2]
            
            # 检查分辨率合理性
            if w > 10000 or h > 10000:
                return {
                    "type": "file_attributes",
                    "severity": "medium",
                    "confidence": 0.6,
                    "description": f"分辨率异常: {w}x{h}"
                }
            
            # 检查宽高比
            aspect_ratio = w / h
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                return {
                    "type": "file_attributes",
                    "severity": "low",
                    "confidence": 0.4,
                    "description": f"宽高比异常: {aspect_ratio:.3f}"
                }
            
            # 检查图像大小
            image_size = image.nbytes
            if image_size > 50 * 1024 * 1024:  # 50MB
                return {
                    "type": "file_attributes",
                    "severity": "medium",
                    "confidence": 0.5,
                    "description": f"图像文件过大: {image_size / 1024 / 1024:.1f}MB"
                }
            
        except Exception as e:
            self.logger.error(f"文件属性分析失败: {e}")
        
        return None
    
    def _analyze_image_fingerprint(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """分析图像指纹"""
        try:
            # 将numpy数组转换为PIL图像
            pil_image = Image.fromarray(image)
            
            # 计算多种图像哈希
            hash_average = imagehash.average_hash(pil_image)
            hash_phash = imagehash.phash(pil_image)
            hash_dhash = imagehash.dhash(pil_image)
            hash_whash = imagehash.whash(pil_image)
            
            # 分析哈希值的特征
            hash_values = [hash_average, hash_phash, hash_dhash, hash_whash]
            
            # 计算哈希值的一致性
            hash_consistency = self._calculate_hash_consistency(hash_values)
            
            if hash_consistency < 0.5:
                return {
                    "type": "image_fingerprint",
                    "severity": "medium",
                    "confidence": 0.6,
                    "description": f"图像指纹异常，一致性: {hash_consistency:.3f}"
                }
            
        except Exception as e:
            self.logger.error(f"图像指纹分析失败: {e}")
        
        return None
    
    def _calculate_hash_consistency(self, hash_values: List) -> float:
        """计算哈希值一致性"""
        try:
            # 将哈希值转换为二进制字符串
            hash_binaries = []
            for hash_val in hash_values:
                hash_bin = format(int(str(hash_val), 16), '064b')
                hash_binaries.append(hash_bin)
            
            # 计算哈希值之间的相似度
            similarities = []
            for i in range(len(hash_binaries)):
                for j in range(i + 1, len(hash_binaries)):
                    # 计算汉明距离
                    hamming_distance = sum(a != b for a, b in zip(hash_binaries[i], hash_binaries[j]))
                    similarity = 1.0 - (hamming_distance / 64.0)
                    similarities.append(similarity)
            
            # 返回平均相似度
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"哈希一致性计算失败: {e}")
            return 0.0


class AIDetector(BaseDetector):
    """AI深度学习检测器"""
    
    def __init__(self, model_path: str = None):
        super().__init__("AIDetector")
        self.model_path = model_path
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """加载AI模型"""
        try:
            import torch
            from torch import nn
            import timm
            
            # 设置设备
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            
            # 加载预训练模型（这里使用EfficientNet作为示例）
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
            self.model.to(self.device)
            self.model.eval()
            
            # 如果有自定义模型路径，加载自定义模型
            if self.model_path and os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            
            self.logger.info(f"AI模型加载成功，设备: {self.device}")
            
        except Exception as e:
            self.logger.error(f"AI模型加载失败: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray, metadata: Dict[str, Any]) -> DetectionResult:
        """执行AI深度学习检测"""
        import time
        start_time = time.time()
        
        if self.model is None:
            return DetectionResult(
                is_authentic=True,
                confidence=0.5,
                risk_factors=[],
                processing_time=time.time() - start_time,
                detector_name=self.name
            )
        
        try:
            # 图像预处理
            processed_image = self._preprocess_for_model(image)
            
            # 模型推理
            with torch.no_grad():
                output = self.model(processed_image)
                probabilities = torch.softmax(output, dim=1)
                
                # 获取真实性和置信度
                authentic_prob = probabilities[0, 1].item()  # 假设1表示真实
                confidence = max(authentic_prob, 1 - authentic_prob)
                is_authentic = authentic_prob > 0.5
            
            processing_time = time.time() - start_time
            
            risk_factors = []
            if not is_authentic:
                risk_factors.append({
                    "type": "ai_detection",
                    "severity": "high",
                    "confidence": confidence,
                    "description": f"AI模型检测为伪造，置信度: {confidence:.3f}"
                })
            
            return DetectionResult(
                is_authentic=is_authentic,
                confidence=confidence,
                risk_factors=risk_factors,
                processing_time=processing_time,
                detector_name=self.name
            )
            
        except Exception as e:
            self.logger.error(f"AI检测失败: {e}")
            return DetectionResult(
                is_authentic=True,
                confidence=0.5,
                risk_factors=[],
                processing_time=time.time() - start_time,
                detector_name=self.name
            )
    
    def _preprocess_for_model(self, image: np.ndarray) -> torch.Tensor:
        """为模型预处理图像"""
        import torch
        import torchvision.transforms as transforms
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image)
        
        # 定义变换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 应用变换
        tensor = transform(pil_image)
        tensor = tensor.unsqueeze(0)  # 添加batch维度
        tensor = tensor.to(self.device)
        
        return tensor


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FeatureExtractor")
    
    def extract_text_features(self, image: np.ndarray) -> Dict[str, Any]:
        """提取文字特征"""
        features = {}
        
        try:
            # 使用OCR提取文字
            import pytesseract
            
            # 转换为PIL图像
            pil_image = Image.fromarray(image)
            
            # 提取文字
            text = pytesseract.image_to_string(pil_image, lang='chi_sim+eng')
            
            # 分析文字特征
            features['text_length'] = len(text)
            features['text_density'] = len(text) / (image.shape[0] * image.shape[1])
            
            # 检测异常字符
            suspicious_chars = self._detect_suspicious_characters(text)
            features['suspicious_chars'] = suspicious_chars
            
            # 检测文字格式一致性
            format_consistency = self._check_text_format_consistency(text)
            features['format_consistency'] = format_consistency
            
        except Exception as e:
            self.logger.error(f"文字特征提取失败: {e}")
        
        return features
    
    def extract_ui_features(self, image: np.ndarray) -> Dict[str, Any]:
        """提取UI元素特征"""
        features = {}
        
        try:
            # 检测UI元素
            features['ui_elements'] = self._detect_ui_elements(image)
            
            # 检测对齐精度
            features['alignment_accuracy'] = self._check_alignment_accuracy(image)
            
            # 检测图标清晰度
            features['icon_clarity'] = self._check_icon_clarity(image)
            
        except Exception as e:
            self.logger.error(f"UI特征提取失败: {e}")
        
        return features
    
    def extract_content_logic_features(self, image: np.ndarray) -> Dict[str, Any]:
        """提取内容逻辑特征"""
        features = {}
        
        try:
            # 检测时间戳合理性
            features['timestamp_consistency'] = self._check_timestamp_consistency(image)
            
            # 检测数据逻辑一致性
            features['data_logic_consistency'] = self._check_data_logic_consistency(image)
            
            # 检测App版本匹配
            features['app_version_match'] = self._check_app_version_match(image)
            
        except Exception as e:
            self.logger.error(f"内容逻辑特征提取失败: {e}")
        
        return features
    
    def _detect_suspicious_characters(self, text: str) -> List[str]:
        """检测可疑字符"""
        suspicious = []
        
        # 检测特殊字符
        special_chars = ['@', '#', '$', '%', '&', '*', '!', '?']
        for char in special_chars:
            if char in text:
                suspicious.append(char)
        
        # 检测重复字符
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                suspicious.append(f"重复字符: {text[i]}")
        
        return suspicious
    
    def _check_text_format_consistency(self, text: str) -> float:
        """检查文字格式一致性"""
        # 这里可以实现更复杂的文字格式检查
        # 例如检查字体大小、颜色、对齐等
        return 0.8  # 示例返回值
    
    def _detect_ui_elements(self, image: np.ndarray) -> Dict[str, Any]:
        """检测UI元素"""
        elements = {}
        
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 检测按钮
            buttons = self._detect_buttons(gray)
            elements['buttons'] = buttons
            
            # 检测输入框
            input_fields = self._detect_input_fields(gray)
            elements['input_fields'] = input_fields
            
            # 检测图标
            icons = self._detect_icons(gray)
            elements['icons'] = icons
            
        except Exception as e:
            self.logger.error(f"UI元素检测失败: {e}")
        
        return elements
    
    def _detect_buttons(self, gray: np.ndarray) -> int:
        """检测按钮数量"""
        # 简化的按钮检测
        # 实际实现中可以使用更复杂的算法
        return 0
    
    def _detect_input_fields(self, gray: np.ndarray) -> int:
        """检测输入框数量"""
        # 简化的输入框检测
        return 0
    
    def _detect_icons(self, gray: np.ndarray) -> int:
        """检测图标数量"""
        # 简化的图标检测
        return 0
    
    def _check_alignment_accuracy(self, image: np.ndarray) -> float:
        """检查对齐精度"""
        # 简化的对齐检查
        return 0.8
    
    def _check_icon_clarity(self, image: np.ndarray) -> float:
        """检查图标清晰度"""
        # 简化的清晰度检查
        return 0.8
    
    def _check_timestamp_consistency(self, image: np.ndarray) -> float:
        """检查时间戳一致性"""
        # 简化的时间戳检查
        return 0.8
    
    def _check_data_logic_consistency(self, image: np.ndarray) -> float:
        """检查数据逻辑一致性"""
        # 简化的数据逻辑检查
        return 0.8
    
    def _check_app_version_match(self, image: np.ndarray) -> float:
        """检查App版本匹配"""
        # 简化的版本匹配检查
        return 0.8