"""
验证引擎
整合所有检测器并实现融合决策
"""
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .detectors import (
    TraditionalDetector, 
    MetadataDetector, 
    AIDetector, 
    FeatureExtractor,
    DetectionResult
)
from config.settings import settings


@dataclass
class VerificationResult:
    """验证结果数据类"""
    is_authentic: bool
    confidence: float
    risk_factors: List[Dict[str, Any]]
    processing_time: float
    detector_results: Dict[str, DetectionResult]
    detailed_report: Dict[str, Any]


class VerificationEngine:
    """验证引擎主类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 初始化检测器
        self.detectors = {}
        self._initialize_detectors()
        
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("验证引擎初始化完成")
    
    def _initialize_detectors(self):
        """初始化所有检测器"""
        try:
            # 传统图像分析检测器
            if settings.traditional_detector_enabled:
                self.detectors['traditional'] = TraditionalDetector()
                self.logger.info("传统检测器初始化完成")
            
            # 元数据分析检测器
            if settings.metadata_detector_enabled:
                self.detectors['metadata'] = MetadataDetector()
                self.logger.info("元数据检测器初始化完成")
            
            # AI深度学习检测器
            if settings.ai_detector_enabled:
                self.detectors['ai'] = AIDetector(settings.efficientnet_model_path)
                self.logger.info("AI检测器初始化完成")
            
        except Exception as e:
            self.logger.error(f"检测器初始化失败: {e}")
    
    def verify_screenshot(self, image: np.ndarray, metadata: Dict[str, Any]) -> VerificationResult:
        """验证截图真伪"""
        start_time = time.time()
        
        try:
            # 第一阶段：快速筛选
            quick_result = self._quick_screening(image, metadata)
            if quick_result:
                return quick_result
            
            # 第二阶段：并行检测
            detector_results = self._parallel_detection(image, metadata)
            
            # 第三阶段：特征提取
            features = self._extract_features(image)
            
            # 第四阶段：融合决策
            final_result = self._fusion_decision(detector_results, features, metadata)
            
            # 第五阶段：生成详细报告
            detailed_report = self._generate_detailed_report(
                detector_results, features, metadata, final_result
            )
            
            processing_time = time.time() - start_time
            
            return VerificationResult(
                is_authentic=final_result['is_authentic'],
                confidence=final_result['confidence'],
                risk_factors=final_result['risk_factors'],
                processing_time=processing_time,
                detector_results=detector_results,
                detailed_report=detailed_report
            )
            
        except Exception as e:
            self.logger.error(f"验证过程失败: {e}")
            return self._create_error_result(start_time)
    
    def _quick_screening(self, image: np.ndarray, metadata: Dict[str, Any]) -> Optional[VerificationResult]:
        """快速筛选阶段（<50ms）"""
        try:
            # 检查图像大小
            if image.nbytes > settings.max_image_size:
                return VerificationResult(
                    is_authentic=False,
                    confidence=0.9,
                    risk_factors=[{
                        "type": "file_size",
                        "severity": "high",
                        "confidence": 0.9,
                        "description": f"图像文件过大: {image.nbytes / 1024 / 1024:.1f}MB"
                    }],
                    processing_time=0.01,
                    detector_results={},
                    detailed_report={"error": "图像文件过大"}
                )
            
            # 检查分辨率
            h, w = image.shape[:2]
            if w > 10000 or h > 10000:
                return VerificationResult(
                    is_authentic=False,
                    confidence=0.8,
                    risk_factors=[{
                        "type": "resolution",
                        "severity": "medium",
                        "confidence": 0.8,
                        "description": f"分辨率异常: {w}x{h}"
                    }],
                    processing_time=0.01,
                    detector_results={},
                    detailed_report={"error": "分辨率异常"}
                )
            
            # 检查格式
            if len(image.shape) != 3 or image.shape[2] not in [1, 3, 4]:
                return VerificationResult(
                    is_authentic=False,
                    confidence=0.7,
                    risk_factors=[{
                        "type": "format",
                        "severity": "medium",
                        "confidence": 0.7,
                        "description": f"图像格式异常: {image.shape}"
                    }],
                    processing_time=0.01,
                    detector_results={},
                    detailed_report={"error": "图像格式异常"}
                )
            
        except Exception as e:
            self.logger.error(f"快速筛选失败: {e}")
        
        return None
    
    def _parallel_detection(self, image: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, DetectionResult]:
        """并行检测阶段"""
        detector_results = {}
        
        # 提交检测任务
        future_to_detector = {}
        for name, detector in self.detectors.items():
            future = self.executor.submit(detector.detect, image, metadata)
            future_to_detector[future] = name
        
        # 收集结果
        for future in as_completed(future_to_detector):
            detector_name = future_to_detector[future]
            try:
                result = future.result()
                detector_results[detector_name] = result
                self.logger.debug(f"{detector_name} 检测完成，耗时: {result.processing_time:.3f}s")
            except Exception as e:
                self.logger.error(f"{detector_name} 检测失败: {e}")
                # 创建默认结果
                detector_results[detector_name] = DetectionResult(
                    is_authentic=True,
                    confidence=0.5,
                    risk_factors=[],
                    processing_time=0.0,
                    detector_name=detector_name
                )
        
        return detector_results
    
    def _extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """特征提取阶段"""
        features = {}
        
        try:
            # 文字特征提取
            if settings.text_detection_enabled:
                features['text'] = self.feature_extractor.extract_text_features(image)
            
            # UI元素特征提取
            if settings.ui_element_detection_enabled:
                features['ui'] = self.feature_extractor.extract_ui_features(image)
            
            # 内容逻辑特征提取
            if settings.content_logic_detection_enabled:
                features['content_logic'] = self.feature_extractor.extract_content_logic_features(image)
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
        
        return features
    
    def _fusion_decision(self, detector_results: Dict[str, DetectionResult], 
                        features: Dict[str, Any], 
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """融合决策阶段"""
        try:
            # 收集所有检测结果
            all_risk_factors = []
            total_confidence = 0.0
            valid_detectors = 0
            
            for detector_name, result in detector_results.items():
                if result.confidence > 0:
                    all_risk_factors.extend(result.risk_factors)
                    total_confidence += result.confidence
                    valid_detectors += 1
            
            # 计算平均置信度
            avg_confidence = total_confidence / max(valid_detectors, 1)
            
            # 基于风险因子调整置信度
            adjusted_confidence = self._adjust_confidence_by_risk_factors(
                avg_confidence, all_risk_factors
            )
            
            # 基于特征调整置信度
            final_confidence = self._adjust_confidence_by_features(
                adjusted_confidence, features
            )
            
            # 判断真伪
            is_authentic = final_confidence > settings.confidence_threshold
            
            # 合并风险因子
            merged_risk_factors = self._merge_risk_factors(all_risk_factors)
            
            return {
                'is_authentic': is_authentic,
                'confidence': final_confidence,
                'risk_factors': merged_risk_factors
            }
            
        except Exception as e:
            self.logger.error(f"融合决策失败: {e}")
            return {
                'is_authentic': True,
                'confidence': 0.5,
                'risk_factors': []
            }
    
    def _adjust_confidence_by_risk_factors(self, confidence: float, 
                                         risk_factors: List[Dict[str, Any]]) -> float:
        """基于风险因子调整置信度"""
        adjusted_confidence = confidence
        
        for risk_factor in risk_factors:
            severity = risk_factor.get('severity', 'low')
            risk_confidence = risk_factor.get('confidence', 0.0)
            
            # 根据严重程度调整置信度
            if severity == 'critical':
                adjusted_confidence *= (1.0 - risk_confidence * 0.5)
            elif severity == 'high':
                adjusted_confidence *= (1.0 - risk_confidence * 0.3)
            elif severity == 'medium':
                adjusted_confidence *= (1.0 - risk_confidence * 0.2)
            elif severity == 'low':
                adjusted_confidence *= (1.0 - risk_confidence * 0.1)
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _adjust_confidence_by_features(self, confidence: float, 
                                     features: Dict[str, Any]) -> float:
        """基于特征调整置信度"""
        adjusted_confidence = confidence
        
        try:
            # 文字特征调整
            if 'text' in features:
                text_features = features['text']
                
                # 检测可疑字符
                suspicious_chars = text_features.get('suspicious_chars', [])
                if suspicious_chars:
                    adjusted_confidence *= (1.0 - len(suspicious_chars) * 0.1)
                
                # 文字格式一致性
                format_consistency = text_features.get('format_consistency', 0.8)
                adjusted_confidence *= format_consistency
            
            # UI特征调整
            if 'ui' in features:
                ui_features = features['ui']
                
                # 对齐精度
                alignment_accuracy = ui_features.get('alignment_accuracy', 0.8)
                adjusted_confidence *= alignment_accuracy
                
                # 图标清晰度
                icon_clarity = ui_features.get('icon_clarity', 0.8)
                adjusted_confidence *= icon_clarity
            
            # 内容逻辑特征调整
            if 'content_logic' in features:
                content_features = features['content_logic']
                
                # 时间戳一致性
                timestamp_consistency = content_features.get('timestamp_consistency', 0.8)
                adjusted_confidence *= timestamp_consistency
                
                # 数据逻辑一致性
                data_logic_consistency = content_features.get('data_logic_consistency', 0.8)
                adjusted_confidence *= data_logic_consistency
                
                # App版本匹配
                app_version_match = content_features.get('app_version_match', 0.8)
                adjusted_confidence *= app_version_match
            
        except Exception as e:
            self.logger.error(f"特征调整失败: {e}")
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _merge_risk_factors(self, risk_factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并风险因子"""
        if not risk_factors:
            return []
        
        # 按类型分组
        risk_groups = {}
        for risk_factor in risk_factors:
            risk_type = risk_factor.get('type', 'unknown')
            if risk_type not in risk_groups:
                risk_groups[risk_type] = []
            risk_groups[risk_type].append(risk_factor)
        
        # 合并同类型的风险因子
        merged_risks = []
        for risk_type, risks in risk_groups.items():
            if len(risks) == 1:
                merged_risks.append(risks[0])
            else:
                # 合并多个同类型风险因子
                max_severity = max(risk.get('severity', 'low') for risk in risks)
                avg_confidence = np.mean([risk.get('confidence', 0.0) for risk in risks])
                descriptions = [risk.get('description', '') for risk in risks]
                
                merged_risks.append({
                    'type': risk_type,
                    'severity': max_severity,
                    'confidence': avg_confidence,
                    'description': f"多个{risk_type}风险: {'; '.join(descriptions)}"
                })
        
        return merged_risks
    
    def _generate_detailed_report(self, detector_results: Dict[str, DetectionResult],
                                features: Dict[str, Any],
                                metadata: Dict[str, Any],
                                final_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成详细报告"""
        report = {
            'summary': {
                'is_authentic': final_result['is_authentic'],
                'confidence': final_result['confidence'],
                'risk_level': self._calculate_risk_level(final_result['risk_factors'])
            },
            'detector_results': {},
            'features': features,
            'metadata': metadata,
            'risk_factors': final_result['risk_factors'],
            'recommendations': self._generate_recommendations(final_result)
        }
        
        # 添加各检测器的详细结果
        for detector_name, result in detector_results.items():
            report['detector_results'][detector_name] = {
                'is_authentic': result.is_authentic,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'risk_factors': result.risk_factors
            }
        
        return report
    
    def _calculate_risk_level(self, risk_factors: List[Dict[str, Any]]) -> str:
        """计算风险等级"""
        if not risk_factors:
            return 'low'
        
        severity_scores = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        
        max_score = max(severity_scores.get(risk.get('severity', 'low'), 1) 
                       for risk in risk_factors)
        
        if max_score >= 4:
            return 'critical'
        elif max_score >= 3:
            return 'high'
        elif max_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, final_result: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if not final_result['is_authentic']:
            recommendations.append("建议进行人工复核")
            
            risk_factors = final_result['risk_factors']
            for risk in risk_factors:
                risk_type = risk.get('type', '')
                if 'compression' in risk_type:
                    recommendations.append("检测到压缩痕迹，建议检查图像来源")
                elif 'metadata' in risk_type:
                    recommendations.append("元数据异常，建议验证图像真实性")
                elif 'ai_detection' in risk_type:
                    recommendations.append("AI检测为伪造，建议进一步验证")
        
        if final_result['confidence'] < 0.6:
            recommendations.append("置信度较低，建议使用其他验证方法")
        
        return recommendations
    
    def _create_error_result(self, start_time: float) -> VerificationResult:
        """创建错误结果"""
        return VerificationResult(
            is_authentic=True,
            confidence=0.5,
            risk_factors=[{
                "type": "system_error",
                "severity": "medium",
                "confidence": 0.5,
                "description": "系统处理错误"
            }],
            processing_time=time.time() - start_time,
            detector_results={},
            detailed_report={"error": "系统处理错误"}
        )
    
    def get_detector_status(self) -> Dict[str, Any]:
        """获取检测器状态"""
        status = {}
        for name, detector in self.detectors.items():
            status[name] = {
                'enabled': True,
                'type': detector.__class__.__name__
            }
        return status
    
    def shutdown(self):
        """关闭验证引擎"""
        self.executor.shutdown(wait=True)
        self.logger.info("验证引擎已关闭")