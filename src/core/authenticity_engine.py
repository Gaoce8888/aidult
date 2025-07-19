"""
Core authenticity detection engine that integrates all detection methods
"""
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import logging
from pathlib import Path

# Import detection modules
from ..detectors.traditional_detector import TraditionalDetector
from ..detectors.metadata_detector import MetadataDetector
from ..models.deep_learning_models import DeepLearningDetector

logger = logging.getLogger(__name__)


class RiskAssessment:
    """Risk assessment and scoring system"""
    
    @staticmethod
    def calculate_risk_score(detections: Dict) -> Dict:
        """Calculate overall risk score from all detections"""
        
        # Weight factors for different detection methods
        weights = {
            'traditional': 0.25,
            'metadata': 0.20,
            'deep_learning': 0.45,
            'ui_analysis': 0.10
        }
        
        risk_scores = {}
        evidence = []
        
        # Traditional detection score
        if 'traditional' in detections and detections['traditional'].get('overall_traditional_score') is not None:
            risk_scores['traditional'] = detections['traditional']['overall_traditional_score']
            if detections['traditional']['traditional_suspicious']:
                evidence.append("Traditional image analysis detected anomalies")
        
        # Metadata detection score
        if 'metadata' in detections and detections['metadata'].get('overall_metadata_score') is not None:
            risk_scores['metadata'] = detections['metadata']['overall_metadata_score']
            if detections['metadata']['metadata_suspicious']:
                evidence.append("Metadata analysis found inconsistencies")
        
        # Deep learning score
        if 'deep_learning' in detections and detections['deep_learning'].get('fake_probability') is not None:
            risk_scores['deep_learning'] = detections['deep_learning']['fake_probability']
            if detections['deep_learning']['fake_probability'] > 0.5:
                evidence.append("AI models detected manipulation patterns")
        
        # UI analysis score (placeholder for future implementation)
        risk_scores['ui_analysis'] = 0.0
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        
        for method, score in risk_scores.items():
            if score is not None and method in weights:
                weighted_sum += score * weights[method]
                total_weight += weights[method]
        
        overall_risk = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Determine risk level
        if overall_risk < 0.3:
            risk_level = "LOW"
        elif overall_risk < 0.6:
            risk_level = "MEDIUM"
        elif overall_risk < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Calculate confidence based on agreement between methods
        confidence = RiskAssessment._calculate_confidence(risk_scores)
        
        return {
            'overall_risk_score': overall_risk,
            'risk_level': risk_level,
            'confidence': confidence,
            'individual_scores': risk_scores,
            'evidence': evidence,
            'weights_used': weights
        }
    
    @staticmethod
    def _calculate_confidence(scores: Dict) -> float:
        """Calculate confidence based on agreement between detection methods"""
        valid_scores = [score for score in scores.values() if score is not None]
        
        if len(valid_scores) < 2:
            return 0.5  # Low confidence with only one method
        
        # Calculate variance - lower variance means higher confidence
        mean_score = np.mean(valid_scores)
        variance = np.var(valid_scores)
        
        # Convert variance to confidence (inverse relationship)
        confidence = 1.0 / (1.0 + variance * 10)  # Scaling factor
        
        return min(max(confidence, 0.0), 1.0)


class PerformanceMonitor:
    """Monitor performance and timing of detection stages"""
    
    def __init__(self):
        self.stage_times = {}
        self.start_time = None
    
    def start_stage(self, stage_name: str):
        """Start timing a detection stage"""
        self.stage_times[stage_name] = {'start': time.time()}
    
    def end_stage(self, stage_name: str):
        """End timing a detection stage"""
        if stage_name in self.stage_times:
            self.stage_times[stage_name]['end'] = time.time()
            self.stage_times[stage_name]['duration'] = (
                self.stage_times[stage_name]['end'] - 
                self.stage_times[stage_name]['start']
            )
    
    def get_performance_report(self) -> Dict:
        """Get performance report"""
        total_time = 0
        stage_report = {}
        
        for stage, times in self.stage_times.items():
            if 'duration' in times:
                duration_ms = times['duration'] * 1000
                stage_report[stage] = {
                    'duration_ms': duration_ms,
                    'percentage': 0  # Will be calculated below
                }
                total_time += duration_ms
        
        # Calculate percentages
        for stage in stage_report:
            if total_time > 0:
                stage_report[stage]['percentage'] = (
                    stage_report[stage]['duration_ms'] / total_time * 100
                )
        
        return {
            'total_time_ms': total_time,
            'stage_breakdown': stage_report,
            'within_sla': {
                'stage1_target_ms': 50,
                'stage2_target_ms': 200,
                'overall_target_ms': 1000,
                'stage1_achieved': stage_report.get('basic_checks', {}).get('duration_ms', 0) <= 50,
                'stage2_achieved': stage_report.get('deep_analysis', {}).get('duration_ms', 0) <= 200,
                'overall_achieved': total_time <= 1000
            }
        }


class AuthenticityEngine:
    """
    Core engine for screenshot authenticity detection
    Implements multi-layer detection architecture with performance optimization
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize detection modules
        self._initialize_detectors()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Authenticity Engine initialized")
    
    def _initialize_detectors(self):
        """Initialize all detection modules"""
        try:
            # Traditional detector
            if self.config.get('enable_traditional_methods', True):
                self.traditional_detector = TraditionalDetector(self.config)
                logger.info("Traditional detector initialized")
            else:
                self.traditional_detector = None
            
            # Metadata detector
            if self.config.get('enable_metadata_analysis', True):
                self.metadata_detector = MetadataDetector(self.config)
                logger.info("Metadata detector initialized")
            else:
                self.metadata_detector = None
            
            # Deep learning detector
            if self.config.get('enable_deep_learning', True):
                self.deep_learning_detector = DeepLearningDetector(self.config)
                logger.info("Deep learning detector initialized")
            else:
                self.deep_learning_detector = None
            
        except Exception as e:
            logger.error(f"Error initializing detectors: {e}")
            raise
    
    async def analyze_screenshot(self, image_path: str, context: Optional[Dict] = None) -> Dict:
        """
        Comprehensive screenshot authenticity analysis
        
        Args:
            image_path: Path to screenshot image
            context: Additional context information (app type, device info, etc.)
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'error': 'Could not load image',
                    'authentic': True,
                    'confidence': 0.0
                }
            
            # Stage 1: Quick basic checks (< 50ms target)
            self.performance_monitor.start_stage('basic_checks')
            basic_results = await self._stage1_basic_checks(image, image_path)
            self.performance_monitor.end_stage('basic_checks')
            
            # Early exit if clearly authentic or fake
            if basic_results.get('early_exit'):
                return self._format_final_result(basic_results, context)
            
            # Stage 2: Deep analysis (< 200ms target)
            self.performance_monitor.start_stage('deep_analysis')
            deep_results = await self._stage2_deep_analysis(image, image_path, basic_results)
            self.performance_monitor.end_stage('deep_analysis')
            
            # Stage 3: Advanced verification if needed
            if self._needs_stage3_verification(deep_results):
                self.performance_monitor.start_stage('advanced_verification')
                advanced_results = await self._stage3_advanced_verification(image, image_path, deep_results)
                self.performance_monitor.end_stage('advanced_verification')
                
                # Merge results
                final_results = self._merge_results(basic_results, deep_results, advanced_results)
            else:
                final_results = self._merge_results(basic_results, deep_results)
            
            # Calculate final risk assessment
            risk_assessment = RiskAssessment.calculate_risk_score(final_results['detections'])
            final_results['risk_assessment'] = risk_assessment
            
            # Add performance metrics
            final_results['performance'] = self.performance_monitor.get_performance_report()
            
            # Format and return final result
            return self._format_final_result(final_results, context)
            
        except Exception as e:
            logger.error(f"Error in authenticity analysis: {e}")
            return {
                'error': str(e),
                'authentic': True,  # Default to authentic on error
                'confidence': 0.0,
                'analysis_time_ms': (time.time() - start_time) * 1000
            }
    
    async def _stage1_basic_checks(self, image: np.ndarray, image_path: str) -> Dict:
        """
        Stage 1: Basic checks for obvious indicators
        Target: < 50ms
        """
        results = {
            'stage': 1,
            'detections': {},
            'early_exit': False
        }
        
        # Basic image properties
        height, width = image.shape[:2]
        
        # Check for obviously suspicious characteristics
        suspicion_flags = []
        
        # Resolution check
        if width < 100 or height < 100:
            suspicion_flags.append("Image too small to be a screenshot")
            results['early_exit'] = True
            results['authentic'] = False
            results['confidence'] = 0.9
            
        # Aspect ratio check (common phone ratios)
        aspect_ratio = width / height
        common_ratios = [
            (16, 9), (18, 9), (19, 9), (20, 9),  # Modern phones
            (4, 3), (3, 2),  # Older devices/tablets
        ]
        
        ratio_match = any(
            abs(aspect_ratio - (w/h)) < 0.1 or abs(aspect_ratio - (h/w)) < 0.1
            for w, h in common_ratios
        )
        
        if not ratio_match:
            suspicion_flags.append("Unusual aspect ratio for mobile screenshot")
        
        # Quick format check
        if self.metadata_detector:
            format_check = self.metadata_detector._analyze_file_format(image_path)
            if format_check.get('suspicious', False):
                suspicion_flags.append("File format inconsistency")
        
        results['basic_checks'] = {
            'resolution': (width, height),
            'aspect_ratio': aspect_ratio,
            'ratio_match': ratio_match,
            'suspicion_flags': suspicion_flags
        }
        
        # Early exit for obviously suspicious cases
        if len(suspicion_flags) >= 2:
            results['early_exit'] = True
            results['authentic'] = False
            results['confidence'] = 0.8
        
        return results
    
    async def _stage2_deep_analysis(self, image: np.ndarray, image_path: str, basic_results: Dict) -> Dict:
        """
        Stage 2: Deep analysis using all detection methods
        Target: < 200ms
        """
        results = {
            'stage': 2,
            'detections': {}
        }
        
        # Run detections in parallel for better performance
        detection_futures = []
        
        # Traditional analysis
        if self.traditional_detector:
            future = self.executor.submit(self.traditional_detector.analyze, image)
            detection_futures.append(('traditional', future))
        
        # Metadata analysis
        if self.metadata_detector:
            future = self.executor.submit(self.metadata_detector.analyze, image_path)
            detection_futures.append(('metadata', future))
        
        # Deep learning analysis (lightweight model first)
        if self.deep_learning_detector:
            future = self.executor.submit(
                self.deep_learning_detector.predict, 
                image, 
                'efficientnet'  # Faster model for stage 2
            )
            detection_futures.append(('deep_learning', future))
        
        # Collect results
        for method_name, future in detection_futures:
            try:
                method_result = future.result(timeout=0.15)  # 150ms timeout per method
                results['detections'][method_name] = method_result
            except Exception as e:
                logger.warning(f"Error in {method_name} detection: {e}")
                results['detections'][method_name] = {'error': str(e), 'suspicious': False}
        
        return results
    
    async def _stage3_advanced_verification(self, image: np.ndarray, image_path: str, previous_results: Dict) -> Dict:
        """
        Stage 3: Advanced verification with ensemble models
        Target: < 1000ms total
        """
        results = {
            'stage': 3,
            'detections': {}
        }
        
        # Use ensemble model for final verification
        if self.deep_learning_detector:
            try:
                ensemble_result = self.deep_learning_detector.predict(image, 'ensemble')
                results['detections']['ensemble'] = ensemble_result
            except Exception as e:
                logger.error(f"Error in ensemble prediction: {e}")
                results['detections']['ensemble'] = {'error': str(e)}
        
        # Additional detailed analysis if needed
        if self._requires_detailed_analysis(previous_results):
            # Run more detailed traditional analysis
            if self.traditional_detector:
                detailed_traditional = self._detailed_traditional_analysis(image)
                results['detections']['detailed_traditional'] = detailed_traditional
        
        return results
    
    def _needs_stage3_verification(self, deep_results: Dict) -> bool:
        """Determine if stage 3 verification is needed"""
        detections = deep_results.get('detections', {})
        
        # Check if results are conflicting or uncertain
        suspicious_count = 0
        total_count = 0
        
        for method, result in detections.items():
            if not result.get('error'):
                total_count += 1
                if result.get('suspicious', False) or result.get('fake_probability', 0) > 0.5:
                    suspicious_count += 1
        
        if total_count == 0:
            return True  # Need more analysis if no valid results
        
        # If results are split or uncertain, need stage 3
        uncertainty_ratio = suspicious_count / total_count
        return 0.3 < uncertainty_ratio < 0.7  # Uncertain range
    
    def _requires_detailed_analysis(self, previous_results: Dict) -> bool:
        """Check if detailed analysis is required"""
        # Implementation for determining when detailed analysis is needed
        return True  # For now, always run detailed analysis in stage 3
    
    def _detailed_traditional_analysis(self, image: np.ndarray) -> Dict:
        """Run more detailed traditional analysis"""
        # This could include additional checks not done in the main analysis
        return {'detailed_analysis': 'not_implemented'}
    
    def _merge_results(self, *result_dicts) -> Dict:
        """Merge results from different stages"""
        merged = {
            'detections': {},
            'stages_completed': []
        }
        
        for result_dict in result_dicts:
            if result_dict:
                merged['stages_completed'].append(result_dict.get('stage', 'unknown'))
                merged['detections'].update(result_dict.get('detections', {}))
        
        return merged
    
    def _format_final_result(self, analysis_results: Dict, context: Optional[Dict] = None) -> Dict:
        """Format the final analysis result"""
        
        # Get risk assessment
        risk_assessment = analysis_results.get('risk_assessment', {})
        
        # Determine final authentication status
        risk_score = risk_assessment.get('overall_risk_score', 0.0)
        confidence = risk_assessment.get('confidence', 0.5)
        
        # Decision logic
        if risk_score < 0.3:
            authentic = True
            decision_confidence = confidence
        elif risk_score > 0.7:
            authentic = False
            decision_confidence = confidence
        else:
            # Uncertain case - lean towards authentic unless high confidence in fake
            authentic = risk_score < 0.5 or confidence < 0.7
            decision_confidence = max(0.3, confidence * 0.8)  # Reduce confidence for uncertain cases
        
        # Prepare risk factors for detailed reporting
        risk_factors = []
        detections = analysis_results.get('detections', {})
        
        for method, result in detections.items():
            if isinstance(result, dict) and result.get('suspicious', False):
                risk_factors.append({
                    'method': method,
                    'description': f"Suspicious patterns detected by {method}",
                    'severity': self._calculate_severity(result),
                    'details': result
                })
        
        # Performance metrics
        performance = analysis_results.get('performance', {})
        
        # Final result structure
        final_result = {
            'authentic': authentic,
            'confidence': decision_confidence,
            'risk_assessment': risk_assessment,
            'risk_factors': risk_factors,
            'detection_summary': {
                'methods_used': list(detections.keys()),
                'stages_completed': analysis_results.get('stages_completed', []),
                'total_analysis_time_ms': performance.get('total_time_ms', 0),
                'performance_sla_met': performance.get('within_sla', {}).get('overall_achieved', False)
            },
            'detailed_results': detections,
            'context': context or {},
            'timestamp': time.time()
        }
        
        # Add performance metrics if available
        if performance:
            final_result['performance_metrics'] = performance
        
        return final_result
    
    def _calculate_severity(self, detection_result: Dict) -> str:
        """Calculate severity level for a detection result"""
        if detection_result.get('error'):
            return 'unknown'
        
        # Check various indicators for severity
        indicators = []
        
        if detection_result.get('fake_probability', 0) > 0.8:
            indicators.append('high')
        elif detection_result.get('fake_probability', 0) > 0.6:
            indicators.append('medium')
        
        if detection_result.get('suspicious', False):
            indicators.append('medium')
        
        # Traditional detection specific
        if 'traditional_suspicious' in detection_result and detection_result['traditional_suspicious']:
            indicators.append('medium')
        
        # Metadata specific
        if 'metadata_suspicious' in detection_result and detection_result['metadata_suspicious']:
            indicators.append('low')
        
        # Determine overall severity
        if 'high' in indicators:
            return 'high'
        elif 'medium' in indicators:
            return 'medium'
        else:
            return 'low'
    
    def get_system_status(self) -> Dict:
        """Get system status and health information"""
        status = {
            'detectors': {},
            'performance': {},
            'configuration': {}
        }
        
        # Check detector status
        status['detectors']['traditional'] = self.traditional_detector is not None
        status['detectors']['metadata'] = self.metadata_detector is not None
        status['detectors']['deep_learning'] = self.deep_learning_detector is not None
        
        # Configuration summary
        status['configuration'] = {
            'enable_traditional_methods': self.config.get('enable_traditional_methods', True),
            'enable_metadata_analysis': self.config.get('enable_metadata_analysis', True),
            'enable_deep_learning': self.config.get('enable_deep_learning', True),
            'enable_ensemble': self.config.get('enable_ensemble', True)
        }
        
        return status
    
    def close(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("Authenticity Engine closed")