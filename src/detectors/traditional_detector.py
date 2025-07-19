"""
Traditional image analysis detector for screenshot authenticity verification
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from skimage import feature, measure, filters
from skimage.feature import greycomatrix, greycoprops
import logging

logger = logging.getLogger(__name__)


class TraditionalDetector:
    """
    Traditional image analysis methods for detecting fake screenshots
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.jpeg_quality_threshold = self.config.get('jpeg_quality_threshold', 0.8)
        self.noise_variance_threshold = self.config.get('noise_variance_threshold', 0.1)
        self.edge_sharpness_threshold = self.config.get('edge_sharpness_threshold', 0.5)
    
    def detect_compression_artifacts(self, image: np.ndarray) -> Dict:
        """
        Detect JPEG compression artifacts and double compression
        
        Args:
            image: Input image array (BGR format)
            
        Returns:
            Dictionary with compression analysis results
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect blocking artifacts
            block_artifacts = self._detect_blocking_artifacts(gray)
            
            # Detect double compression
            double_compression = self._detect_double_compression(gray)
            
            # Estimate JPEG quality
            quality_score = self._estimate_jpeg_quality(gray)
            
            # Calculate compression inconsistency
            compression_inconsistency = self._detect_compression_inconsistency(image)
            
            result = {
                'block_artifacts_score': block_artifacts,
                'double_compression_score': double_compression,
                'quality_estimate': quality_score,
                'compression_inconsistency': compression_inconsistency,
                'suspicious': (
                    block_artifacts > 0.7 or 
                    double_compression > 0.6 or 
                    compression_inconsistency > 0.5
                )
            }
            
            logger.debug(f"Compression artifacts detection: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in compression artifacts detection: {e}")
            return {'error': str(e), 'suspicious': False}
    
    def detect_noise_patterns(self, image: np.ndarray) -> Dict:
        """
        Detect inconsistent noise patterns that indicate tampering
        
        Args:
            image: Input image array (BGR format)
            
        Returns:
            Dictionary with noise analysis results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Analyze noise distribution
            noise_variance = self._analyze_noise_variance(gray)
            
            # Detect local noise inconsistencies
            local_inconsistencies = self._detect_local_noise_inconsistencies(gray)
            
            # Analyze noise texture patterns
            texture_analysis = self._analyze_noise_texture(gray)
            
            # Statistical distribution analysis
            distribution_analysis = self._analyze_noise_distribution(gray)
            
            result = {
                'noise_variance': noise_variance,
                'local_inconsistencies': local_inconsistencies,
                'texture_score': texture_analysis,
                'distribution_score': distribution_analysis,
                'suspicious': (
                    local_inconsistencies > 0.6 or
                    abs(noise_variance - 0.05) > self.noise_variance_threshold
                )
            }
            
            logger.debug(f"Noise patterns detection: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in noise patterns detection: {e}")
            return {'error': str(e), 'suspicious': False}
    
    def detect_edge_inconsistency(self, image: np.ndarray) -> Dict:
        """
        Detect edge inconsistencies that suggest tampering
        
        Args:
            image: Input image array (BGR format)
            
        Returns:
            Dictionary with edge analysis results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge sharpness analysis
            edge_sharpness = self._analyze_edge_sharpness(gray)
            
            # Detect unnatural boundaries
            boundary_analysis = self._detect_unnatural_boundaries(gray)
            
            # Analyze edge continuity
            continuity_score = self._analyze_edge_continuity(gray)
            
            # Detect mixing artifacts
            mixing_artifacts = self._detect_mixing_artifacts(image)
            
            result = {
                'edge_sharpness': edge_sharpness,
                'boundary_score': boundary_analysis,
                'continuity_score': continuity_score,
                'mixing_artifacts': mixing_artifacts,
                'suspicious': (
                    boundary_analysis > 0.7 or
                    mixing_artifacts > 0.6 or
                    edge_sharpness < self.edge_sharpness_threshold
                )
            }
            
            logger.debug(f"Edge inconsistency detection: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in edge inconsistency detection: {e}")
            return {'error': str(e), 'suspicious': False}
    
    def _detect_blocking_artifacts(self, gray: np.ndarray) -> float:
        """Detect JPEG blocking artifacts"""
        h, w = gray.shape
        
        # Calculate horizontal and vertical differences
        h_diff = np.abs(gray[:, 1:] - gray[:, :-1])
        v_diff = np.abs(gray[1:, :] - gray[:-1, :])
        
        # Focus on 8x8 block boundaries
        h_block_diff = h_diff[:, 7::8]
        v_block_diff = v_diff[7::8, :]
        
        # Calculate blocking artifact score
        h_block_score = np.mean(h_block_diff) if h_block_diff.size > 0 else 0
        v_block_score = np.mean(v_block_diff) if v_block_diff.size > 0 else 0
        
        overall_diff = np.mean([np.mean(h_diff), np.mean(v_diff)])
        
        if overall_diff > 0:
            blocking_score = (h_block_score + v_block_score) / (2 * overall_diff)
        else:
            blocking_score = 0
        
        return min(blocking_score, 1.0)
    
    def _detect_double_compression(self, gray: np.ndarray) -> float:
        """Detect double JPEG compression"""
        # Apply DCT to 8x8 blocks
        h, w = gray.shape
        dct_coeffs = []
        
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                dct_coeffs.append(dct_block)
        
        if not dct_coeffs:
            return 0.0
        
        # Analyze coefficient distribution for double compression signatures
        all_coeffs = np.array(dct_coeffs).flatten()
        hist, bins = np.histogram(all_coeffs, bins=100, range=(-50, 50))
        
        # Look for periodic patterns in histogram
        peaks = self._find_peaks(hist)
        
        # Double compression typically shows periodic patterns
        if len(peaks) > 2:
            peak_distances = np.diff(peaks)
            periodic_score = 1.0 - np.std(peak_distances) / (np.mean(peak_distances) + 1e-6)
        else:
            periodic_score = 0.0
        
        return min(periodic_score, 1.0)
    
    def _estimate_jpeg_quality(self, gray: np.ndarray) -> float:
        """Estimate JPEG quality from image"""
        # Apply DCT and analyze quantization patterns
        h, w = gray.shape
        total_variation = 0
        block_count = 0
        
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # Calculate high-frequency energy
                hf_energy = np.sum(np.abs(dct_block[4:, 4:]))
                total_variation += hf_energy
                block_count += 1
        
        if block_count > 0:
            avg_hf_energy = total_variation / block_count
            # Normalize to 0-1 range (higher energy = higher quality)
            quality_estimate = min(avg_hf_energy / 100.0, 1.0)
        else:
            quality_estimate = 0.5
        
        return quality_estimate
    
    def _detect_compression_inconsistency(self, image: np.ndarray) -> float:
        """Detect compression inconsistencies across image regions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Divide image into regions
        region_size = min(h, w) // 4
        qualities = []
        
        for i in range(0, h - region_size, region_size // 2):
            for j in range(0, w - region_size, region_size // 2):
                region = gray[i:i+region_size, j:j+region_size]
                quality = self._estimate_jpeg_quality(region)
                qualities.append(quality)
        
        # Calculate inconsistency as standard deviation
        if len(qualities) > 1:
            inconsistency = np.std(qualities) / (np.mean(qualities) + 1e-6)
        else:
            inconsistency = 0.0
        
        return min(inconsistency, 1.0)
    
    def _analyze_noise_variance(self, gray: np.ndarray) -> float:
        """Analyze noise variance in the image"""
        # Apply Gaussian blur and calculate noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float32) - blurred.astype(np.float32)
        
        # Calculate variance
        noise_variance = np.var(noise)
        
        # Normalize to reasonable range
        return min(noise_variance / 100.0, 1.0)
    
    def _detect_local_noise_inconsistencies(self, gray: np.ndarray) -> float:
        """Detect local noise inconsistencies"""
        h, w = gray.shape
        region_size = 64
        noise_variances = []
        
        for i in range(0, h - region_size, region_size // 2):
            for j in range(0, w - region_size, region_size // 2):
                region = gray[i:i+region_size, j:j+region_size]
                
                # Calculate local noise
                blurred = cv2.GaussianBlur(region, (3, 3), 0)
                noise = region.astype(np.float32) - blurred.astype(np.float32)
                
                noise_variances.append(np.var(noise))
        
        # Calculate inconsistency
        if len(noise_variances) > 1:
            inconsistency = np.std(noise_variances) / (np.mean(noise_variances) + 1e-6)
        else:
            inconsistency = 0.0
        
        return min(inconsistency, 1.0)
    
    def _analyze_noise_texture(self, gray: np.ndarray) -> float:
        """Analyze noise texture patterns using GLCM"""
        # Resize for computational efficiency
        resized = cv2.resize(gray, (256, 256))
        
        # Calculate GLCM
        glcm = greycomatrix(resized, [1], [0, 45, 90, 135], levels=256, symmetric=True, normed=True)
        
        # Calculate texture properties
        contrast = greycoprops(glcm, 'contrast')[0].mean()
        homogeneity = greycoprops(glcm, 'homogeneity')[0].mean()
        energy = greycoprops(glcm, 'energy')[0].mean()
        
        # Combine properties into texture score
        texture_score = (contrast + (1 - homogeneity) + energy) / 3
        
        return min(texture_score, 1.0)
    
    def _analyze_noise_distribution(self, gray: np.ndarray) -> float:
        """Analyze noise distribution statistics"""
        # Calculate noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float32) - blurred.astype(np.float32)
        
        # Calculate distribution properties
        noise_flat = noise.flatten()
        skewness = self._calculate_skewness(noise_flat)
        kurtosis = self._calculate_kurtosis(noise_flat)
        
        # Natural noise should be close to Gaussian (skewness~0, kurtosis~3)
        deviation_score = abs(skewness) + abs(kurtosis - 3) / 3
        
        return min(deviation_score, 1.0)
    
    def _analyze_edge_sharpness(self, gray: np.ndarray) -> float:
        """Analyze edge sharpness"""
        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate edge magnitude
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Calculate sharpness as mean edge strength
        sharpness = np.mean(edge_magnitude) / 255.0
        
        return min(sharpness, 1.0)
    
    def _detect_unnatural_boundaries(self, gray: np.ndarray) -> float:
        """Detect unnatural boundaries"""
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Analyze contour properties
        unnatural_score = 0.0
        for contour in contours:
            if len(contour) > 10:  # Skip very small contours
                # Calculate contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    # Roundness (1 = perfect circle, 0 = line)
                    roundness = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Very low roundness might indicate artificial edges
                    if roundness < 0.1:
                        unnatural_score += 0.1
        
        return min(unnatural_score, 1.0)
    
    def _analyze_edge_continuity(self, gray: np.ndarray) -> float:
        """Analyze edge continuity"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Find line segments using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return 0.5
        
        # Analyze line continuity
        continuity_score = 0.0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Longer lines suggest better continuity
            continuity_score += min(length / 100.0, 1.0)
        
        # Normalize by number of lines
        avg_continuity = continuity_score / len(lines) if lines is not None else 0.5
        
        return min(avg_continuity, 1.0)
    
    def _detect_mixing_artifacts(self, image: np.ndarray) -> float:
        """Detect artifacts from image mixing/compositing"""
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Analyze color transitions
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Calculate color gradients
        l_grad = np.gradient(l_channel.astype(np.float32))
        a_grad = np.gradient(a_channel.astype(np.float32))
        b_grad = np.gradient(b_channel.astype(np.float32))
        
        # Look for abrupt color transitions
        l_abrupt = np.mean(np.abs(l_grad[0]) + np.abs(l_grad[1]))
        a_abrupt = np.mean(np.abs(a_grad[0]) + np.abs(a_grad[1]))
        b_abrupt = np.mean(np.abs(b_grad[0]) + np.abs(b_grad[1]))
        
        # Combine scores
        mixing_score = (l_abrupt + a_abrupt + b_abrupt) / (3 * 255.0)
        
        return min(mixing_score, 1.0)
    
    def _find_peaks(self, data: np.ndarray, min_distance: int = 5) -> List[int]:
        """Find peaks in 1D data"""
        peaks = []
        for i in range(min_distance, len(data) - min_distance):
            if all(data[i] > data[i-j] for j in range(1, min_distance + 1)) and \
               all(data[i] > data[i+j] for j in range(1, min_distance + 1)):
                peaks.append(i)
        return peaks
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4)
        return kurtosis
    
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Comprehensive traditional analysis
        
        Args:
            image: Input image array (BGR format)
            
        Returns:
            Dictionary with all analysis results
        """
        results = {
            'compression_artifacts': self.detect_compression_artifacts(image),
            'noise_patterns': self.detect_noise_patterns(image),
            'edge_inconsistency': self.detect_edge_inconsistency(image)
        }
        
        # Calculate overall suspicion score
        suspicion_scores = []
        for analysis in results.values():
            if isinstance(analysis.get('suspicious'), bool):
                suspicion_scores.append(1.0 if analysis['suspicious'] else 0.0)
        
        overall_suspicion = np.mean(suspicion_scores) if suspicion_scores else 0.0
        
        results['overall_traditional_score'] = overall_suspicion
        results['traditional_suspicious'] = overall_suspicion > 0.5
        
        return results