"""
Metadata analysis detector for screenshot authenticity verification
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import logging

logger = logging.getLogger(__name__)


class MetadataDetector:
    """
    Metadata analysis for detecting fake screenshots
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Known device patterns
        self.device_patterns = {
            'android': {
                'manufacturers': ['samsung', 'huawei', 'xiaomi', 'oppo', 'vivo', 'oneplus', 'google'],
                'models': ['galaxy', 'mate', 'mi', 'redmi', 'find', 'pixel'],
                'software_patterns': ['android', 'miui', 'emui', 'coloros', 'funtouch', 'oxygenos']
            },
            'ios': {
                'manufacturers': ['apple'],
                'models': ['iphone', 'ipad'],
                'software_patterns': ['ios', 'ipados']
            }
        }
        
        # Screenshot app signatures
        self.screenshot_app_signatures = {
            'native_android': ['screenshot', 'screen capture', 'screencapture'],
            'native_ios': ['screenshot'],
            'third_party': ['snapseed', 'lightshot', 'greenshot', 'puush', 'gyazo'],
            'editing_apps': ['photoshop', 'gimp', 'canva', 'figma', 'sketch']
        }
    
    def extract_exif_data(self, image_path: str) -> Dict:
        """
        Extract and analyze EXIF data from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with EXIF analysis results
        """
        try:
            # Open image and extract EXIF
            with Image.open(image_path) as img:
                exif_dict = img._getexif()
                
            if not exif_dict:
                return {
                    'has_exif': False,
                    'suspicious': True,
                    'reason': 'No EXIF data found - unusual for phone screenshots'
                }
            
            # Convert EXIF data to readable format
            exif_data = {}
            for tag_id, value in exif_dict.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_data[tag_name] = value
            
            # Analyze EXIF data
            analysis_result = self._analyze_exif_data(exif_data)
            analysis_result['raw_exif'] = exif_data
            analysis_result['has_exif'] = True
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error extracting EXIF data: {e}")
            return {
                'has_exif': False,
                'error': str(e),
                'suspicious': False
            }
    
    def analyze_file_properties(self, image_path: str) -> Dict:
        """
        Analyze file properties and timestamps
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with file analysis results
        """
        try:
            file_path = Path(image_path)
            
            # Get file stats
            stat = file_path.stat()
            
            # Get file timestamps
            creation_time = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            modification_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            access_time = datetime.fromtimestamp(stat.st_atime, tz=timezone.utc)
            
            # Analyze file properties
            file_size = stat.st_size
            file_extension = file_path.suffix.lower()
            
            # Check for suspicious patterns
            suspicious_indicators = []
            
            # Check file size (screenshots are usually within certain ranges)
            if file_size < 10000:  # Very small files
                suspicious_indicators.append("Unusually small file size")
            elif file_size > 50000000:  # Very large files (50MB+)
                suspicious_indicators.append("Unusually large file size")
            
            # Check timestamp consistency
            time_diff = abs((modification_time - creation_time).total_seconds())
            if time_diff > 300:  # More than 5 minutes difference
                suspicious_indicators.append("Large gap between creation and modification times")
            
            # Check file extension consistency with content
            format_analysis = self._analyze_file_format(image_path)
            
            result = {
                'file_size': file_size,
                'extension': file_extension,
                'creation_time': creation_time.isoformat(),
                'modification_time': modification_time.isoformat(),
                'access_time': access_time.isoformat(),
                'time_difference_seconds': time_diff,
                'format_analysis': format_analysis,
                'suspicious_indicators': suspicious_indicators,
                'suspicious': len(suspicious_indicators) > 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing file properties: {e}")
            return {
                'error': str(e),
                'suspicious': False
            }
    
    def calculate_image_fingerprint(self, image_path: str) -> Dict:
        """
        Calculate various image fingerprints and hashes
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with fingerprint analysis results
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not read image', 'suspicious': False}
            
            # Calculate various hashes
            fingerprints = {}
            
            # File hash
            with open(image_path, 'rb') as f:
                file_content = f.read()
                fingerprints['md5'] = hashlib.md5(file_content).hexdigest()
                fingerprints['sha256'] = hashlib.sha256(file_content).hexdigest()
            
            # Image content hashes
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Average hash (aHash)
            fingerprints['average_hash'] = self._calculate_average_hash(gray)
            
            # Perceptual hash (pHash)
            fingerprints['perceptual_hash'] = self._calculate_perceptual_hash(gray)
            
            # Difference hash (dHash)
            fingerprints['difference_hash'] = self._calculate_difference_hash(gray)
            
            # Wavelet hash
            fingerprints['wavelet_hash'] = self._calculate_wavelet_hash(gray)
            
            # Color histogram fingerprint
            fingerprints['color_histogram'] = self._calculate_color_histogram(image)
            
            return {
                'fingerprints': fingerprints,
                'suspicious': False  # Fingerprints themselves don't indicate suspicion
            }
            
        except Exception as e:
            logger.error(f"Error calculating image fingerprint: {e}")
            return {
                'error': str(e),
                'suspicious': False
            }
    
    def detect_steganography(self, image_path: str) -> Dict:
        """
        Detect potential steganography or hidden information
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with steganography analysis results
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not read image', 'suspicious': False}
            
            # LSB analysis
            lsb_analysis = self._analyze_lsb_patterns(image)
            
            # Chi-square test for randomness
            chi_square_test = self._chi_square_test(image)
            
            # Frequency analysis
            frequency_analysis = self._analyze_frequency_patterns(image)
            
            # Check for unusual data patterns
            unusual_patterns = self._detect_unusual_data_patterns(image)
            
            # Overall suspicion based on multiple indicators
            suspicious_indicators = []
            
            if lsb_analysis['suspicious']:
                suspicious_indicators.append("Suspicious LSB patterns")
            
            if chi_square_test['p_value'] < 0.01:
                suspicious_indicators.append("Non-random data distribution")
            
            if frequency_analysis['anomalies'] > 3:
                suspicious_indicators.append("Frequency analysis anomalies")
            
            if unusual_patterns['score'] > 0.7:
                suspicious_indicators.append("Unusual data patterns detected")
            
            result = {
                'lsb_analysis': lsb_analysis,
                'chi_square_test': chi_square_test,
                'frequency_analysis': frequency_analysis,
                'unusual_patterns': unusual_patterns,
                'suspicious_indicators': suspicious_indicators,
                'suspicious': len(suspicious_indicators) > 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in steganography detection: {e}")
            return {
                'error': str(e),
                'suspicious': False
            }
    
    def _analyze_exif_data(self, exif_data: Dict) -> Dict:
        """Analyze EXIF data for authenticity indicators"""
        suspicious_indicators = []
        device_info = {}
        
        # Extract device information
        if 'Make' in exif_data:
            device_info['manufacturer'] = exif_data['Make'].lower()
        if 'Model' in exif_data:
            device_info['model'] = exif_data['Model'].lower()
        if 'Software' in exif_data:
            device_info['software'] = exif_data['Software'].lower()
        
        # Check device consistency
        if device_info:
            consistency_check = self._check_device_consistency(device_info)
            if not consistency_check['consistent']:
                suspicious_indicators.extend(consistency_check['issues'])
        
        # Check GPS data (screenshots shouldn't have GPS)
        if any(tag.startswith('GPS') for tag in exif_data.keys()):
            suspicious_indicators.append("GPS data present in screenshot")
        
        # Check for editing software signatures
        if 'Software' in exif_data:
            software = exif_data['Software'].lower()
            for app_type, apps in self.screenshot_app_signatures.items():
                if any(app in software for app in apps):
                    if app_type == 'editing_apps':
                        suspicious_indicators.append(f"Image edited with {software}")
                    break
        
        # Check timestamp consistency
        timestamp_analysis = self._analyze_timestamps(exif_data)
        if timestamp_analysis['suspicious']:
            suspicious_indicators.extend(timestamp_analysis['issues'])
        
        # Check for unusual EXIF structure
        if len(exif_data) < 3:
            suspicious_indicators.append("Minimal EXIF data - possibly stripped")
        elif len(exif_data) > 50:
            suspicious_indicators.append("Excessive EXIF data - possibly fabricated")
        
        return {
            'device_info': device_info,
            'timestamp_analysis': timestamp_analysis,
            'suspicious_indicators': suspicious_indicators,
            'suspicious': len(suspicious_indicators) > 0
        }
    
    def _check_device_consistency(self, device_info: Dict) -> Dict:
        """Check if device information is consistent"""
        issues = []
        
        manufacturer = device_info.get('manufacturer', '').lower()
        model = device_info.get('model', '').lower()
        software = device_info.get('software', '').lower()
        
        # Check Android device consistency
        if any(android_man in manufacturer for android_man in self.device_patterns['android']['manufacturers']):
            if 'ios' in software or 'iphone' in model:
                issues.append("Android manufacturer with iOS software")
        
        # Check iOS device consistency
        elif 'apple' in manufacturer:
            if any(android_soft in software for android_soft in self.device_patterns['android']['software_patterns']):
                issues.append("Apple device with Android software")
        
        # Check for impossible combinations
        if 'samsung' in manufacturer and 'iphone' in model:
            issues.append("Samsung manufacturer with iPhone model")
        
        if 'apple' in manufacturer and any(android_model in model for android_model in self.device_patterns['android']['models']):
            issues.append("Apple manufacturer with Android model")
        
        return {
            'consistent': len(issues) == 0,
            'issues': issues
        }
    
    def _analyze_timestamps(self, exif_data: Dict) -> Dict:
        """Analyze timestamp consistency in EXIF data"""
        issues = []
        timestamps = {}
        
        # Extract timestamps
        timestamp_fields = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
        for field in timestamp_fields:
            if field in exif_data:
                try:
                    timestamps[field] = datetime.strptime(exif_data[field], '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    issues.append(f"Invalid timestamp format in {field}")
        
        # Check timestamp consistency
        if len(timestamps) > 1:
            timestamp_values = list(timestamps.values())
            
            # Check if timestamps are too far apart
            for i in range(len(timestamp_values)):
                for j in range(i + 1, len(timestamp_values)):
                    time_diff = abs((timestamp_values[i] - timestamp_values[j]).total_seconds())
                    if time_diff > 3600:  # More than 1 hour difference
                        issues.append("Large time differences between EXIF timestamps")
        
        # Check if timestamps are in the future
        current_time = datetime.now()
        for field, timestamp in timestamps.items():
            if timestamp > current_time:
                issues.append(f"Future timestamp in {field}")
        
        return {
            'timestamps': {k: v.isoformat() for k, v in timestamps.items()},
            'issues': issues,
            'suspicious': len(issues) > 0
        }
    
    def _analyze_file_format(self, image_path: str) -> Dict:
        """Analyze file format consistency"""
        try:
            # Check file extension vs actual format
            file_extension = Path(image_path).suffix.lower()
            
            with Image.open(image_path) as img:
                actual_format = img.format.lower() if img.format else 'unknown'
            
            # Check consistency
            format_map = {
                '.jpg': 'jpeg',
                '.jpeg': 'jpeg',
                '.png': 'png',
                '.webp': 'webp',
                '.bmp': 'bmp'
            }
            
            expected_format = format_map.get(file_extension, 'unknown')
            consistent = (expected_format == actual_format or 
                         (expected_format in ['jpg', 'jpeg'] and actual_format in ['jpg', 'jpeg']))
            
            return {
                'file_extension': file_extension,
                'actual_format': actual_format,
                'consistent': consistent,
                'suspicious': not consistent
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'suspicious': False
            }
    
    def _calculate_average_hash(self, gray: np.ndarray, hash_size: int = 8) -> str:
        """Calculate average hash (aHash)"""
        resized = cv2.resize(gray, (hash_size, hash_size))
        avg = resized.mean()
        binary = resized > avg
        return ''.join(str(int(b)) for b in binary.flatten())
    
    def _calculate_perceptual_hash(self, gray: np.ndarray, hash_size: int = 8) -> str:
        """Calculate perceptual hash (pHash) using DCT"""
        # Resize to 32x32 for DCT
        resized = cv2.resize(gray, (32, 32))
        
        # Apply DCT
        dct = cv2.dct(resized.astype(np.float32))
        
        # Extract top-left 8x8 DCT coefficients
        dct_low = dct[:hash_size, :hash_size]
        
        # Calculate median
        median = np.median(dct_low)
        
        # Generate binary hash
        binary = dct_low > median
        return ''.join(str(int(b)) for b in binary.flatten())
    
    def _calculate_difference_hash(self, gray: np.ndarray, hash_size: int = 8) -> str:
        """Calculate difference hash (dHash)"""
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        
        # Calculate horizontal gradient
        diff = resized[:, 1:] > resized[:, :-1]
        
        return ''.join(str(int(b)) for b in diff.flatten())
    
    def _calculate_wavelet_hash(self, gray: np.ndarray, hash_size: int = 8) -> str:
        """Calculate wavelet hash using Haar wavelet"""
        try:
            import pywt
            
            # Resize image
            resized = cv2.resize(gray, (32, 32))
            
            # Apply Haar wavelet transform
            coeffs = pywt.dwt2(resized, 'haar')
            cA, (cH, cV, cD) = coeffs
            
            # Use approximation coefficients
            hash_coeffs = cv2.resize(cA, (hash_size, hash_size))
            
            # Generate binary hash
            median = np.median(hash_coeffs)
            binary = hash_coeffs > median
            
            return ''.join(str(int(b)) for b in binary.flatten())
            
        except ImportError:
            # Fallback to DCT-based method if PyWavelets not available
            return self._calculate_perceptual_hash(gray, hash_size)
    
    def _calculate_color_histogram(self, image: np.ndarray) -> str:
        """Calculate color histogram fingerprint"""
        # Calculate histogram for each channel
        hist_b = cv2.calcHist([image], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [16], [0, 256])
        
        # Combine histograms
        combined_hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        
        # Normalize and create binary representation
        normalized = combined_hist / np.sum(combined_hist)
        
        # Convert to hex string for compact representation
        return hashlib.md5(normalized.tobytes()).hexdigest()
    
    def _analyze_lsb_patterns(self, image: np.ndarray) -> Dict:
        """Analyze least significant bit patterns for steganography"""
        # Extract LSBs from each channel
        lsb_patterns = []
        
        for channel in range(3):
            channel_data = image[:, :, channel]
            lsb = channel_data & 1  # Extract LSB
            
            # Calculate randomness of LSB
            flat_lsb = lsb.flatten()
            
            # Chi-square test for randomness
            ones = np.sum(flat_lsb)
            zeros = len(flat_lsb) - ones
            expected = len(flat_lsb) / 2
            
            chi_square = ((ones - expected) ** 2 + (zeros - expected) ** 2) / expected
            
            lsb_patterns.append({
                'channel': channel,
                'ones_ratio': ones / len(flat_lsb),
                'chi_square': chi_square
            })
        
        # Overall suspicion
        avg_chi_square = np.mean([p['chi_square'] for p in lsb_patterns])
        suspicious = avg_chi_square > 3.84  # 95% confidence threshold
        
        return {
            'patterns': lsb_patterns,
            'average_chi_square': avg_chi_square,
            'suspicious': suspicious
        }
    
    def _chi_square_test(self, image: np.ndarray) -> Dict:
        """Perform chi-square test for randomness"""
        # Flatten image
        flat_image = image.flatten()
        
        # Calculate histogram
        hist, _ = np.histogram(flat_image, bins=256, range=(0, 256))
        
        # Expected frequency (uniform distribution)
        expected = len(flat_image) / 256
        
        # Chi-square statistic
        chi_square = np.sum((hist - expected) ** 2 / expected)
        
        # Degrees of freedom
        df = 255
        
        # Critical value for 99% confidence
        critical_value = 310.46  # Approximate for df=255
        
        # P-value approximation (simplified)
        p_value = 1 - (chi_square / (2 * df))  # Rough approximation
        p_value = max(0, min(1, p_value))
        
        return {
            'chi_square': chi_square,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'critical_value': critical_value,
            'significant': chi_square > critical_value
        }
    
    def _analyze_frequency_patterns(self, image: np.ndarray) -> Dict:
        """Analyze frequency domain patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Analyze frequency distribution
        center_y, center_x = np.array(magnitude_spectrum.shape) // 2
        
        # Calculate radial frequency distribution
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Analyze different frequency bands
        low_freq = np.mean(magnitude_spectrum[radius < 10])
        mid_freq = np.mean(magnitude_spectrum[(radius >= 10) & (radius < 50)])
        high_freq = np.mean(magnitude_spectrum[radius >= 50])
        
        # Look for anomalies
        anomalies = 0
        
        # Check for unusual frequency distribution
        total_energy = low_freq + mid_freq + high_freq
        if total_energy > 0:
            low_ratio = low_freq / total_energy
            high_ratio = high_freq / total_energy
            
            # Natural images typically have more low frequency content
            if low_ratio < 0.5:
                anomalies += 1
            if high_ratio > 0.3:
                anomalies += 1
        
        return {
            'low_frequency': low_freq,
            'mid_frequency': mid_freq,
            'high_frequency': high_freq,
            'anomalies': anomalies
        }
    
    def _detect_unusual_data_patterns(self, image: np.ndarray) -> Dict:
        """Detect unusual data patterns that might indicate tampering"""
        unusual_score = 0.0
        
        # Check for repeated patterns
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Template matching for repeated blocks
        template_size = 16
        h, w = gray.shape
        
        if h > template_size * 2 and w > template_size * 2:
            # Extract a template from the center
            center_y, center_x = h // 2, w // 2
            template = gray[center_y:center_y+template_size, center_x:center_x+template_size]
            
            # Match template across image
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            
            # Count high correlation matches
            threshold = 0.8
            matches = np.sum(result > threshold)
            
            if matches > 5:  # Too many exact matches
                unusual_score += 0.3
        
        # Check for artificial patterns in color distribution
        for channel in range(3):
            channel_data = image[:, :, channel].flatten()
            
            # Check for quantization artifacts
            hist, _ = np.histogram(channel_data, bins=256)
            
            # Look for spikes in histogram (quantization)
            spikes = np.sum(hist > np.mean(hist) + 3 * np.std(hist))
            
            if spikes > 10:
                unusual_score += 0.1
        
        return {
            'score': min(unusual_score, 1.0),
            'suspicious': unusual_score > 0.5
        }
    
    def analyze(self, image_path: str) -> Dict:
        """
        Comprehensive metadata analysis
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with all metadata analysis results
        """
        results = {
            'exif_analysis': self.extract_exif_data(image_path),
            'file_properties': self.analyze_file_properties(image_path),
            'image_fingerprint': self.calculate_image_fingerprint(image_path),
            'steganography_detection': self.detect_steganography(image_path)
        }
        
        # Calculate overall metadata suspicion score
        suspicion_indicators = []
        
        for analysis_name, analysis_result in results.items():
            if analysis_result.get('suspicious', False):
                suspicion_indicators.append(analysis_name)
        
        overall_suspicion = len(suspicion_indicators) / len(results)
        
        results['overall_metadata_score'] = overall_suspicion
        results['metadata_suspicious'] = overall_suspicion > 0.3  # Lower threshold for metadata
        results['suspicious_analyses'] = suspicion_indicators
        
        return results