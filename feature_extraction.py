# feature_extraction.py
import numpy as np
import librosa
from scipy import stats


class FeatureExtractor:
    """特征提取类

    从音频片段中提取各种声学特征
    """

    def __init__(self, config_manager):
        """初始化特征提取器

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager

    def extract_features(self, y, sr):
        """从音频片段中提取丰富的特征集

        Args:
            y: 音频数据
            sr: 采样率

        Returns:
            np.ndarray: 特征向量
        """
        features = []
        feature_config = self.config.get("feature_extraction")

        # 1. MFCC特征
        if feature_config.get("use_mfcc", True):
            mfcc_features = self.extract_mfcc_features(y, sr)
            features.extend(mfcc_features)

        # 2. 频谱特征
        if feature_config.get("use_spectral", True):
            spectral_features = self.extract_spectral_features(y, sr)
            features.extend(spectral_features)

        # 3. 时域特征
        if feature_config.get("use_temporal", True):
            temporal_features = self.extract_temporal_features(y, sr)
            features.extend(temporal_features)

        # 4. 小波变换特征
        if feature_config.get("use_wavelet", True):
            wavelet_features = self.extract_wavelet_features(y)
            features.extend(wavelet_features)

        # 5. 和声特征
        if feature_config.get("use_chroma", True):
            chroma_features = self.extract_chroma_features(y, sr)
            features.extend(chroma_features)

        return np.array(features)

    def extract_mfcc_features(self, y, sr):
        """提取增强的MFCC特征"""
        # 提取MFCCs
        n_mfcc = self.config.get("mfcc_coef", 40)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # 计算统计量
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # 添加MFCC的delta和delta-delta特征
        # 检查是否有足够的帧来计算delta
        if mfccs.shape[1] >= 9:  # 默认width=9
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

            delta_mean = np.mean(mfcc_delta, axis=1)
            delta_std = np.std(mfcc_delta, axis=1)
            delta2_mean = np.mean(mfcc_delta2, axis=1)
            delta2_std = np.std(mfcc_delta2, axis=1)
        else:
            # 如果帧数不足，则使用较小的width参数或填充零
            width = min(3, mfccs.shape[1] - 1)
            if width > 1:
                mfcc_delta = librosa.feature.delta(mfccs, width=width)
                mfcc_delta2 = librosa.feature.delta(mfccs, order=2, width=width)

                delta_mean = np.mean(mfcc_delta, axis=1)
                delta_std = np.std(mfcc_delta, axis=1)
                delta2_mean = np.mean(mfcc_delta2, axis=1)
                delta2_std = np.std(mfcc_delta2, axis=1)
            else:
                # 如果帧数太少，无法计算差分，则用零填充
                delta_mean = np.zeros_like(mfcc_mean)
                delta_std = np.zeros_like(mfcc_std)
                delta2_mean = np.zeros_like(mfcc_mean)
                delta2_std = np.zeros_like(mfcc_std)

        # 计算偏度和峰度（如果可能）
        try:
            mfcc_skew = stats.skew(mfccs, axis=1)
            mfcc_kurtosis = stats.kurtosis(mfccs, axis=1)
        except:
            # 如果计算失败，用零填充
            mfcc_skew = np.zeros_like(mfcc_mean)
            mfcc_kurtosis = np.zeros_like(mfcc_mean)

        # 合并所有特征
        features = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_skew, mfcc_kurtosis,
            delta_mean, delta_std, delta2_mean, delta2_std
        ])

        return features

    def extract_spectral_features(self, y, sr):
        """提取频谱特征"""
        # 基本频谱特征
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)

        # 计算统计量
        features = [
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(np.mean(spectral_contrast, axis=0)),
            np.std(np.std(spectral_contrast, axis=0)),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(spectral_flatness),
            np.std(spectral_flatness)
        ]

        # 添加频谱峰值特征
        S = np.abs(librosa.stft(y))
        peak_freq_idx = np.argmax(S, axis=0)
        peak_freqs = librosa.fft_frequencies(sr=sr)[peak_freq_idx]

        features.extend([
            np.mean(peak_freqs),
            np.std(peak_freqs),
            np.median(peak_freqs)
        ])

        return features

    def extract_temporal_features(self, y, sr):
        """提取时域特征"""
        # 基本统计特征
        features = [
            np.mean(y),
            np.std(y),
            np.max(y),
            np.min(y),
            np.median(y),
            stats.skew(y),
            stats.kurtosis(y)
        ]

        # 过零率
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([
            np.mean(zcr),
            np.std(zcr)
        ])

        # RMS能量
        rms = librosa.feature.rms(y=y)[0]
        features.extend([
            np.mean(rms),
            np.std(rms),
            np.max(rms)
        ])

        # 包络特征
        envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sr))
        features.extend([
            np.mean(envelope),
            np.std(envelope),
            np.max(envelope)
        ])

        # 攻击时间估计
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        try:
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            if len(onset_frames) > 0:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                # 使用第一个onset作为攻击时间
                attack_time = onset_times[0]
            else:
                attack_time = 0
        except:
            attack_time = 0

        features.append(attack_time)

        return features

    def extract_wavelet_features(self, y):
        """提取小波变换特征"""
        try:
            import pywt

            # 应用小波变换
            wavelet = 'db4'
            coeffs = pywt.wavedec(y, wavelet, level=5)

            # 提取每个系数集的统计特征
            features = []
            for i, coef in enumerate(coeffs):
                features.extend([
                    np.mean(np.abs(coef)),
                    np.std(coef),
                    np.max(np.abs(coef)),
                    np.sum(coef ** 2) / len(coef)  # 能量
                ])

            return features
        except ImportError:
            # 如果pywt不可用，返回空列表
            print("未安装PyWavelets库，无法提取小波特征")
            return []

    def extract_chroma_features(self, y, sr):
        """提取和声特征"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        chroma_max = np.max(chroma, axis=1)

        return np.concatenate([chroma_mean, chroma_std, chroma_max])

    def extract_all_features(self, audio_segments, sr):
        """从所有音频段中提取特征

        Args:
            audio_segments: 音频片段列表
            sr: 采样率

        Returns:
            list: 特征列表
        """
        features_list = []

        for segment in audio_segments:
            # 确保段落有足够的样本
            frame_length = self.config.get("frame_length", 1024)
            if len(segment) >= frame_length:
                features = self.extract_features(segment, sr)
                features_list.append(features)
            else:
                print(f"警告: 片段长度 {len(segment)} 小于帧长 {frame_length}，跳过")

        return features_list