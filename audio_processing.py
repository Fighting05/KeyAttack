# audio_processing.py - 优化版
import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks, medfilt
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
import time
from tqdm import tqdm
from collections import Counter
import warnings

# 忽略警告，减少不必要的输出
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class AudioProcessor:
    """音频处理类

    提供音频加载、预处理、按键检测等功能
    """

    def __init__(self, config_manager):
        """初始化音频处理器

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager
        self.sr = self.config.get("sample_rate")

        # 初始化可视化设置
        self.visualization_enabled = True
        self.visualization_dir = self.config.get_path("results_dir")

        # 检测方法统计
        self.method_stats = {
            "peak_detection": 0,
            "equal_segments": 0
        }

        # 防止递归过深
        self._recursion_depth = 0
        self._max_recursion_depth = 3

    def load_audio(self, file_path, normalize=True, trim_silence=True):
        """加载并预处理音频文件

        Args:
            file_path: 音频文件路径
            normalize: 是否规范化音频
            trim_silence: 是否裁剪首尾静音

        Returns:
            tuple: (音频数据, 采样率)
        """
        try:
            # 先尝试使用 soundfile
            audio_data, sr = sf.read(file_path)
            # 如果是立体声，转为单声道
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                y = np.mean(audio_data, axis=1)
            else:
                y = audio_data
            print(f"使用 soundfile 加载音频: {file_path}")
        except Exception as e:
            print(f"使用 soundfile 加载失败：{e}，尝试使用 librosa")
            try:
                y, sr = librosa.load(file_path, sr=self.sr)
            except Exception as e2:
                print(f"使用 librosa 加载也失败: {e2}")
                # 返回一个短的空音频，避免程序崩溃
                return np.zeros(1000), self.sr

        if normalize:
            y = librosa.util.normalize(y)

        # 裁剪首尾静音
        if trim_silence:
            try:
                y, _ = librosa.effects.trim(y, top_db=30)
            except Exception as e:
                print(f"裁剪静音失败: {e}")

        # 应用带通滤波器
        filter_config = self.config.get("bandpass_filter")
        if filter_config:
            try:
                y = self.bandpass_filter(
                    y,
                    filter_config["lowcut"],
                    filter_config["highcut"],
                    sr,
                    filter_config["order"]
                )
            except Exception as e:
                print(f"应用带通滤波器失败: {e}")

        return y, sr

    def bandpass_filter(self, data, lowcut, highcut, sr, order=5):
        """应用带通滤波器

        Args:
            data: 音频数据
            lowcut: 低频截止
            highcut: 高频截止
            sr: 采样率
            order: 滤波器阶数

        Returns:
            滤波后的音频数据
        """
        try:
            nyq = 0.5 * sr
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, data)
        except Exception as e:
            print(f"带通滤波器应用失败: {e}")
            return data  # 返回原始数据

    def estimate_noise_threshold(self, energy, percentile=20):
        """估计噪声阈值

        Args:
            energy: 音频能量曲线
            percentile: 用于噪声估计的百分位数

        Returns:
            tuple: (噪声水平, 动态阈值)
        """
        noise_level = np.percentile(energy, percentile)
        signal_level = np.percentile(energy, 95)
        return noise_level, noise_level + 0.7 * (signal_level - noise_level)

    def enhanced_preprocessing(self, y, sr):
        """增强的音频预处理 - 改进版
        结合多种噪声消除和信号增强技术
        """
        try:
            # 1. 应用带通滤波
            filter_config = self.config.get("bandpass_filter")
            if filter_config:
                y_filtered = self.bandpass_filter(
                    y,
                    filter_config["lowcut"],
                    filter_config["highcut"],
                    sr,
                    filter_config["order"]
                )
            else:
                y_filtered = y

            # 2. 尝试提高瞬态信号的强度（按键敲击声）
            try:
                # 使用HPSS分离谐波和打击乐成分
                y_harmonic, y_percussive = librosa.effects.hpss(y_filtered)
                # 加权组合，增强打击乐（敲击）部分
                y_enhanced = y_percussive * 0.8 + y_filtered * 0.2
            except:
                # 如果分离失败，使用原始滤波信号
                y_enhanced = y_filtered

            # 3. 应用简化的动态范围压缩
            # 计算信号的均方根能量
            rms = np.sqrt(np.mean(y_enhanced ** 2))
            # 设置阈值和压缩比
            threshold = rms * 0.7
            ratio = 0.6  # 压缩比 1.67:1

            # 应用压缩 - 使用向量化操作而不是逐元素处理
            mask = np.abs(y_enhanced) > threshold
            y_compressed = y_enhanced.copy()
            y_compressed[mask] = np.sign(y_enhanced[mask]) * (
                    threshold + (np.abs(y_enhanced[mask]) - threshold) * ratio
            )

            # 4. 最终规范化
            y_final = librosa.util.normalize(y_compressed)

            return y_final

        except Exception as e:
            print(f"增强预处理失败: {e}")
            return y  # 返回原始数据

    # 在audio_processing.py中添加simple_keystroke_detection方法
    def detect_keystrokes(self, y, sr, expected_length=None, min_interval=0.05):
        """增强的按键检测方法，考虑按键回弹特性，来自simplified_audio_processor.py

        Args:
            y: 音频数据
            sr: 采样率
            expected_length: 预期按键数量
            min_interval: 最小按键间隔(秒)

        Returns:
            tuple: (按键片段列表, 按键时间列表, 能量曲线)
        """
        # 计算短时能量
        frame_length = int(self.config.get("frame_length", 1024))
        hop_length = int(self.config.get("hop_length", 256))
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # 自适应阈值 - 基于能量统计
        noise_level = np.percentile(energy, 20)
        signal_level = np.percentile(energy, 95)
        adaptive_threshold = noise_level + 0.7 * (signal_level - noise_level)

        # 转换峰值帧到时间
        feature_times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)

        # 设置最小距离约为50毫秒，确保按键分离
        min_distance = int(min_interval * sr / hop_length)
        # 设置回弹峰值的间隔窗口 - 通常回弹峰值出现在主峰值后的0.05-0.15秒内
        rebound_window = int(0.12 * sr / hop_length)  # 约120ms的窗口

        try:
            # 找到所有候选峰值 - 使用较低阈值以捕获可能的回弹峰值
            all_peaks, _ = find_peaks(energy, height=adaptive_threshold * 0.7, distance=int(0.03 * sr / hop_length))

            # 过滤并合并回弹峰值
            true_peaks = []
            i = 0
            while i < len(all_peaks):
                current_peak = all_peaks[i]
                true_peaks.append(current_peak)

                # 检查后续的峰值是否是回弹峰值(在短时间窗口内)
                j = i + 1
                while j < len(all_peaks) and all_peaks[j] - current_peak <= rebound_window:
                    # 跳过这个峰值，它被认为是回弹峰值
                    j += 1

                # 跳到下一个非回弹峰值
                i = j

            # 现在true_peaks包含了过滤后的真实按键峰值
            peaks = true_peaks

            # 如果有预期长度但检测到的峰值数量不匹配，尝试调整阈值
            if expected_length and len(peaks) != expected_length:
                # 尝试不同的阈值乘数
                # 更广泛的阈值范围，包括更低的阈值以捕获弱按键
                threshold_multipliers = [2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
                best_diff = abs(len(peaks) - expected_length)
                best_peaks = peaks

                for mult in threshold_multipliers:
                    new_threshold = np.mean(energy) + mult * np.std(energy)
                    # 找到所有候选峰值
                    all_new_peaks, _ = find_peaks(energy, height=new_threshold * 0.7,
                                                  distance=int(0.03 * sr / hop_length))

                    # 过滤回弹峰值
                    new_true_peaks = []
                    i = 0
                    while i < len(all_new_peaks):
                        current_peak = all_new_peaks[i]
                        new_true_peaks.append(current_peak)

                        # 检查回弹峰值
                        j = i + 1
                        while j < len(all_new_peaks) and all_new_peaks[j] - current_peak <= rebound_window:
                            j += 1

                        # 跳到下一个非回弹峰值
                        i = j

                    new_diff = abs(len(new_true_peaks) - expected_length)

                    # 如果找到更好的结果，更新
                    if new_diff < best_diff:
                        best_diff = new_diff
                        best_peaks = new_true_peaks

                    # 如果完全匹配，立即使用
                    if new_diff == 0:
                        print(f"通过调整阈值找到精确匹配 (乘数: {mult})")
                        peaks = new_true_peaks
                        break

                # 如果尝试所有阈值后仍然没有精确匹配，但找到了更好的结果
                if len(peaks) != expected_length and best_diff < abs(len(peaks) - expected_length):
                    peaks = best_peaks

        except Exception as e:
            print(f"峰值检测失败: {e}，使用备用方法")
            # 备用方法：手动查找超过阈值的点
            peaks = []
            for i in range(1, len(energy) - 1):
                # 如果当前点高于阈值且是局部最大值
                if (energy[i] > adaptive_threshold and
                        energy[i] > energy[i - 1] and energy[i] > energy[i + 1]):
                    # 添加到峰值列表
                    if not peaks or (i - peaks[-1]) >= min_distance:
                        peaks.append(i)

        # 如果检测到的峰值仍然与预期不符，进行进一步调整
        if expected_length and len(peaks) != expected_length:
            if len(peaks) > expected_length:
                # 如果峰值太多，保留能量最高的expected_length个
                peak_energies = [(peak, energy[peak]) for peak in peaks]
                peak_energies.sort(key=lambda x: x[1], reverse=True)
                peaks = sorted([peak for peak, _ in peak_energies[:expected_length]])
                print(f"保留 {len(peaks)} 个能量最高的峰值")
            elif len(peaks) < expected_length and len(peaks) > 0:
                # 按照峰值间的平均距离插入额外峰值
                avg_distance = len(energy) / expected_length
                print(f"峰值数量不足，通过插值补充到预期的 {expected_length} 个")
                current_peaks = sorted(peaks)
                new_peaks = []

                # 如果峰值间距太大，或者峰值数量太少，尝试等间隔分布
                if len(peaks) <= expected_length / 2 or (len(peaks) > 1 and peaks[-1] - peaks[0] > 0.8 * len(energy)):
                    # 使用等间隔分布
                    frame_indices = np.linspace(0, len(energy) - 1, expected_length, dtype=int)
                    print(f"使用等间隔分布创建 {len(frame_indices)} 个峰值")
                    peaks = frame_indices
                else:
                    # 尝试在现有峰值之间插入新峰值
                    # 首先添加现有峰值
                    new_peaks = list(peaks)

                    # 计算需要添加的峰值数量
                    to_add = expected_length - len(peaks)

                    if to_add > 0 and len(peaks) > 1:
                        # 计算所有峰值间距
                        distances = []
                        for i in range(len(peaks) - 1):
                            distances.append((peaks[i + 1] - peaks[i], i))

                        # 按距离降序排序
                        distances.sort(reverse=True)

                        # 在最大间距处插入峰值
                        for _ in range(to_add):
                            if not distances:
                                break

                            dist, idx = distances.pop(0)
                            if dist < min_distance * 2:
                                continue  # 间距太小，不插入

                            # 计算插入位置
                            insert_pos = peaks[idx] + dist // 2
                            new_peaks.append(insert_pos)

                        # 排序
                        new_peaks.sort()

                    peaks = new_peaks if len(new_peaks) > len(peaks) else peaks
                    print(f"通过插值，峰值数量从 {len(current_peaks)} 增加到 {len(peaks)}")

                # 如果上述方法仍然不足，使用等间隔分布
                if len(peaks) < expected_length:
                    peaks = np.linspace(0, len(energy) - 1, expected_length, dtype=int)
                    print(f"最终使用等间隔分布")

        # 提取音频片段
        segments = []
        segment_times = []

        # 设置片段参数 - 考虑回弹峰值，增加后段长度
        segment_before = 0.05  # 50ms, 按键前
        segment_after = 0.20  # 200ms, 按键后(增加以包含回弹)

        for peak in peaks:
            # 计算时间边界
            peak_time = feature_times[peak]
            start_time = max(0, peak_time - segment_before)
            end_time = min(len(y) / sr, peak_time + segment_after)

            # 提取音频片段
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            if start_sample < end_sample and start_sample >= 0 and end_sample <= len(y):
                # 确保使用正步长的数组 - 解决PyTorch错误
                segment = y[start_sample:end_sample].copy()
                segments.append(segment)
                segment_times.append((start_time, end_time))

        # 如果没有找到任何峰值，使用等间隔分割
        if not segments and expected_length:
            print(f"未检测到任何峰值，使用等间隔分割")
            # 实现等间隔分割逻辑...
            margin = int(0.2 * sr)
            if len(y) > 2 * margin:
                y_effective = y[margin:-margin]
                offset = margin
            else:
                y_effective = y
                offset = 0

            # 计算每个片段的长度
            segment_length = len(y_effective) // expected_length

            for i in range(expected_length):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, len(y_effective))

                # 转换回原始音频索引
                start_sample = offset + start_idx
                end_sample = offset + end_idx

                # 确保使用正步长的数组 - 解决PyTorch错误
                segment = y[start_sample:end_sample].copy()
                segments.append(segment)
                segment_times.append((start_sample / sr, end_sample / sr))

        return segments, segment_times, energy

    def isolate_keystrokes_ensemble(self, y, sr, expected_length=None):
        """使用集成方法检测按键

        集成多种检测方法，采用投票机制确定最终结果

        Args:
            y: 音频数据
            sr: 采样率
            expected_length: 预期按键数量

        Returns:
            tuple: (按键片段列表, 按键时间列表)
        """
        # 使用改进的detect_keystrokes方法(包含回弹特性处理)代替复杂的集成方法
        segments, segment_times, _ = self.detect_keystrokes(y, sr, expected_length)

        # 直接使用detect_keystrokes的结果，不需要额外处理
        return segments, segment_times

    def process_audio_files(self, input_dir='./latest/', output_dir='./dataset/'):
        """处理输入目录中的音频文件，提取按键并保存

        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径

        Returns:
            tuple: (成功处理的数量, 失败的数量)
        """
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 按键映射
        keys_s = '1234567890qwertyuiopasdfghjklzxcvbnm'
        mapper = {char: idx for idx, char in enumerate(keys_s)}

        # 保存映射以供将来参考
        with open(output_dir / 'label_mapping.json', 'w') as f:
            json.dump(mapper, f)

        # 数据集元数据
        data_dict = {'Key': [], 'File': [], 'OriginalFile': [], 'TimeInFile': [], 'Method': []}

        # 获取输入目录中的WAV文件列表
        input_dir = Path(input_dir)
        wav_files = list(input_dir.glob('*.wav'))
        print(f"找到 {len(wav_files)} 个WAV文件")

        # 统计信息
        success_count = 0
        fail_count = 0

        # 处理每个WAV文件
        for wav_path in tqdm(wav_files, desc="处理音频文件"):
            filename = wav_path.name
            basename = wav_path.stem  # 不带扩展名的文件名

            print(f"\n正在处理 {filename}...")
            file_success = False  # 跟踪文件处理是否成功

            try:
                # 加载音频文件
                y, sr = self.load_audio(wav_path)

                # 设置预期的按键数量
                expected_length = len(basename)

                # 检测按键
                segments, segment_times, _ = self.detect_keystrokes(
                    y, sr,
                    expected_length=expected_length,
                    min_interval=0.1
                )

                # 将检测到的按键与文件名中的字符匹配
                matched_strokes = []

                if segments and len(segments) == expected_length:
                    # 按键数量与预期完全匹配
                    matched_strokes = [(segments[i], basename[i], segment_times[i]) for i in range(expected_length)]

                    # 检查匹配情况
                    if len(matched_strokes) == expected_length:
                        # 保存所有按键 - 在这里尝试保存
                        all_saved = True  # 跟踪是否所有片段都成功保存

                        for i, (stroke, key, (start_time, end_time)) in enumerate(matched_strokes):
                            if key not in mapper:
                                print(f"警告：映射中不存在键 '{key}'，跳过")
                                continue

                            try:
                                # 生成唯一ID
                                uid = len(data_dict['Key'])

                                # 保存张量文件 - 使用copy()避免负步长错误
                                save_path = output_dir / f"{mapper[key]}_{basename}_{i}.pt"
                                # 确保使用正步长的数组
                                stroke_tensor = torch.from_numpy(stroke.copy())
                                torch.save(stroke_tensor, save_path)

                                # 保存WAV文件
                                sf.write(
                                    output_dir / f"{mapper[key]}_{basename}_{i}.wav",
                                    stroke.astype(np.float32),
                                    sr
                                )

                                # 更新数据集信息
                                data_dict['Key'].append(key)
                                data_dict['File'].append(str(save_path))
                                data_dict['OriginalFile'].append(filename)
                                data_dict['TimeInFile'].append((start_time + end_time) / 2)
                                data_dict['Method'].append("peak_detection")

                            except Exception as e:
                                print(f"保存片段 {i} 失败: {e}")
                                all_saved = False

                        if all_saved:
                            file_success = True
                            print(f"成功处理: {filename}")
                        else:
                            print(f"部分片段保存失败: {filename}")
                    else:
                        print(f"匹配失败: {filename}")
                else:
                    print(f"警告: 跳过文件 {filename} - 分段数量不匹配")
                    print(f"预期按键数: {expected_length}, 检测到的分段数: {len(segments)}")

            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")
                import traceback
                traceback.print_exc()

            # 更新统计信息
            if file_success:
                success_count += 1
            else:
                fail_count += 1

        # 保存数据集元数据
        if data_dict['Key']:
            df = pd.DataFrame(data_dict)
            df.to_csv(output_dir / 'dataset_metadata.csv', index=False)

        # 打印处理统计信息
        print(f"\n处理完成:")
        print(f"成功处理: {success_count} 个文件")
        print(f"失败处理: {fail_count} 个文件")
        if success_count + fail_count > 0:
            print(f"成功率: {success_count / (success_count + fail_count) * 100:.2f}%")

        return success_count, fail_count

    def visualize_audio(self, y, sr, segments=None, segment_times=None, title="音频波形", save=True,
                        show_features=False):
        """可视化音频波形和检测到的按键

        Args:
            y: 音频数据
            sr: 采样率
            segments: 按键片段列表
            segment_times: 按键时间列表
            title: 图表标题
            save: 是否保存图像
            show_features: 是否显示特征曲线

        Returns:
            str or None: 如果save=True，返回保存路径，否则返回None
        """
        try:
            if show_features:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
            else:
                fig, ax1 = plt.subplots(figsize=(12, 4))

            # 绘制波形
            ax1.set_title(f"{title}")
            librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax1)

            # 标记检测到的按键
            if segment_times:
                for i, (start, end) in enumerate(segment_times):
                    ax1.axvspan(start, end, alpha=0.2, color='red')
                    ax1.axvline(start, color='green', linestyle='--', alpha=0.7)
                    ax1.axvline(end, color='blue', linestyle='--', alpha=0.7)
                    ax1.text((start + end) / 2, ax1.get_ylim()[1] * 0.9, f"{i + 1}",
                             horizontalalignment='center', fontsize=10)

            ax1.set_xlabel('时间 (秒)')
            ax1.set_ylabel('振幅')

            # 如果需要显示特征曲线
            if show_features:
                # 计算能量曲线
                frame_length = int(self.config.get("frame_length"))
                hop_length = int(self.config.get("hop_length"))
                energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                times = librosa.times_like(energy, sr=sr, hop_length=hop_length)

                # 计算频谱通量
                spec = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
                spectral_flux = np.sum(np.diff(spec, axis=1) ** 2, axis=0)
                spectral_flux = np.concatenate([[0], spectral_flux])
                spectral_flux = librosa.util.normalize(spectral_flux)

                # 计算过零率
                zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
                zcr = librosa.util.normalize(zcr)

                # 绘制特征曲线
                ax2.plot(times, energy, label='能量', color='blue', alpha=0.8)
                ax2.plot(times, spectral_flux, label='频谱通量', color='red', alpha=0.6)
                ax2.plot(times, zcr, label='过零率', color='green', alpha=0.6)

                # 计算平均特征
                combined = (energy * 0.6 + spectral_flux * 0.3 + zcr * 0.1)
                ax2.plot(times, combined, label='组合特征', color='purple', alpha=0.9, linewidth=2)

                # 标记检测到的按键
                if segment_times:
                    for start, end in segment_times:
                        ax2.axvspan(start, end, alpha=0.1, color='red')
                        ax2.axvline(start, color='green', linestyle='--', alpha=0.5)

                ax2.set_xlabel('时间 (秒)')
                ax2.set_ylabel('特征强度')
                ax2.legend(loc='upper right')

            plt.tight_layout()

            if save:
                # 保存图像到结果目录
                results_dir = self.visualization_dir
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_path = os.path.join(results_dir, f"audio_visualization_{timestamp}.png")
                plt.savefig(file_path)
                plt.close()
                return file_path
            else:
                plt.show()
                plt.close()
                return None

        except Exception as e:
            print(f"可视化音频时出错: {e}")
            plt.close()
            return None

    def visualize_segmentation_comparison(self, y, sr, segments_list, segment_times_list, method_names,
                                          title="分割方法对比"):
        """可视化多种分割方法的对比 - 优化版

        Args:
            y: 音频数据
            sr: 采样率
            segments_list: 多个方法的片段列表的列表
            segment_times_list: 多个方法的时间列表的列表
            method_names: 方法名称列表
            title: 图表标题

        Returns:
            str: 保存文件路径
        """
        try:
            num_methods = len(segments_list)
            if num_methods == 0:
                return None

            # 创建带有共享x轴的子图
            fig, axes = plt.subplots(num_methods + 1, 1, figsize=(12, 3 * (num_methods + 1)), sharex=True)
            fig.suptitle(title, fontsize=16)

            # 绘制原始波形
            librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=axes[0])
            axes[0].set_title("原始音频波形")
            axes[0].set_ylabel("振幅")

            # 计算能量曲线
            frame_length = int(self.config.get("frame_length"))
            hop_length = int(self.config.get("hop_length"))
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            times = librosa.times_like(energy, sr=sr, hop_length=hop_length)

            # 添加能量曲线
            axes_energy = axes[0].twinx()
            axes_energy.plot(times, energy, color='red', alpha=0.5, linestyle='-', linewidth=1)
            axes_energy.set_ylabel("能量", color='red')
            axes_energy.tick_params(axis='y', labelcolor='red')

            # 为每种方法绘制分割结果
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
            for i, (segments, segment_times, method_name) in enumerate(
                    zip(segments_list, segment_times_list, method_names)):
                # 检查分割结果是否有效
                if not segments or not segment_times:
                    axes[i + 1].set_title(f"{method_name} (未找到按键)")
                    continue

                # 现在在原始波形上显示分割
                color = colors[i % len(colors)]

                # 在原始波形上标记当前方法的分割
                for j, (start, end) in enumerate(segment_times):
                    axes[0].axvline(start, color=color, linestyle='--', alpha=0.5, linewidth=1)
                    axes[0].axvline(end, color=color, linestyle=':', alpha=0.5, linewidth=1)

                # 绘制该方法的分割结果
                for j, (segment, (start, end)) in enumerate(zip(segments, segment_times)):
                    # 显示该方法名称
                    axes[i + 1].set_title(f"{method_name} (找到 {len(segments)} 个按键)")

                    # 在每个方法自己的轴上显示原始波形
                    librosa.display.waveshow(y, sr=sr, alpha=0.2, ax=axes[i + 1], color='gray')

                    # 标记该方法检测到的分割
                    axes[i + 1].axvspan(start, end, alpha=0.3, color=color)
                    axes[i + 1].text((start + end) / 2, axes[i + 1].get_ylim()[1] * 0.8,
                                     f"{j + 1}", horizontalalignment='center', fontsize=10, color=color)

                axes[i + 1].set_ylabel("振幅")

            # 设置最后一个轴的x轴标签
            axes[-1].set_xlabel("时间 (秒)")

            plt.tight_layout()

            # 保存图像
            results_dir = self.visualization_dir
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(results_dir, f"segmentation_comparison_{timestamp}.png")
            plt.savefig(file_path)
            plt.close()

            return file_path

        except Exception as e:
            print(f"可视化分割对比时出错: {e}")
            plt.close()
            return None

    def visualize_method_stats(self, title="检测方法使用统计"):
        """可视化不同检测方法使用统计

        Args:
            title: 图表标题

        Returns:
            str: 保存的文件路径
        """
        try:
            # 过滤出使用次数非零的方法
            methods = []
            counts = []
            for method, count in self.method_stats.items():
                if count > 0:
                    methods.append(method)
                    counts.append(count)

            if not methods:
                return None

            # 创建饼图
            plt.figure(figsize=(10, 7))

            # 计算百分比
            total = sum(counts)
            percentages = [count / total * 100 for count in counts]

            # 绘制饼图
            wedges, texts, autotexts = plt.pie(
                counts,
                labels=methods,
                autopct='%1.1f%%',
                explode=[0.05] * len(methods),
                shadow=True,
                startangle=90,
                textprops={'fontsize': 12}
            )

            # 设置自动文本颜色
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)

            plt.axis('equal')  # 确保饼图是圆的
            plt.title(title, fontsize=16)

            # 添加统计信息
            info_text = f"总处理文件数: {total}\n"
            for method, count, percentage in zip(methods, counts, percentages):
                info_text += f"{method}: {count} ({percentage:.1f}%)\n"

            plt.figtext(0.95, 0.5, info_text, fontsize=12,
                        horizontalalignment='right', verticalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

            # 保存图像
            results_dir = self.visualization_dir
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(results_dir, f"method_stats_{timestamp}.png")
            plt.savefig(file_path)
            plt.close()

            return file_path

        except Exception as e:
            print(f"可视化方法统计时出错: {e}")
            plt.close()
            return None