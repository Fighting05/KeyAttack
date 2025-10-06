import os
import numpy as np
import pickle
import json
import time
import traceback
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

from config_manager import ConfigManager
from audio_processing import AudioProcessor
from feature_extraction import FeatureExtractor
from keystroke_model import KeystrokeModelTrainer, SequenceModeling
from data_manager import DataManager


class KeystrokeRecognitionSystem:
    """键盘声音识别系统的主接口类"""

    def __init__(self, config_path=None, config_manager=None):
        """初始化键盘声音识别系统

        Args:
            config_path: 配置文件路径，默认为None时使用默认配置
            config_manager: 配置管理器实例，如果提供则优先使用
        """
        if config_manager:
            self.config_manager = config_manager
        else:
            self.config_manager = ConfigManager(config_path)

        self.audio_processor = AudioProcessor(self.config_manager)
        self.feature_extractor = FeatureExtractor(self.config_manager)
        self.model_trainer = KeystrokeModelTrainer(self.config_manager)
        # 移除n-gram功能
        # self.sequence_model = SequenceModeling(self.config_manager)
        self.data_manager = DataManager(self.config_manager)
        self.models = {}
        self.class_indices = None
        self.scaler = None

        # 尝试加载现有模型
        try:
            self.load_models()
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("需要先训练模型")

    def load_models(self):
        """加载所有已训练的模型"""
        model_dir = self.config_manager.get_path("model_dir")
        models = {}

        print(f"尝试从 {model_dir} 加载模型...")

        # 检查模型目录
        if os.path.exists(model_dir):
            print(f"模型目录存在，包含文件: {os.listdir(model_dir)}")
        else:
            print(f"警告: 模型目录 {model_dir} 不存在")
            os.makedirs(model_dir, exist_ok=True)
            return {}

        # 加载传统机器学习模型
        for model_name in ['random_forest', 'gradient_boosting']:
            model_path = os.path.join(model_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        models[model_name] = pickle.load(f)
                    print(f"已加载 {model_name} 模型")
                except Exception as e:
                    print(f"加载 {model_name} 模型时出错: {e}")

        # 尝试多种方法加载深度学习模型
        try:
            # 方法1: 直接使用keras
            try:
                import keras
                print(f"已找到keras版本: {keras.__version__}")

                cnn_path = os.path.join(model_dir, 'cnn_model.h5')
                if os.path.exists(cnn_path):
                    models['cnn'] = keras.models.load_model(cnn_path)
                    print("使用keras成功加载CNN模型")

                lstm_path = os.path.join(model_dir, 'lstm_model.h5')
                if os.path.exists(lstm_path):
                    models['lstm'] = keras.models.load_model(lstm_path)
                    print("使用keras成功加载LSTM模型")
            except Exception as e1:
                print(f"使用keras加载模型失败: {e1}")

                # 方法2: 使用tensorflow.keras
                try:
                    import tensorflow as tf
                    print(f"已找到tensorflow版本: {tf.__version__}")

                    cnn_path = os.path.join(model_dir, 'cnn_model.h5')
                    if os.path.exists(cnn_path):
                        models['cnn'] = tf.keras.models.load_model(cnn_path)
                        print("使用tensorflow.keras成功加载CNN模型")

                    lstm_path = os.path.join(model_dir, 'lstm_model.h5')
                    if os.path.exists(lstm_path):
                        models['lstm'] = tf.keras.models.load_model(lstm_path)
                        print("使用tensorflow.keras成功加载LSTM模型")
                except Exception as e2:
                    print(f"使用tensorflow.keras加载模型失败: {e2}")
        except Exception as e:
            print(f"加载深度学习模型时出错: {e}")

        # 加载类别索引映射
        class_indices_path = os.path.join(model_dir, 'class_indices.json')
        if os.path.exists(class_indices_path):
            try:
                with open(class_indices_path, 'r') as f:
                    self.class_indices = json.load(f)
                    # 创建逆映射
                    self.idx_to_class = {v: k for k, v in self.class_indices.items()}
                print(f"已加载类别索引映射")
            except Exception as e:
                print(f"加载类别索引映射时出错: {e}")

        # 加载缩放器
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    self.model_trainer.scaler = self.scaler  # 确保model_trainer也有缩放器
                print(f"已加载特征缩放器")
            except Exception as e:
                print(f"加载特征缩放器时出错: {e}")

        # 不再加载n-gram模型
        # self.sequence_model.load_model()

        self.models = models
        print(f"总共加载了 {len(models)} 个模型")
        return models

    def train_from_samples(self, sample_dir=None):
        """从样本目录训练模型，支持编码的连续按键序列

        Args:
            sample_dir: 样本目录，默认从配置加载

        Returns:
            dict: 训练好的模型
        """
        sample_dir = sample_dir or self.config_manager.get_path("data_dir")

        X = []  # 特征列表
        y = []  # 标签列表
        sequences = []  # 序列列表，用于训练序列模型

        print("开始处理训练样本...")

        # 检查目录是否存在
        if not os.path.exists(sample_dir):
            print(f"错误: 训练目录 {sample_dir} 不存在")
            return None

        # 收集样本文件
        samples = []
        for filename in os.listdir(sample_dir):
            if filename.endswith('.wav'):
                # 直接从文件名中提取所有数字
                sequence = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
                if sequence:  # 只有当序列包含有效数字时才添加
                    file_path = os.path.join(sample_dir, filename)
                    samples.append((file_path, sequence))
                    print(f"添加样本: {filename} (序列: {sequence})")

        print(f"\n找到 {len(samples)} 个训练样本")

        # 处理每个样本
        successful_samples = 0
        failed_samples = 0

        for file_path, true_sequence in samples:
            try:
                # 加载音频
                y_audio, sr = self.audio_processor.load_audio(file_path)

                # 设置预期的按键数量
                expected_keys = len(true_sequence)

                # 检测按键
                segments, segment_times, energy = self.audio_processor.detect_keystrokes(
                    y_audio, sr,
                    expected_length=expected_keys,  # 传入预期长度
                    min_interval=0.1
                )

                # 只处理正确分割的样本
                if len(segments) == len(true_sequence):
                    # 处理每个分段
                    for segment, true_char in zip(segments, true_sequence):
                        features = self.feature_extractor.extract_features(segment, sr)
                        X.append(features)
                        y.append(int(true_char))  # 确保标签是整数

                    # 添加到序列列表
                    sequences.append(true_sequence)
                    successful_samples += 1
                    print(f"成功处理: {os.path.basename(file_path)}")
                else:
                    failed_samples += 1
                    print(f"警告: 跳过文件 {os.path.basename(file_path)} - 分段数量不匹配")
                    print(f"预期按键数: {len(true_sequence)}, 检测到的分段数: {len(segments)}")

            except Exception as e:
                failed_samples += 1
                print(f"处理 {os.path.basename(file_path)} 时出错: {str(e)}")

        # 打印处理统计信息
        print(f"\n处理完成:")
        print(f"成功处理的样本数: {successful_samples}")
        print(f"失败的样本数: {failed_samples}")

        if successful_samples + failed_samples > 0:
            print(f"成功率: {successful_samples / (successful_samples + failed_samples) * 100:.2f}%")
        else:
            print("成功率: 0.00% (未处理任何样本)")

        if not X:
            print("错误: 未能从样本中提取有效特征")
            return None

        X = np.array(X)
        y = np.array(y)

        print(f"\n特征提取结果:")
        print(f"特征形状: {X.shape}")
        print(f"标签数量: {len(y)}")
        print(f"唯一标签: {np.unique(y)}")

        # 确保每个类别至少有两个样本
        label_counts = np.bincount(y)
        valid_labels = np.where(label_counts >= 2)[0]

        if len(valid_labels) < len(np.unique(y)):
            print("\n警告: 某些类别的样本数量不足，将被过滤")
            mask = np.isin(y, valid_labels)
            X = X[mask]
            y = y[mask]
            print(f"过滤后的特征形状: {X.shape}")
            print(f"过滤后的标签数量: {len(y)}")
            print(f"过滤后的唯一标签: {np.unique(y)}")

        # 训练模型
        print("\n开始训练模型...")
        models = self.model_trainer.train(X, y)
        self.models = models

        # 训练序列模型
        if sequences and self.config_manager.get("sequence_model.use_ngram", True):
            print("\n训练N-gram序列模型...")
            # self.sequence_model.train_ngram_model(sequences)

        return models

    def predict_from_file(self, file_path, verbose=False):
        """从音频文件预测按键序列

        Args:
            file_path: 音频文件路径
            verbose: 是否打印详细信息

        Returns:
            str: 预测的按键序列
        """
        try:
            # 加载音频
            y, sr = self.audio_processor.load_audio(file_path, trim_silence=True)

            # 尝试从文件名猜测预期的按键数量
            expected_length = None
            filename = os.path.basename(file_path)
            digit_part = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
            if digit_part:
                expected_length = len(digit_part)
                if verbose:
                    print(f"从文件名猜测的预期按键数量: {expected_length}")

            # 使用集成方法检测按键
            segments, segment_times = self.audio_processor.isolate_keystrokes_ensemble(y, sr, expected_length)

            if not segments:
                print("未检测到按键")
                return ""

            # 使用检测到的片段进行预测
            predicted_sequence = ""
            for segment in segments:
                # 提取特征
                features = self.feature_extractor.extract_features(segment, sr)
                features = features.reshape(1, -1)

                # 标准化特征
                if not hasattr(self.model_trainer, 'scaler') or self.model_trainer.scaler is None:
                    print("错误: 未加载特征缩放器，无法进行预测")
                    return ""

                features_scaled = self.model_trainer.scaler.transform(features)

                # 使用所有模型进行预测
                predictions = {}
                for name, model in self.models.items():
                    if name in ['cnn', 'lstm']:
                        # 深度学习模型
                        features_reshaped = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1],
                                                                    1)
                        pred_probs = model.predict(features_reshaped, verbose=0)
                        pred_class = np.argmax(pred_probs[0])

                        # 转换回原始标签
                        if hasattr(self.model_trainer, 'idx_to_class') and self.model_trainer.idx_to_class:
                            if str(pred_class) in self.model_trainer.idx_to_class:
                                predictions[name] = self.model_trainer.idx_to_class[str(pred_class)]
                            else:
                                predictions[name] = str(pred_class)
                        else:
                            # 没有映射时直接使用预测的类别索引
                            predictions[name] = str(pred_class)
                    else:
                        # 传统机器学习模型
                        pred = model.predict(features_scaled)[0]
                        predictions[name] = str(pred)

                # 投票选择最终预测
                votes = Counter(predictions.values())
                if votes:
                    predicted_char = votes.most_common(1)[0][0]
                else:
                    print("警告: 没有有效的预测结果")
                    continue  # 跳过当前段落
                predicted_sequence += predicted_char

                if verbose:
                    print(f"最终预测: {predicted_char}")

            if verbose:
                print(f"\n检测到的总按键数: {len(segments)}")
                print(f"预测序列: {predicted_sequence}")

                # 可视化结果 (如果需要)
                try:
                    vis_path = self.visualize_predictions(file_path, predicted_sequence)
                    print(f"可视化已保存至: {vis_path}")
                except Exception as e:
                    print(f"可视化结果时出错: {e}")

            # 直接返回原始预测结果，不使用n-gram纠正
            return predicted_sequence

        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            traceback.print_exc()
            return ""

    def visualize_predictions(self, audio_file, predictions):
        """可视化音频文件和预测结果

        Args:
            audio_file: 音频文件路径
            predictions: 预测结果字符串

        Returns:
            str: 可视化图像保存路径
        """
        y, sr = self.audio_processor.load_audio(audio_file)
        segments, segment_times, energy = self.audio_processor.detect_keystrokes(y, sr)

        # 创建可视化
        plt.figure(figsize=(15, 6))

        # 显示波形
        plt.subplot(2, 1, 1)
        times = np.linspace(0, len(y) / sr, len(y))
        plt.plot(times, y)
        plt.title("音频波形")
        plt.xlabel("时间 (秒)")
        plt.ylabel("振幅")

        # 如果分段数量与预测数量不匹配，只标记分段
        if len(segment_times) != len(predictions):
            for i, (start, end) in enumerate(segment_times):
                plt.axvspan(start, end, alpha=0.2, color='red')
                plt.text((start + end) / 2, plt.ylim()[1] * 0.9, f"{i + 1}",
                         horizontalalignment='center', fontsize=10)
        else:
            # 标记检测到的按键并添加预测标签
            for i, ((start, end), pred) in enumerate(zip(segment_times, predictions)):
                plt.axvspan(start, end, alpha=0.2, color='red')
                plt.text((start + end) / 2, plt.ylim()[1] * 0.9, pred,
                         horizontalalignment='center', fontsize=12, color='red')

        # 显示能量曲线
        plt.subplot(2, 1, 2)
        frame_times = np.linspace(0, len(y) / sr, len(energy))
        plt.plot(frame_times, energy)
        plt.title("能量曲线")
        plt.xlabel("时间 (秒)")
        plt.ylabel("能量")

        # 标记能量峰值
        for start, end in segment_times:
            mid = (start + end) / 2
            nearest_idx = np.argmin(np.abs(frame_times - mid))
            if nearest_idx < len(energy):
                plt.axvline(mid, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 保存图像
        results_dir = self.config_manager.get_path("results_dir")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(results_dir, f'prediction_visualization_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()

        return save_path
    # 在 keystroke_recognition.py 的 KeystrokeRecognitionSystem 类中添加这个新方法
# 保持原有逻辑不变，在此基础上增加多候选功能

    def predict_from_file_with_candidates(self, file_path, num_candidates=10):
        """
        基于原有预测逻辑，返回多个候选结果
        确保第一个候选就是原来的单一预测结果
        
        Args:
            file_path: 音频文件路径
            num_candidates: 候选数量
            
        Returns:
            list: [(序列, 置信度), ...] 候选列表
        """
        try:
            # 首先获取原来的单一预测结果（保持原有逻辑）
            original_prediction = self.predict_from_file(file_path)
            
            if not original_prediction:
                return []
            
            # 重新进行预测，但这次收集每个位置的多个候选
            # 保持和原有方法相同的音频处理逻辑
            y, sr = self.audio_processor.load_audio(file_path, trim_silence=True)
            
            # 尝试从文件名猜测预期的按键数量
            expected_length = None
            filename = os.path.basename(file_path)
            digit_part = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
            if digit_part:
                expected_length = len(digit_part)
            
            # 使用与原方法相同的按键检测逻辑
            segments, segment_times = self.audio_processor.isolate_keystrokes_ensemble(y, sr, expected_length)
            
            if not segments:
                return []
            
            # 为每个按键段收集候选预测
            position_candidates = []
            
            for segment in segments:
                # 提取特征（与原方法相同）
                features = self.feature_extractor.extract_features(segment, sr)
                features = features.reshape(1, -1)
                
                # 标准化特征
                if not hasattr(self.model_trainer, 'scaler') or self.model_trainer.scaler is None:
                    return []
                
                features_scaled = self.model_trainer.scaler.transform(features)
                
                # 收集每个模型的多个候选预测
                model_candidates = self._collect_model_candidates(features_scaled)
                position_candidates.append(model_candidates)
            
            # 生成多个完整序列候选
            sequence_candidates = self._generate_sequence_candidates_from_positions(
                position_candidates, original_prediction, num_candidates)
            
            return sequence_candidates
            
        except Exception as e:
            print(f"多候选预测出错: {e}")
            # 如果出错，至少返回原始预测
            original_prediction = self.predict_from_file(file_path)
            if original_prediction:
                return [(original_prediction, 1.0)]
            return []

    def _collect_model_candidates(self, features_scaled, top_k=3):
        """
        收集每个模型对单个按键的候选预测
        保持与原有预测逻辑的一致性
        
        Args:
            features_scaled: 标准化后的特征
            top_k: 每个模型返回的候选数
            
        Returns:
            dict: {候选字符: [模型置信度列表]}
        """
        candidate_scores = {}
        
        for name, model in self.models.items():
            try:
                if name in ['cnn', 'lstm']:
                    # 深度学习模型 - 获取概率分布
                    features_reshaped = features_scaled.reshape(
                        features_scaled.shape[0], features_scaled.shape[1], 1)
                    pred_probs = model.predict(features_reshaped, verbose=0)[0]
                    
                    # 获取top-k预测
                    top_indices = np.argsort(pred_probs)[-top_k:][::-1]
                    
                    for idx in top_indices:
                        prob = pred_probs[idx]
                        
                        # 转换回字符（与原逻辑相同）
                        if (hasattr(self.model_trainer, 'idx_to_class') and 
                            self.model_trainer.idx_to_class and 
                            str(idx) in self.model_trainer.idx_to_class):
                            predicted_char = self.model_trainer.idx_to_class[str(idx)]
                        else:
                            predicted_char = str(idx)
                        
                        if predicted_char not in candidate_scores:
                            candidate_scores[predicted_char] = []
                        candidate_scores[predicted_char].append(prob)
                        
                else:
                    # 传统机器学习模型
                    # 先获取主要预测
                    pred = model.predict(features_scaled)[0]
                    predicted_char = str(pred)
                    
                    if predicted_char not in candidate_scores:
                        candidate_scores[predicted_char] = []
                    candidate_scores[predicted_char].append(0.8)  # 固定置信度
                    
                    # 如果模型支持predict_proba，获取其他候选
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(features_scaled)[0]
                            # 获取其他高概率的候选
                            top_indices = np.argsort(proba)[-top_k:][::-1]
                            
                            for idx in top_indices:
                                if idx != pred:  # 跳过已经添加的主预测
                                    prob = proba[idx]
                                    if prob > 0.1:  # 只考虑概率较高的候选
                                        candidate_char = str(idx)
                                        if candidate_char not in candidate_scores:
                                            candidate_scores[candidate_char] = []
                                        candidate_scores[candidate_char].append(prob)
                        except:
                            pass  # 如果获取概率失败，只保留主预测
            except Exception as e:
                print(f"模型 {name} 预测出错: {e}")
                continue
        
        # 计算每个候选的综合分数（与原有集成逻辑保持一致）
        final_candidates = {}
        for char, scores in candidate_scores.items():
            # 使用平均分数作为综合置信度
            final_candidates[char] = np.mean(scores)
        
        # 按分数排序
        sorted_candidates = sorted(final_candidates.items(), 
                                key=lambda x: x[1], reverse=True)
        
        return sorted_candidates

    def _generate_sequence_candidates_from_positions(self, position_candidates, original_prediction, num_candidates):
        """
        从位置候选生成完整序列候选
        确保第一个候选是原始预测结果
        
        Args:
            position_candidates: 每个位置的候选列表
            original_prediction: 原始单一预测结果
            num_candidates: 目标候选数量
            
        Returns:
            list: [(序列, 置信度)] 候选列表
        """
        import itertools
        
        candidates = []
        
        # 首先确保原始预测在候选列表中（置信度最高）
        original_score = 1.0
        candidates.append((original_prediction, original_score))
        
        # 生成其他候选组合
        # 限制每个位置的候选数量以避免组合爆炸
        max_per_position = min(3, max(1, (num_candidates * 2) // len(position_candidates)))
        
        limited_candidates = []
        for pos_cands in position_candidates:
            # 取前几个候选
            limited_candidates.append(pos_cands[:max_per_position])
        
        # 生成所有可能的组合
        combination_set = set()  # 用于去重
        
        for combination in itertools.product(*limited_candidates):
            sequence = ''.join([cand[0] for cand in combination])
            
            # 跳过与原始预测相同的序列（已经添加过了）
            if sequence == original_prediction:
                continue
                
            # 避免重复
            if sequence in combination_set:
                continue
            combination_set.add(sequence)
            
            # 计算序列总分数（几何平均）
            scores = [cand[1] for cand in combination]
            if scores:
                total_score = np.prod(scores) ** (1.0 / len(scores))
                # 确保其他候选的分数都低于原始预测
                total_score = min(total_score, original_score - 0.001)
                candidates.append((sequence, total_score))
        
        # 按分数排序（原始预测应该排在第一）
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 返回指定数量的候选
        return candidates[:num_candidates]


# def test_run():
#     """使用全部训练数据进行训练和测试"""
#     import shutil
#
#     # 初始化系统
#     system = KeystrokeRecognitionSystem()
#
#     try:
#         print("\n开始训练模型...")
#         print("使用全部训练数据...")
#
#         # 直接使用训练目录中的所有数据
#         train_dir = system.config_manager.get_path("train_dir")
#         test_dir = system.config_manager.get_path("test_dir")
#
#         # 训练模型
#         models = system.train_from_samples(train_dir)
#
#         if models:
#             print("\n开始评估...")
#             # 读取映射文件获取真实标签
#             mapping_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
#             if not mapping_files:
#                 print("错误：未找到测试集映射文件")
#                 return
#
#             mapping_file = os.path.join(test_dir, mapping_files[0])
#             true_labels = {}
#
#             with open(mapping_file, 'r', encoding='utf-8') as f:
#                 import csv
#                 reader = csv.DictReader(f)
#                 for row in reader:
#                     true_labels[row['new_name']] = row['true_sequence']
#
#             # 评估每个测试文件
#             total_correct = 0
#             total_chars = 0
#
#             for filename in os.listdir(test_dir):
#                 if filename.endswith('.wav'):
#                     file_path = os.path.join(test_dir, filename)
#
#                     print(f"\n测试文件: {filename}")
#                     predicted_sequence = system.predict_from_file(file_path, verbose=True)
#
#                     # 从映射文件获取真实序列
#                     if filename in true_labels:
#                         true_sequence = true_labels[filename]
#
#                         # 计算准确率
#                         min_len = min(len(true_sequence), len(predicted_sequence))
#                         correct_chars = sum(1 for i in range(min_len)
#                                             if true_sequence[i] == predicted_sequence[i])
#
#                         total_correct += correct_chars
#                         total_chars += len(true_sequence)
#
#                         accuracy = (correct_chars / len(true_sequence)) * 100
#
#                         print(f"真实序列: {true_sequence} 预测序列: {predicted_sequence}")
#                         print(f"序列长度: 预期={len(true_sequence)}, 实际={len(predicted_sequence)}")
#                         print(f"准确率: {accuracy:.2f}%")
#
#                         # 打印详细的匹配情况
#                         print("\n详细匹配分析:")
#                         for i, pred_char in enumerate(predicted_sequence):
#                             if i < len(true_sequence):
#                                 match = "✓" if pred_char == true_sequence[i] else "✗"
#                                 print(f"位置 {i + 1:2d}: 预测值={pred_char}, 真实值={true_sequence[i]} {match}")
#                     else:
#                         print(f"警告：在映射文件中未找到文件 {filename} 的真实序列")
#
#             # 打印总体评估结果
#             if total_chars > 0:
#                 overall_accuracy = (total_correct / total_chars) * 100
#                 print(f"\n总体评估结果:")
#                 print(f"总字符数: {total_chars}")
#                 print(f"正确预测: {total_correct}")
#                 print(f"总体准确率: {overall_accuracy:.2f}%")
#
#     except Exception as e:
#         print(f"\n错误：{str(e)}")
#         traceback.print_exc()
#         print("训练或评估过程失败。")


# 替换 keystroke_recognition.py 中的 test_run() 函数

# 将这个函数放在 keystroke_recognition.py 文件中，作为独立函数（不在类内部）

def test_run():
    """使用全部训练数据进行训练和测试 - 支持多候选预测（保持原逻辑）"""
    import shutil
    from collections import defaultdict

    # 初始化系统
    system = KeystrokeRecognitionSystem()

    try:
        print("\n开始训练模型...")
        print("使用全部训练数据...")

        # 直接使用训练目录中的所有数据
        train_dir = system.config_manager.get_path("train_dir")
        test_dir = system.config_manager.get_path("test_dir")

        # 训练模型
        models = system.train_from_samples(train_dir)

        if models:
            print("\n开始评估...")
            
            # 询问是否使用多候选预测
            use_multiple = input("是否使用多候选预测？(y/n) [y]: ").lower() != 'n'
            
            num_candidates = 10
            if use_multiple:
                try:
                    num_candidates = int(input("请输入候选数量 [默认10]: ") or "10")
                except ValueError:
                    num_candidates = 10
                    
                print(f"使用 {num_candidates} 个候选进行评估")
            else:
                print("使用传统单一预测进行评估")
            
            # 读取映射文件获取真实标签
            mapping_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
            if not mapping_files:
                print("错误：未找到测试集映射文件")
                return

            mapping_file = os.path.join(test_dir, mapping_files[0])
            true_labels = {}

            with open(mapping_file, 'r', encoding='utf-8') as f:
                import csv
                reader = csv.DictReader(f)
                for row in reader:
                    true_labels[row['new_name']] = row['true_sequence']

            # 评估统计
            total_correct = 0
            total_chars = 0
            total_sequences = 0
            correct_sequences = 0
            
            # 多候选统计
            hit_stats = defaultdict(int)
            no_hit_count = 0

            for filename in os.listdir(test_dir):
                if filename.endswith('.wav'):
                    file_path = os.path.join(test_dir, filename)

                    print(f"\n测试文件: {filename}")
                    
                    # 从映射文件获取真实序列
                    if filename not in true_labels:
                        print(f"警告：在映射文件中未找到文件 {filename} 的真实序列")
                        continue
                        
                    true_sequence = true_labels[filename]
                    total_sequences += 1
                    print(f"真实序列: {true_sequence}")

                    if use_multiple:
                        # 使用多候选预测
                        candidates = system.predict_from_file_with_candidates(file_path, num_candidates)
                        
                        if not candidates:
                            print(f"预测失败: {filename}")
                            continue

                        # 第一个候选是原始预测
                        predicted_sequence = candidates[0][0]
                        original_confidence = candidates[0][1]
                        
                        print(f"原始预测: {predicted_sequence} (置信度: {original_confidence:.4f})")
                        
                        # 多候选分析
                        hit_rank = None
                        for i, (seq, score) in enumerate(candidates):
                            if seq == true_sequence:
                                hit_rank = i + 1
                                break

                        if hit_rank:
                            hit_stats[hit_rank] += 1
                            print(f"�� 多候选结果: 在第 {hit_rank} 名命中!")
                        else:
                            no_hit_count += 1
                            print(f"�� 多候选结果: 未在前 {num_candidates} 名中命中")

                        # 显示前5个候选
                        print(f"前5个候选:")
                        for i, (seq, score) in enumerate(candidates[:5]):
                            marker = " ★" if seq == true_sequence else ""
                            print(f"  {i+1}. {seq} ({score:.4f}){marker}")
                            
                    else:
                        # 使用传统单一预测
                        predicted_sequence = system.predict_from_file(file_path)
                        
                        if not predicted_sequence:
                            print(f"预测失败: {filename}")
                            continue
                            
                        print(f"预测结果: {predicted_sequence}")

                    # 计算传统准确率指标
                    # 字符级准确率
                    min_len = min(len(true_sequence), len(predicted_sequence))
                    correct_chars = sum(1 for i in range(min_len)
                                        if true_sequence[i] == predicted_sequence[i])

                    total_correct += correct_chars
                    total_chars += len(true_sequence)

                    char_accuracy = (correct_chars / len(true_sequence)) * 100

                    # 检查序列是否完全匹配
                    sequence_match = (predicted_sequence == true_sequence)
                    if sequence_match:
                        correct_sequences += 1
                        match_status = "✓ 完全匹配"
                    else:
                        match_status = "✗ 不匹配"

                    print(f"序列长度: 预期={len(true_sequence)}, 实际={len(predicted_sequence)}")
                    print(f"字符准确率: {char_accuracy:.2f}%")
                    print(f"序列匹配: {match_status}")

                    # 打印详细的匹配情况
                    print("\n详细匹配分析:")
                    for i, pred_char in enumerate(predicted_sequence):
                        if i < len(true_sequence):
                            match = "✓" if pred_char == true_sequence[i] else "✗"
                            print(f"位置 {i + 1:2d}: 预测值={pred_char}, 真实值={true_sequence[i]} {match}")

            # 打印总体评估结果
            print("\n" + "="*80)
            print(" " * 25 + "总体评估结果")
            print("="*80)

            if total_chars > 0:
                overall_char_accuracy = (total_correct / total_chars) * 100
                print(f"\n�� 基础预测性能:")
                print(f"  总字符数: {total_chars}")
                print(f"  正确预测: {total_correct}")
                print(f"  字符准确率: {overall_char_accuracy:.2f}%")

            if total_sequences > 0:
                sequence_accuracy = (correct_sequences / total_sequences) * 100
                print(f"  总序列数: {total_sequences}")
                print(f"  完全正确: {correct_sequences}")
                print(f"  序列准确率: {sequence_accuracy:.2f}%")

                # 多候选统计（如果启用）
                if use_multiple:
                    total_hits = sum(hit_stats.values())
                    overall_hit_rate = (total_hits / total_sequences) * 100

                    print(f"\n�� 多候选预测性能:")
                    print(f"  总命中数: {total_hits}")
                    print(f"  总命中率: {overall_hit_rate:.2f}%")
                    print(f"  未命中数: {no_hit_count}")

                    # 验证一致性
                    top1_hits = hit_stats.get(1, 0)
                    top1_rate = (top1_hits / total_sequences) * 100
                    print(f"\n�� 一致性检查:")
                    print(f"  原始序列准确率: {sequence_accuracy:.2f}%")
                    print(f"  多候选Top-1准确率: {top1_rate:.2f}%")
                    if abs(sequence_accuracy - top1_rate) < 0.1:
                        print("  ✓ 一致性检查通过")
                    else:
                        print("  ⚠️ 一致性检查失败，可能存在逻辑问题")

                    # 分排名命中率
                    print(f"\n�� 分排名命中统计:")
                    cumulative_hits = 0
                    for rank in range(1, min(num_candidates + 1, 11)):
                        rank_hits = hit_stats.get(rank, 0)
                        cumulative_hits += rank_hits
                        cumulative_rate = (cumulative_hits / total_sequences) * 100

                        if rank <= 5 or rank_hits > 0:
                            print(f"  Top-{rank}: {rank_hits} 次 (累计: {cumulative_rate:.1f}%)")

                    # 性能提升分析
                    top1_rate = (hit_stats.get(1, 0) / total_sequences) * 100
                    top5_rate = (sum(hit_stats.get(i, 0) for i in range(1, 6)) / total_sequences) * 100
                    top10_rate = (sum(hit_stats.get(i, 0) for i in range(1, 11)) / total_sequences) * 100

                    print(f"\n�� 性能提升分析:")
                    print(f"  Top-1 准确率: {top1_rate:.2f}%")
                    print(f"  Top-5 准确率: {top5_rate:.2f}% (提升: {top5_rate-top1_rate:.2f}%)")
                    print(f"  Top-10 准确率: {top10_rate:.2f}% (提升: {top10_rate-top1_rate:.2f}%)")

                    print(f"\n�� 详细分析:")
                    print(f"  传统方法正确序列: {correct_sequences}")
                    print(f"  多候选额外挽回序列: {total_hits - correct_sequences}")
                    recovery_rate = ((total_hits - correct_sequences) / (total_sequences - correct_sequences)) * 100 if total_sequences > correct_sequences else 0
                    print(f"  错误序列挽回率: {recovery_rate:.2f}%")

    except Exception as e:
        print(f"\n错误：{str(e)}")
        import traceback
        traceback.print_exc()
        print("训练或评估过程失败。")





if __name__ == "__main__":
    test_run()
