# data_manager.py
import os
import shutil
import random
import string
import csv
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from collections import Counter


class DataManager:
    """数据管理类

    负责数据处理、分割和操作
    """

    def __init__(self, config_manager):
        """初始化数据管理器

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager

    def convert_audio_files(self, source_dir=None, train_dir=None, test_dir=None, format_from='m4a', format_to='wav',
                            train_ratio=0.8, ffmpeg_path=None, ffprobe_path=None):
        """转换音频文件格式并分割训练/测试集

        Args:
            source_dir: 源文件目录，默认从配置加载
            train_dir: 训练集目录，默认从配置加载
            test_dir: 测试集目录，默认从配置加载
            format_from: 源文件格式
            format_to: 目标文件格式
            train_ratio: 训练集比例
            ffmpeg_path: ffmpeg可执行文件路径
            ffprobe_path: ffprobe可执行文件路径

        Returns:
            tuple: (成功转换的训练样本数, 成功转换的测试样本数)
        """
        try:
            from pydub import AudioSegment

            # 设置ffmpeg路径
            if ffmpeg_path:
                AudioSegment.converter = ffmpeg_path
                AudioSegment.ffmpeg = ffmpeg_path
            if ffprobe_path:
                AudioSegment.ffprobe = ffprobe_path

            # 测试ffmpeg配置
            try:
                test_audio = AudioSegment.silent(duration=1000)
                print("FFmpeg配置成功!")
            except Exception as e:
                print(f"FFmpeg配置错误: {str(e)}")
                return 0, 0

            # 设置目录
            source_dir = source_dir or os.path.join(os.getcwd(), 'newdata')
            train_dir = train_dir or self.config.get_path("train_dir")
            test_dir = test_dir or self.config.get_path("test_dir")

            # 确保目标目录存在
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # 获取所有指定格式的文件
            audio_files = [f for f in os.listdir(source_dir) if f.endswith(f'.{format_from}')]
            print(f"找到 {len(audio_files)} 个 {format_from.upper()} 文件")

            # 随机打乱文件顺序
            random.shuffle(audio_files)

            # 计算训练集大小（80%）
            train_size = int(len(audio_files) * train_ratio)

            train_success = 0
            test_success = 0

            # 处理每个文件
            for i, filename in enumerate(audio_files):
                try:
                    input_path = os.path.join(source_dir, filename)
                    print(f"\n处理文件 {i + 1}/{len(audio_files)}: {filename}")

                    # 生成输出文件名
                    base_name = os.path.splitext(filename)[0]
                    output_filename = f"key_{base_name}_{i + 1:03d}.{format_to}"

                    # 确定目标目录
                    target_dir = train_dir if i < train_size else test_dir
                    output_path = os.path.join(target_dir, output_filename)

                    print(f"输入路径: {input_path}")
                    print(f"输出路径: {output_path}")

                    # 检查输入文件是否存在
                    if not os.path.exists(input_path):
                        print(f"输入文件不存在: {input_path}")
                        continue

                    # 加载并转换音频
                    audio = AudioSegment.from_file(input_path, format=format_from)
                    audio.export(output_path, format=format_to)

                    # 更新成功计数
                    if i < train_size:
                        train_success += 1
                    else:
                        test_success += 1

                    print(f"成功转换: {filename}")

                except Exception as e:
                    print(f"处理 {filename} 时出错: {str(e)}")
                    continue

            print("\n数据集分割完成:")
            print(f"成功处理训练样本: {train_success}/{train_size}")
            print(f"成功处理测试样本: {test_success}/{len(audio_files) - train_size}")

            return train_success, test_success

        except ImportError:
            print("错误: 未安装pydub库。请使用pip install pydub安装。")
            return 0, 0

    def create_anonymous_test_set(self, original_test_dir=None, new_test_dir=None, mapping_file=None):
        """创建匿名测试集，用于盲测

        Args:
            original_test_dir: 原始测试目录，默认从配置加载
            new_test_dir: 新测试目录，默认从配置加载
            mapping_file: 映射文件路径，默认自动生成

        Returns:
            tuple: (新测试目录路径, 映射文件路径)
        """
        # 设置目录
        original_test_dir = original_test_dir or self.config.get_path("test_dir")
        new_test_dir = new_test_dir or self.config.get_path("original_test_dir")

        # 创建或清理目标目录
        if os.path.exists(new_test_dir):
            for file in os.listdir(new_test_dir):
                file_path = os.path.join(new_test_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(new_test_dir)

        # 创建映射列表
        mapping = []
        used_ids = set()

        # 重命名并复制文件
        for filename in os.listdir(original_test_dir):
            if filename.endswith('.wav'):
                # 提取真实序列
                parts = filename.split('_')
                if len(parts) >= 2:
                    true_sequence = parts[1]
                else:
                    true_sequence = filename.split('.')[0]  # 使用文件名作为序列

                # 生成唯一随机ID
                while True:
                    random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                    if random_id not in used_ids:
                        used_ids.add(random_id)
                        break

                new_filename = f'test_{random_id}.wav'

                # 保存映射信息
                mapping.append({
                    'original_name': filename,
                    'new_name': new_filename,
                    'true_sequence': true_sequence
                })

                # 复制文件到新目录
                src_path = os.path.join(original_test_dir, filename)
                dst_path = os.path.join(new_test_dir, new_filename)
                shutil.copy2(src_path, dst_path)

        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存映射到CSV
        if not mapping_file:
            mapping_file = os.path.join(new_test_dir, f'test_mapping_{timestamp}.csv')

        with open(mapping_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['new_name', 'true_sequence', 'original_name'])
            writer.writeheader()
            writer.writerows(mapping)

        print(f"\n处理完成!")
        print(f"处理了 {len(mapping)} 个文件")
        print(f"新测试文件保存在: {new_test_dir}")
        print(f"映射文件保存在: {mapping_file}")

        return new_test_dir, mapping_file

    def analyze_dataset(self, data_dir=None, show_distribution=True):
        """分析数据集，获取统计信息

        Args:
            data_dir: 数据目录，默认从配置加载
            show_distribution: 是否显示分布信息

        Returns:
            dict: 统计信息
        """
        data_dir = data_dir or self.config.get_path("data_dir")

        if not os.path.exists(data_dir):
            print(f"错误: 数据目录 {data_dir} 不存在")
            return {}

        # 文件格式统计
        formats = Counter()
        # 标签分布
        labels = Counter()
        # 文件大小
        file_sizes = []
        # 音频长度 (只统计WAV文件)
        audio_lengths = []

        # 统计文件
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                # 文件格式
                ext = os.path.splitext(filename)[1].lower()[1:]
                formats[ext] += 1

                # 文件大小
                file_size = os.path.getsize(file_path) / 1024  # KB
                file_sizes.append(file_size)

                # 尝试提取标签
                parts = filename.split('_')
                if len(parts) > 1:
                    try:
                        label = int(parts[0])
                        labels[label] += 1
                    except ValueError:
                        pass

                # 对于WAV文件，获取音频长度
                if ext == 'wav':
                    try:
                        audio_data, sr = sf.read(file_path)
                        duration = len(audio_data) / sr
                        audio_lengths.append(duration)
                    except Exception:
                        pass

        # 计算统计量
        stats = {
            'total_files': sum(formats.values()),
            'formats': dict(formats),
            'labels': dict(labels),
            'file_sizes': {
                'min': min(file_sizes) if file_sizes else 0,
                'max': max(file_sizes) if file_sizes else 0,
                'mean': sum(file_sizes) / len(file_sizes) if file_sizes else 0,
                'total': sum(file_sizes) if file_sizes else 0
            },
            'audio_lengths': {
                'min': min(audio_lengths) if audio_lengths else 0,
                'max': max(audio_lengths) if audio_lengths else 0,
                'mean': sum(audio_lengths) / len(audio_lengths) if audio_lengths else 0,
                'total': sum(audio_lengths) if audio_lengths else 0
            }
        }

        # 打印统计信息
        if show_distribution:
            print("\n数据集统计:")
            print(f"总文件数: {stats['total_files']}")
            print("\n文件格式分布:")
            for fmt, count in formats.most_common():
                print(f"  {fmt}: {count} 文件 ({count / stats['total_files'] * 100:.1f}%)")

            if labels:
                print("\n标签分布:")
                for label, count in sorted(labels.items()):
                    print(f"  {label}: {count} 文件 ({count / stats['total_files'] * 100:.1f}%)")

            print("\n文件大小:")
            print(f"  最小: {stats['file_sizes']['min']:.2f} KB")
            print(f"  最大: {stats['file_sizes']['max']:.2f} KB")
            print(f"  平均: {stats['file_sizes']['mean']:.2f} KB")
            print(f"  总计: {stats['file_sizes']['total']:.2f} KB ({stats['file_sizes']['total'] / 1024:.2f} MB)")

            if audio_lengths:
                print("\n音频长度:")
                print(f"  最短: {stats['audio_lengths']['min']:.2f} 秒")
                print(f"  最长: {stats['audio_lengths']['max']:.2f} 秒")
                print(f"  平均: {stats['audio_lengths']['mean']:.2f} 秒")
                print(
                    f"  总计: {stats['audio_lengths']['total']:.2f} 秒 ({stats['audio_lengths']['total'] / 60:.2f} 分钟)")

        return stats

    def save_keystroke_data(self, segments, labels, output_dir=None, base_filename="keystroke"):
        """保存按键数据和标签

        Args:
            segments: 按键音频数据列表
            labels: 标签列表
            output_dir: 输出目录，默认从配置加载
            base_filename: 基础文件名

        Returns:
            list: 保存的文件路径列表
        """
        output_dir = output_dir or self.config.get_path("data_dir")
        os.makedirs(output_dir, exist_ok=True)

        saved_files = []

        for i, (segment, label) in enumerate(zip(segments, labels)):
            # 生成文件名
            filename = f"{label}_{base_filename}_{i + 1}.pt"
            filepath = os.path.join(output_dir, filename)

            # 转换为张量并保存
            tensor_data = torch.tensor(segment)
            torch.save(tensor_data, filepath)

            saved_files.append(filepath)

            # 同时保存为WAV文件以便检查
            wav_filename = f"{label}_{base_filename}_{i + 1}.wav"
            wav_filepath = os.path.join(output_dir, wav_filename)

            # 确保采样率正确（默认44100）
            sr = self.config.get("sample_rate", 44100)
            sf.write(wav_filepath, segment, sr)

        return saved_files

    def split_dataset(self, data_dir=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        """按比例分割数据集为训练集、验证集和测试集

        Args:
            data_dir: 数据目录，默认从配置加载
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子

        Returns:
            tuple: (训练文件列表, 验证文件列表, 测试文件列表)
        """
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            print(f"警告: 分割比例之和不为1: {train_ratio + val_ratio + test_ratio}")
            # 正规化比例
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
            print(f"已调整为: 训练={train_ratio:.2f}, 验证={val_ratio:.2f}, 测试={test_ratio:.2f}")

        data_dir = data_dir or self.config.get_path("data_dir")

        # 获取所有PT文件
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]

        if not all_files:
            print(f"错误: 目录 {data_dir} 中没有.pt文件")
            return [], [], []

        # 设置随机种子
        random.seed(random_seed)
        # 随机打乱文件
        random.shuffle(all_files)

        # 计算分割点
        train_end = int(len(all_files) * train_ratio)
        val_end = train_end + int(len(all_files) * val_ratio)

        # 分割数据
        train_files = all_files[:train_end]
        val_files = all_files[train_end:val_end]
        test_files = all_files[val_end:]

        print(f"\n数据集分割完成: ")
        print(f"总文件数: {len(all_files)}")
        print(f"训练集: {len(train_files)} 文件 ({len(train_files) / len(all_files) * 100:.1f}%)")
        print(f"验证集: {len(val_files)} 文件 ({len(val_files) / len(all_files) * 100:.1f}%)")
        print(f"测试集: {len(test_files)} 文件 ({len(test_files) / len(all_files) * 100:.1f}%)")

        return train_files, val_files, test_files