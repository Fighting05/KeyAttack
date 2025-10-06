import os
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import librosa


def play_wav_file(file_path):
    """播放单个WAV文件并显示其波形"""
    print(f"播放文件: {file_path}")

    # 加载WAV文件
    data, sr = sf.read(file_path)

    # 显示波形
    plt.figure(figsize=(10, 4))
    time_axis = np.linspace(0, len(data) / sr, len(data))
    plt.plot(time_axis, data)
    plt.title(f"文件: {os.path.basename(file_path)}")
    plt.xlabel('时间 (秒)')
    plt.ylabel('振幅')
    plt.grid(True)
    plt.show(block=False)

    # 播放音频
    sd.play(data, sr)

    # 等待播放完成
    duration = len(data) / sr
    print(f"音频长度: {duration:.2f}秒")
    time.sleep(duration + 0.5)

    # 关闭图形
    plt.close()


def browse_wav_files(directory):
    """浏览目录中的所有WAV文件"""
    # 确保使用绝对路径
    directory = os.path.abspath(directory)
    print(f"查找目录: {directory}")

    if not os.path.exists(directory):
        print(f"错误: 目录 {directory} 不存在")
        return

    # 获取所有WAV文件 - 不区分大小写
    wav_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.wav'):
            wav_files.append(os.path.join(directory, filename))

    if not wav_files:
        print(f"在 {directory} 中未找到WAV文件")
        all_files = os.listdir(directory)[:10]
        print(f"目录中的前10个文件: {all_files}")
        return

    # 将路径转换为Path对象
    wav_files = [Path(f) for f in wav_files]

    print(f"在 {directory} 中找到 {len(wav_files)} 个WAV文件")

    # 按文件名排序
    wav_files.sort()

    # 分类按键文件
    key_files = {}
    for wav_file in wav_files:
        # 从文件名中提取按键信息
        filename = wav_file.name
        parts = filename.split('_')
        if len(parts) >= 1:
            key = parts[0]  # 假设第一部分是按键编号

            if key not in key_files:
                key_files[key] = []
            key_files[key].append(wav_file)

    # 显示按键分类
    print("\n按键分类:")
    for key, files in sorted(key_files.items()):
        print(f"按键 {key}: {len(files)} 个文件")

    # 浏览选项
    while True:
        print("\n请选择浏览方式:")
        print("1. 按顺序浏览所有文件")
        print("2. 浏览特定按键的文件")
        print("3. 浏览单个文件")
        print("0. 退出")

        choice = input("请选择: ")

        if choice == "1":
            # 按顺序浏览
            num_to_play = int(input(f"要播放多少个文件? (最大 {len(wav_files)}): ") or "10")
            files_to_play = wav_files[:min(num_to_play, len(wav_files))]

            for file_path in files_to_play:
                play_wav_file(file_path)

                # 询问是否继续
                if input("继续播放下一个? (y/n): ").lower() != 'y':
                    break

        elif choice == "2":
            # 浏览特定按键
            print("\n可用的按键:")
            sorted_keys = sorted(key_files.keys())
            for i, key in enumerate(sorted_keys):
                print(f"{i + 1}. 按键 {key} ({len(key_files[key])} 个文件)")

            key_idx = int(input("请选择按键 (输入序号): ")) - 1
            if 0 <= key_idx < len(sorted_keys):
                selected_key = sorted_keys[key_idx]
                files_for_key = key_files[selected_key]

                print(f"将播放按键 {selected_key} 的 {len(files_for_key)} 个文件")
                for file_path in files_for_key:
                    play_wav_file(file_path)

                    # 询问是否继续
                    if input("继续播放下一个? (y/n): ").lower() != 'y':
                        break
            else:
                print("无效的选择")

        elif choice == "3":
            # 浏览单个文件
            for i, file_path in enumerate(wav_files):
                print(f"{i + 1}. {file_path.name}")
                if (i + 1) % 20 == 0 and i + 1 < len(wav_files):
                    if input("显示更多文件? (y/n): ").lower() != 'y':
                        break

            file_idx = int(input("请选择文件 (输入序号): ")) - 1
            if 0 <= file_idx < len(wav_files):
                play_wav_file(wav_files[file_idx])
            else:
                print("无效的选择")

        elif choice == "0":
            print("退出浏览")
            break

        else:
            print("无效的选择，请重试")


def analyze_original_and_segments(original_dir, segments_dir):
    """分析原始音频和分割后的片段"""
    # 确保使用绝对路径
    original_dir = os.path.abspath(original_dir)
    segments_dir = os.path.abspath(segments_dir)

    print(f"原始音频目录: {original_dir}")
    print(f"分割片段目录: {segments_dir}")

    # 检查目录是否存在
    if not os.path.exists(original_dir):
        print(f"错误: 原始目录 {original_dir} 不存在")
        return
    if not os.path.exists(segments_dir):
        print(f"错误: 分割目录 {segments_dir} 不存在")
        return

    # 获取原始WAV文件 - 不区分大小写
    original_files = []
    for filename in os.listdir(original_dir):
        if filename.lower().endswith('.wav'):
            original_files.append(os.path.join(original_dir, filename))

    if not original_files:
        print(f"在 {original_dir} 中未找到原始WAV文件")
        all_files = os.listdir(original_dir)[:10]
        print(f"目录中的前10个文件: {all_files}")
        return

    # 将路径转换为Path对象
    original_files = [Path(f) for f in original_files]
    original_files.sort()

    print(f"在 {original_dir} 中找到 {len(original_files)} 个原始WAV文件")

    # 选择要分析的文件
    for i, file_path in enumerate(original_files):
        print(f"{i + 1}. {file_path.name}")
        if (i + 1) % 20 == 0 and i + 1 < len(original_files):
            if input("显示更多文件? (y/n): ").lower() != 'y':
                break

    file_idx = int(input("请选择要分析的原始文件 (输入序号): ")) - 1
    if not 0 <= file_idx < len(original_files):
        print("无效的选择")
        return

    original_file = original_files[file_idx]
    basename = original_file.stem

    # 查找对应的分割片段
    segments = []
    for filename in os.listdir(segments_dir):
        if filename.lower().endswith('.wav') and basename in filename:
            segments.append(os.path.join(segments_dir, filename))

    # 将路径转换为Path对象
    segments = [Path(f) for f in segments]

    if not segments:
        print(f"未找到与 {basename} 相关的分割片段")
        return

    # 尝试按文件名中的索引排序
    try:
        segments.sort(key=lambda x: int(x.stem.split('_')[-1]))
    except:
        # 如果排序失败，按文件名排序
        segments.sort()

    print(f"找到 {len(segments)} 个分割片段")

    # 加载原始文件
    orig_data, sr = sf.read(str(original_file))

    # 播放并分析
    # 创建图形
    plt.figure(figsize=(12, 6))

    # 绘制原始波形
    plt.subplot(2, 1, 1)
    plt.title(f"原始文件: {basename}")
    time_axis = np.linspace(0, len(orig_data) / sr, len(orig_data))
    plt.plot(time_axis, orig_data)
    plt.grid(True)

    # 在原始波形上标记分割点
    for i, segment_file in enumerate(segments):
        # 从文件名猜测按键
        key_id = segment_file.stem.split('_')[0]

        # 加载片段获取长度
        seg_data, _ = sf.read(str(segment_file))
        seg_duration = len(seg_data) / sr

        # 从文件名猜测位置
        # 这是个估计，实际位置可能不准确
        # 假设是按顺序分割的
        estimated_start = i * (len(orig_data) / sr) / len(segments)

        # 在原始波形上标记
        plt.axvspan(estimated_start, estimated_start + seg_duration,
                    alpha=0.2, color=f'C{i % 10}', label=f'片段 {i + 1} (按键 {key_id})')

    plt.legend(loc='upper right')

    # 播放原始音频
    print(f"\n播放原始文件: {basename}")
    sd.play(orig_data, sr)
    time.sleep(len(orig_data) / sr + 0.5)

    # 显示和播放每个片段
    for i, segment_file in enumerate(segments):
        # 清除下面的子图
        if i > 0:
            plt.subplot(2, 1, 2).clear()

        # 加载片段
        seg_data, seg_sr = sf.read(str(segment_file))

        # 显示片段波形
        plt.subplot(2, 1, 2)
        key_id = segment_file.stem.split('_')[0]
        plt.title(f"片段 {i + 1}/{len(segments)}: 按键 {key_id}")
        seg_time = np.linspace(0, len(seg_data) / seg_sr, len(seg_data))
        plt.plot(seg_time, seg_data, color=f'C{i % 10}')
        plt.grid(True)

        # 更新图形
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

        # 播放片段
        print(f"播放片段 {i + 1}/{len(segments)}: 按键 {key_id}")
        sd.play(seg_data, seg_sr)

        # 等待播放完成
        time.sleep(len(seg_data) / seg_sr + 0.5)

        # 询问是否继续
        if i < len(segments) - 1:
            if input("继续播放下一个片段? (y/n): ").lower() != 'y':
                break

    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" " * 15 + "WAV文件评估工具")
    print("=" * 50 + "\n")

    # 打印当前工作目录
    print(f"当前工作目录: {os.getcwd()}")

    # 打印音频目录
    audio_result_dir = os.path.abspath("audio_result")
    test797_dir = os.path.abspath("test797")
    print(f"音频结果目录: {audio_result_dir} (存在: {os.path.exists(audio_result_dir)})")
    print(f"测试797目录: {test797_dir} (存在: {os.path.exists(test797_dir)})\n")

    while True:
        print("\n请选择操作:")
        print("1. 浏览分割后的WAV文件")
        print("2. 分析原始音频和分割片段")
        print("0. 退出")

        choice = input("请选择: ")

        if choice == "1":
            segments_dir = input("请输入分割后WAV文件的目录 (默认为audio_result): ") or "audio_result"
            # 使用绝对路径
            if not os.path.isabs(segments_dir):
                segments_dir = os.path.abspath(segments_dir)
            browse_wav_files(segments_dir)

        elif choice == "2":
            original_dir = input("请输入原始WAV文件的目录 (默认为test797): ") or "test797"
            segments_dir = input("请输入分割后WAV文件的目录 (默认为audio_result): ") or "audio_result"

            # 使用绝对路径
            if not os.path.isabs(original_dir):
                original_dir = os.path.abspath(original_dir)
            if not os.path.isabs(segments_dir):
                segments_dir = os.path.abspath(segments_dir)

            analyze_original_and_segments(original_dir, segments_dir)

        elif choice == "0":
            print("退出程序")
            break

        else:
            print("无效的选择，请重试")