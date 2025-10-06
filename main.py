# main.py - 简化和整合版

import os
import argparse
import time
from datetime import datetime
import traceback

from config_manager import ConfigManager
from audio_processing import AudioProcessor
from feature_extraction import FeatureExtractor
from keystroke_model import KeystrokeModelTrainer
from data_manager import DataManager
from keystroke_recognition import KeystrokeRecognitionSystem

def process_audio_files(config_manager):
    """处理音频文件：转换、分割训练/测试集"""
    data_manager = DataManager(config_manager)

    # 获取配置
    source_dir = input("请输入源音频目录路径 [newdata]: ") or "newdata"
    ffmpeg_path = input("请输入ffmpeg路径 (可选): ") or None
    ffprobe_path = input("请输入ffprobe路径 (可选): ") or None
    format_from = input("请输入源文件格式 [m4a]: ") or "m4a"

    # 转换音频文件
    train_success, test_success = data_manager.convert_audio_files(
        source_dir=source_dir,
        format_from=format_from,
        format_to="wav",
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path
    )

    if train_success + test_success > 0:
        print("\n音频转换成功!")
        return True
    else:
        print("\n音频转换失败，请检查配置和文件路径")
        return False

def create_test_set(config_manager):
    """创建匿名测试集"""
    data_manager = DataManager(config_manager)

    # 获取配置
    original_test_dir = input("请输入原始测试数据目录 [test]: ") or config_manager.get_path("test_dir")
    new_test_dir = input("请输入新测试数据目录 [anonymized_test]: ") or "anonymized_test"
    config_manager.set("paths.original_test_dir", new_test_dir)

    # 创建匿名测试集
    new_dir, mapping_file = data_manager.create_anonymous_test_set(
        original_test_dir=original_test_dir,
        new_test_dir=new_test_dir
    )

    print(f"\n匿名测试集创建成功:")
    print(f"测试集目录: {new_dir}")
    print(f"映射文件: {mapping_file}")

    return new_dir, mapping_file

def train_models(config_manager):
    """训练模型"""
    # 使用config_path而不是config_manager
    system = KeystrokeRecognitionSystem(config_path=config_manager.config_path)

    # 获取配置
    train_dir = input("请输入训练数据目录 [train]: ") or config_manager.get_path("train_dir")

    # 训练模型
    print("\n开始训练模型...")
    models = system.train_from_samples(train_dir)

    if models:
        print("\n模型训练成功!")
        return True
    else:
        print("\n模型训练失败，请检查训练数据")
        return False

def test_models(config_manager):
    """测试模型性能"""
    # 使用config_path而不是config_manager
    system = KeystrokeRecognitionSystem(config_path=config_manager.config_path)

    # 获取配置
    test_dir = input("请输入测试数据目录 [test]: ") or config_manager.get_path("test_dir")

    # 检查测试目录是否存在
    if not os.path.exists(test_dir):
        print(f"错误：测试目录 {test_dir} 不存在")
        return False

    # 检查是否有WAV文件
    wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
    if not wav_files:
        print(f"错误：测试目录 {test_dir} 中没有WAV文件")
        return False

    # 评估模型
    print(f"\n开始评估模型，测试文件: {len(wav_files)} 个")

    total_files = 0
    correct_predictions = 0

    for filename in wav_files:
        try:
            file_path = os.path.join(test_dir, filename)

            # 直接从文件名中提取数字
            expected_sequence = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())

            print(f"\n测试文件: {filename}")
            if expected_sequence:
                print(f"预期序列: {expected_sequence}")

            # 预测
            predicted = system.predict_from_file(file_path, verbose=True)

            # 计算准确率
            if expected_sequence:
                min_len = min(len(expected_sequence), len(predicted))
                num_correct = sum(1 for i in range(min_len) if expected_sequence[i] == predicted[i])
                accuracy = num_correct / len(expected_sequence)

                print(f"预测序列: {predicted}")
                print(f"准确率: {accuracy:.2%} ({num_correct}/{len(expected_sequence)})")

                correct_predictions += num_correct
                total_files += len(expected_sequence)

        except Exception as e:
            print(f"测试文件 {filename} 时出错: {e}")

    if total_files > 0:
        print(f"\n总体准确率: {correct_predictions / total_files:.2%}")

    return True

def predict_file(config_manager):
    """预测单个文件"""
    # 使用config_path而不是config_manager
    system = KeystrokeRecognitionSystem(config_path=config_manager.config_path)

    # 获取文件路径
    file_path = input("请输入音频文件路径: ")
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return False

    # 预测
    print(f"\n开始预测文件: {file_path}")
    predicted = system.predict_from_file(file_path, verbose=True)

    print(f"\n预测结果: {predicted}")

    # 可视化
    try:
        vis_path = system.visualize_predictions(file_path, predicted)
        print(f"可视化结果已保存至: {vis_path}")
    except Exception as e:
        print(f"可视化时出错: {e}")

    return True

def visualize_results(config_manager):
    """可视化按键分割和预测结果"""
    # 使用config_path而不是config_manager
    audio_processor = AudioProcessor(config_manager)

    # 获取文件路径
    file_path = input("请输入音频文件路径: ")
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return False

    # 加载音频
    try:
        y, sr = audio_processor.load_audio(file_path)
        print(f"音频加载成功，采样率: {sr}Hz, 长度: {len(y) / sr:.2f}秒")
    except Exception as e:
        print(f"加载音频失败: {e}")
        return False

    # 获取预期按键数量
    expected_length = None
    filename = os.path.basename(file_path)
    digit_part = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
    if digit_part:
        expected_length = len(digit_part)
        print(f"从文件名猜测的预期按键数量: {expected_length}")

    # 检测按键
    segments, segment_times, _ = audio_processor.detect_keystrokes(y, sr, expected_length)

    # 显示分割结果
    vis_path = audio_processor.visualize_audio(
        y, sr, segments, segment_times,
        title=f"文件 {filename} 的分割结果",
        show_features=True
    )

    print(f"\n检测到 {len(segments)} 个按键")
    print(f"可视化结果已保存至: {vis_path}")

    return True

def process_directory(config_manager):
    """处理整个目录的音频文件"""
    audio_processor = AudioProcessor(config_manager)

    # 获取目录路径
    input_dir = input("请输入音频文件目录路径: ")
    output_dir = input("请输入输出目录路径 [默认: dataset]: ") or "dataset"

    # 处理目录
    print(f"\n开始处理目录 {input_dir} 中的音频文件...")
    success_count, fail_count = audio_processor.process_audio_files(input_dir, output_dir)

    # 打印结果摘要
    print(f"\n处理完成:")
    print(f"成功处理: {success_count} 个文件")
    print(f"失败处理: {fail_count} 个文件")
    print(f"成功率: {success_count / (success_count + fail_count) * 100:.2f}%")

    return True

def analyze_data(config_manager):
    """分析数据集统计信息"""
    data_manager = DataManager(config_manager)

    # 获取目录
    data_dir = input("请输入数据目录 [data]: ") or config_manager.get_path("data_dir")

    # 分析数据集
    stats = data_manager.analyze_dataset(data_dir, show_distribution=True)

    if stats:
        print("\n数据分析完成!")
        return stats
    else:
        print("\n数据分析失败")
        return None


def advanced_predict_file(config_manager):
    """使用高级预测功能预测单个文件"""
    from advanced_prediction import advanced_predict_file as predict_file_advanced

    # 调用高级预测功能
    result = predict_file_advanced(config_manager)
    return result


def advanced_predict_directory(config_manager):
    """使用高级预测功能预测整个目录"""
    from advanced_prediction import advanced_predict_directory as predict_dir_advanced

    # 调用高级预测功能
    result = predict_dir_advanced(config_manager)
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="键盘声音识别系统")
    parser.add_argument("--config", type=str, help="配置文件路径")
    args = parser.parse_args()

    # 初始化配置管理器
    config_manager = ConfigManager(args.config)

    while True:
        print("\n" + "=" * 50)
        print(" " * 15 + "键盘声音识别系统")
        print("=" * 50 + "\n")
        print("1. 处理音频文件（转换、分割）")
        print("2. 创建匿名测试集")
        print("3. 训练模型")
        print("4. 测试模型性能")
        print("5. 预测单个文件")
        print("6. 分析数据统计")
        print("7. 可视化分析")
        print("8. 处理整个目录")
        print("9. 高级预测单个文件 (掩码+Seq2Seq)") # 新增选项
        print("10. 高级预测整个目录 (掩码+Seq2Seq)") # 新增选项
        print("0. 退出")

        choice = input("\n请选择操作: ")

        if choice == "1":
            process_audio_files(config_manager)
        elif choice == "2":
            create_test_set(config_manager)
        elif choice == "3":
            train_models(config_manager)
        elif choice == "4":
            test_models(config_manager)
        elif choice == "5":
            predict_file(config_manager)
        elif choice == "6":
            analyze_data(config_manager)
        elif choice == "7":
            visualize_results(config_manager)
        elif choice == "8":
            process_directory(config_manager)
        elif choice == "9": # 新增选项
            advanced_predict_file(config_manager)
        elif choice == "10": # 新增选项
            advanced_predict_directory(config_manager)
        elif choice == "0":
            print("\n感谢使用，再见!")
            break
        else:
            print("\n无效的选择，请重试")

        input("\n按回车键继续...")

if __name__ == "__main__":
    main()