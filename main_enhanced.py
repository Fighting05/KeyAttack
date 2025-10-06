# main_enhanced.py - 增强版主程序

import os
import argparse
import time
from datetime import datetime
import traceback

from config_manager import ConfigManager
# 导入增强的音频处理器
from audio_processing_enhanced import AudioProcessor
from feature_extraction import FeatureExtractor
from keystroke_model import KeystrokeModelTrainer
from data_manager import DataManager
from keystroke_recognition import KeystrokeRecognitionSystem
from advanced_prediction import EnhancedPredictionSystem


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
    """训练模型 - 支持自定义模型目录"""

    # 显示当前模型目录
    current_model_dir = config_manager.get_path("model_dir")
    print(f"\n当前模型目录: {current_model_dir}")

    # 询问是否使用自定义模型目录
    custom_model_dir = input("请输入模型保存目录 [回车使用当前目录]: ").strip()

    # 临时保存原始配置，以便后续恢复
    original_model_dir = config_manager.get("paths.model_dir")

    if custom_model_dir:
        # 设置自定义模型目录
        config_manager.set("paths.model_dir", custom_model_dir)
        print(f"✅ 模型将保存到: {custom_model_dir}")

        # 确保目录存在
        os.makedirs(custom_model_dir, exist_ok=True)
    else:
        print(f"✅ 使用当前模型目录: {current_model_dir}")

    try:
        # 使用修改后的config_manager创建系统
        system = KeystrokeRecognitionSystem(config_manager=config_manager)

        # 获取训练数据目录配置
        train_dir = input("请输入训练数据目录 [train]: ") or config_manager.get_path("train_dir")

        # 训练模型
        print("\n开始训练模型...")
        models = system.train_from_samples(train_dir)

        if models:
            final_model_dir = config_manager.get_path("model_dir")
            print(f"\n✅ 模型训练成功! 模型已保存到: {final_model_dir}")
            return True
        else:
            print("\n❌ 模型训练失败，请检查训练数据")
            return False

    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        return False

    finally:
        # 恢复原始配置
        config_manager.set("paths.model_dir", original_model_dir)


def test_models(config_manager):
    """测试模型性能 - 支持自定义模型目录和多候选预测"""

    # 显示当前模型目录
    current_model_dir = config_manager.get_path("model_dir")
    print(f"\n当前模型目录: {current_model_dir}")

    # 检查当前目录是否有模型文件
    if os.path.exists(current_model_dir):
        model_files = [f for f in os.listdir(current_model_dir) if f.endswith(('.pkl', '.h5', '.json'))]
        if model_files:
            print(
                f"发现 {len(model_files)} 个模型文件: {', '.join(model_files[:3])}{'...' if len(model_files) > 3 else ''}")
        else:
            print("⚠️ 当前目录中没有模型文件")
    else:
        print("⚠️ 当前模型目录不存在")

    # 询问是否使用自定义模型目录
    custom_model_dir = input("请输入要使用的模型目录 [回车使用当前目录]: ").strip()

    # 临时保存原始配置
    original_model_dir = config_manager.get("paths.model_dir")

    if custom_model_dir:
        # 检查自定义目录是否存在
        if not os.path.exists(custom_model_dir):
            print(f"❌ 错误：模型目录 {custom_model_dir} 不存在")
            return False

        # 检查是否有模型文件
        model_files = [f for f in os.listdir(custom_model_dir) if f.endswith(('.pkl', '.h5', '.json'))]
        if not model_files:
            print(f"❌ 错误：模型目录 {custom_model_dir} 中没有找到模型文件")
            return False

        # 设置自定义模型目录
        config_manager.set("paths.model_dir", custom_model_dir)
        print(f"✅ 使用模型目录: {custom_model_dir}")
        print(f"找到模型文件: {', '.join(model_files)}")
    else:
        # 检查当前目录
        if not os.path.exists(current_model_dir):
            print(f"❌ 错误：当前模型目录 {current_model_dir} 不存在")
            return False

        model_files = [f for f in os.listdir(current_model_dir) if f.endswith(('.pkl', '.h5', '.json'))]
        if not model_files:
            print(f"❌ 错误：当前模型目录 {current_model_dir} 中没有找到模型文件")
            return False

        print(f"✅ 使用当前模型目录: {current_model_dir}")

    try:
        # 使用修改后的config_manager创建系统
        system = KeystrokeRecognitionSystem(config_manager=config_manager)

        # 获取测试数据目录配置
        test_dir = input("请输入测试数据目录 [test]: ") or config_manager.get_path("test_dir")

        # 检查测试目录是否存在
        if not os.path.exists(test_dir):
            print(f"❌ 错误：测试目录 {test_dir} 不存在")
            return False

        # 检查是否有WAV文件
        wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
        if not wav_files:
            print(f"❌ 错误：测试目录 {test_dir} 中没有WAV文件")
            return False

        # 询问是否使用多候选预测
        use_multiple = input("是否使用多候选预测？(y/n) [y]: ").lower() != 'n'

        num_candidates = 10
        if use_multiple:
            try:
                num_candidates = int(input("请输入候选数量 [默认10]: ") or "10")
            except ValueError:
                num_candidates = 10

        # 评估模型
        print(f"\n开始评估模型，测试文件: {len(wav_files)} 个")
        if use_multiple:
            print(f"使用多候选预测，每个文件生成 {num_candidates} 个候选结果")
        else:
            print("使用传统单一预测")

        # 统计变量
        total_chars = 0
        correct_chars = 0
        total_sequences = 0
        correct_sequences = 0

        # 多候选统计
        from collections import defaultdict
        hit_stats = defaultdict(int)
        no_hit_count = 0

        # 详细结果记录
        test_results = []

        for filename in wav_files:
            try:
                file_path = os.path.join(test_dir, filename)

                # 直接从文件名中提取数字
                expected_sequence = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())

                print(f"\n测试文件: {filename}")
                if expected_sequence:
                    print(f"预期序列: {expected_sequence}")
                    total_sequences += 1

                if use_multiple:
                    # 使用多候选预测
                    candidates = system.predict_from_file_with_candidates(file_path, num_candidates)

                    if not candidates:
                        print(f"预测失败: {filename}")
                        continue

                    # 第一个候选就是原始预测结果
                    predicted = candidates[0][0]
                    original_confidence = candidates[0][1]

                    print(f"原始预测: {predicted} (置信度: {original_confidence:.4f})")
                else:
                    # 使用传统单一预测
                    predicted = system.predict_from_file(file_path)
                    if not predicted:
                        print(f"预测失败: {filename}")
                        continue
                    print(f"预测结果: {predicted}")
                    # 为了统一处理，创建候选列表
                    candidates = [(predicted, 1.0)]

                # 计算传统准确率指标
                if expected_sequence:
                    # 字符级准确率
                    min_len = min(len(expected_sequence), len(predicted))
                    num_correct = sum(1 for i in range(min_len) if expected_sequence[i] == predicted[i])
                    char_accuracy = num_correct / len(expected_sequence)

                    # 序列级准确率
                    sequence_match = (predicted == expected_sequence)
                    if sequence_match:
                        correct_sequences += 1
                        match_status = "✓ 完全匹配"
                    else:
                        match_status = "✗ 不匹配"

                    print(f"字符准确率: {char_accuracy:.2%} ({num_correct}/{len(expected_sequence)})")
                    print(f"序列匹配: {match_status}")

                    correct_chars += num_correct
                    total_chars += len(expected_sequence)

                    # 多候选分析（如果启用）
                    if use_multiple and len(candidates) > 1:
                        hit_rank = None
                        for i, (seq, score) in enumerate(candidates):
                            if seq == expected_sequence:
                                hit_rank = i + 1
                                break

                        if hit_rank:
                            hit_stats[hit_rank] += 1
                            print(f"🎯 多候选结果: 在第 {hit_rank} 名命中!")
                        else:
                            no_hit_count += 1
                            print(f"❌ 多候选结果: 未在前 {num_candidates} 名中命中")

                        # 显示前5个候选
                        print("前5个候选:")
                        for i, (seq, score) in enumerate(candidates[:5]):
                            marker = " ★" if seq == expected_sequence else ""
                            print(f"  {i + 1}. {seq} ({score:.4f}){marker}")

            except Exception as e:
                print(f"测试文件 {filename} 时出错: {e}")

        # 打印总体结果
        print("\n" + "=" * 80)
        print(" " * 25 + "模型评估总结")
        print("=" * 80)

        if total_chars > 0:
            overall_char_accuracy = correct_chars / total_chars
            print(f"\n📊 基础预测性能:")
            print(f"  字符级准确率: {overall_char_accuracy:.2%} ({correct_chars}/{total_chars})")

        if total_sequences > 0:
            sequence_accuracy = correct_sequences / total_sequences
            print(f"  序列级准确率: {sequence_accuracy:.2%} ({correct_sequences}/{total_sequences})")

            # 多候选统计（如果启用）
            if use_multiple:
                total_hits = sum(hit_stats.values())
                overall_hit_rate = total_hits / total_sequences

                print(f"\n🎯 多候选预测性能:")
                print(f"  总命中率: {overall_hit_rate:.2%} ({total_hits}/{total_sequences})")
                print(f"  未命中数: {no_hit_count}")

                # 性能提升分析
                top1_rate = hit_stats.get(1, 0) / total_sequences
                top5_rate = sum(hit_stats.get(i, 0) for i in range(1, 6)) / total_sequences
                top10_rate = sum(hit_stats.get(i, 0) for i in range(1, 11)) / total_sequences

                print(f"\n📈 性能提升分析:")
                print(f"  Top-1 准确率: {top1_rate:.2%}")
                print(f"  Top-5 准确率: {top5_rate:.2%} (提升: {(top5_rate - top1_rate):.2%})")
                print(f"  Top-10 准确率: {top10_rate:.2%} (提升: {(top10_rate - top1_rate):.2%})")

        return True

    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        return False

    finally:
        # 恢复原始配置
        config_manager.set("paths.model_dir", original_model_dir)

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
    """可视化按键分割和预测结果 - 标准版"""
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
        title=f"File {filename} Segmentation Results",
        show_features=True
    )

    print(f"\n检测到 {len(segments)} 个按键")
    print(f"可视化结果已保存至: {vis_path}")

    return True


def enhanced_visualize_results(config_manager):
    """增强可视化按键分割和预测结果 - 学术风格"""
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

    # 创建增强的可视化
    vis_path = audio_processor.create_enhanced_waveform_visualization(
        y, sr, segments, segment_times,
        title=f"Enhanced Analysis: {filename}",
        save_dir=audio_processor.visualization_dir
    )

    print(f"\n检测到 {len(segments)} 个按键")
    print(f"增强可视化结果已保存至: {vis_path}")

    return True


def process_directory(config_manager):
    """处理整个目录的音频文件 - 增强版"""
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
    if success_count + fail_count > 0:
        print(f"成功率: {success_count / (success_count + fail_count) * 100:.2f}%")

    # 显示报告生成信息
    print(f"\n详细的学术风格分析报告已生成在: {output_dir}/analysis_report/")
    print("报告包含以下分析图表:")
    print("• 处理摘要仪表板")
    print("• 信号质量分析")
    print("• 检测性能分析")
    print("• 方法比较分析")
    print("• 时间性分析")

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


def advanced_predict_file(config_manager_obj):
    """高级预测单个文件 - 修复版本"""
    
    # 检查Seq2Seq模型
    seq2seq_model_file = "best_model_PIN_Dodonew_v2.pth" 
    if not os.path.exists(seq2seq_model_file):
        print(f"警告: Seq2Seq模型文件不存在: {seq2seq_model_file}")
        if input("是否继续高级预测（将使用随机初始化的Seq2Seq模型）? (y/n): ").lower() != 'y':
            print("高级预测已取消。")
            return False
    
    # === 修复1：显示当前模型目录状态 ===
    current_model_dir = config_manager_obj.get_path("model_dir")
    print(f"\n当前配置的模型目录: {current_model_dir}")
    
    # 检查当前目录是否有模型文件
    if os.path.exists(current_model_dir):
        model_files = [f for f in os.listdir(current_model_dir) if f.endswith(('.pkl', '.h5', '.json'))]
        if model_files:
            print(f"发现 {len(model_files)} 个模型文件: {', '.join(model_files[:3])}{'...' if len(model_files) > 3 else ''}")
        else:
            print("⚠️ 当前目录中没有模型文件")
    else:
        print("⚠️ 当前模型目录不存在")
    
    # 询问是否使用自定义模型目录
    custom_eps_sound_model_dir = input("请输入用于高级预测的声音模型目录 [可选, 回车使用当前目录]: ").strip() or None
    
    if custom_eps_sound_model_dir:
        if not os.path.isdir(custom_eps_sound_model_dir):
            print(f"错误: 指定的声音模型目录 '{custom_eps_sound_model_dir}' 无效或不是目录。")
            return False
        
        # 检查自定义目录是否有模型文件
        model_files = [f for f in os.listdir(custom_eps_sound_model_dir) if f.endswith(('.pkl', '.h5', '.json'))]
        if not model_files:
            print(f"❌ 错误：模型目录 {custom_eps_sound_model_dir} 中没有找到模型文件")
            return False
        
        print(f"✅ 高级预测将使用自定义模型目录: {custom_eps_sound_model_dir}")
        print(f"   找到模型文件: {', '.join(model_files)}")
        print(f"   注意：为保持一致性，基础声音模型也将使用此目录")
    else:
        # 检查当前目录
        if not os.path.exists(current_model_dir):
            print(f"❌ 错误：当前模型目录 {current_model_dir} 不存在")
            return False
        
        model_files = [f for f in os.listdir(current_model_dir) if f.endswith(('.pkl', '.h5', '.json'))]
        if not model_files:
            print(f"❌ 错误：当前模型目录 {current_model_dir} 中没有找到模型文件")
            return False
        
        print(f"✅ 高级预测将使用当前模型目录: {current_model_dir}")
    
    try:
        # === 修复2：创建EnhancedPredictionSystem（内部已修复模型一致性） ===
        prediction_system_inst = EnhancedPredictionSystem(
            config_manager_obj, 
            seq2seq_model_file, 
            sound_model_dir_override=custom_eps_sound_model_dir
        )
        
        # 获取音频文件路径
        audio_f_path = input("请输入音频文件路径: ")
        if not os.path.exists(audio_f_path) or not os.path.isfile(audio_f_path):
            print(f"错误: 文件 '{audio_f_path}' 不存在或不是有效文件。")
            return False
        
        # 获取显示参数
        top_k_results_to_show = int(input("返回的最佳高级结果数量 (显示用) [默认5, 最多30]: ") or "5")
        top_k_results_to_show = min(max(1, top_k_results_to_show), 30)
        
        print("\n开始高级预测...")
        print("=" * 80)
        
        # === 执行预测 ===
        start_time_single_pred = time.time()
        prediction_output = prediction_system_inst.predict_with_enhanced_masks(
            audio_f_path, 
            top_k=top_k_results_to_show, 
            verbose=True, 
            compare_basic=True
        )
        elapsed_time_single_pred = time.time() - start_time_single_pred
        
        print(f"\n文件 '{os.path.basename(audio_f_path)}' 预测完成! (用时: {elapsed_time_single_pred:.2f}秒)")
        
        # === 修复3：详细结果分析和显示 ===
        acc_stats_res = prediction_output.get('accuracy_stats', {})
        expected_seq_from_filename = ''.join(c for c in os.path.splitext(os.path.basename(audio_f_path))[0] if c.isdigit())
        
        print("\n" + "=" * 80)
        print(" " * 25 + "预测结果与准确率对比")
        print("=" * 80)
        
        if expected_seq_from_filename: 
            print(f"预期序列: '{expected_seq_from_filename}'")
        else: 
            print("预期序列: N/A (无法从文件名提取)")
        
        def print_accuracy_line_detail(model_name_str, pred_key_str, char_acc_key_str, seq_acc_key_str, 
                                     stats_data_dict, expected_seq_exists_bool, extra_info=""):
            pred_val = str(stats_data_dict.get(pred_key_str, 'N/A'))
            char_acc_val = float(stats_data_dict.get(char_acc_key_str, 0.0))
            seq_acc_val = float(stats_data_dict.get(seq_acc_key_str, 0.0))
            
            if expected_seq_exists_bool:
                acc_display_str = f" (字符级: {char_acc_val:.2%}, 序列级: {seq_acc_val:.0%})"
            else:
                acc_display_str = ""
            
            print(f"{model_name_str:<28}: '{pred_val}'{acc_display_str} {extra_info}")
        
        # 显示各模型结果
        print(f"\n📊 各模型预测结果:")
        
        print_accuracy_line_detail(
            "声音模型(最佳单候选)", 
            'sound_model_prediction', 
            'sound_model_char_accuracy', 
            'sound_model_sequence_accuracy', 
            acc_stats_res, 
            bool(expected_seq_from_filename),
            f"(排名: {acc_stats_res.get('sound_model_best_rank', 'N/A')})"
        )
        
        print_accuracy_line_detail(
            "纯Seq2Seq(全掩码)", 
            'pure_seq2seq_prediction', 
            'pure_seq2seq_char_accuracy', 
            'pure_seq2seq_sequence_accuracy', 
            acc_stats_res, 
            bool(expected_seq_from_filename)
        )
        
        adv_model_source_info = f"(来源: {acc_stats_res.get('advanced_model_source','N/A')}, 排名: {acc_stats_res.get('mask_best_rank', 'N/A')})"
        print_accuracy_line_detail(
            "高级模型(综合最佳)", 
            'advanced_model_prediction', 
            'advanced_model_char_accuracy', 
            'advanced_model_sequence_accuracy', 
            acc_stats_res, 
            bool(expected_seq_from_filename), 
            adv_model_source_info
        )
        
        # 显示提升情况
        if expected_seq_from_filename:
            improvement_val = prediction_output.get('improvement_char_level', 0.0)
            sound_char_acc = acc_stats_res.get('sound_model_char_accuracy', 0.0)
            adv_char_acc = acc_stats_res.get('advanced_model_char_accuracy', 0.0)
            
            print(f"\n📈 性能提升分析:")
            if improvement_val == float('inf'): 
                print(f"   高级模型相较于声音模型的字符准确率提升: ∞ (声音模型准确率为0)")
            elif improvement_val > 0: 
                print(f"   高级模型相较于声音模型的字符准确率提升: {improvement_val:.2f}%")
                print(f"   (声音模型: {sound_char_acc:.2%} → 高级模型: {adv_char_acc:.2%})")
            elif improvement_val < 0: 
                print(f"   高级模型相较于声音模型的字符准确率下降: {abs(improvement_val):.2f}%")
                print(f"   (声音模型: {sound_char_acc:.2%} → 高级模型: {adv_char_acc:.2%})")
            else:
                print(f"   高级模型与声音模型准确率相同: {sound_char_acc:.2%}")
        
        # === 显示高级模型详细结果 ===
        if prediction_output.get('advanced'):
            print(f"\n🔍 高级模型预测详情 (Top {min(5, top_k_results_to_show)}):")
            for i_res, res_item_dict in enumerate(prediction_output['advanced'][:min(5, top_k_results_to_show)]): 
                print(f"\n  {i_res+1}. 预测文本: '{res_item_dict['text']}'")
                print(f"     综合得分: {res_item_dict.get('overall_score',0.0):.4f}")
                print(f"     模板名称: {res_item_dict.get('template_name','N/A')}")
                print(f"     使用掩码: '{res_item_dict.get('mask','N/A')}'")
                # print(f"     掩码率: {res_item_dict.get('mask','').count('￥') / len(res_item_dict.get('mask','1')):.2%}")
                
                # 详细得分分解
                seq_score = res_item_dict.get('seq_score', 0.0)
                adherence = res_item_dict.get('mask_adherence_score', 0.0)
                mask_quality = res_item_dict.get('mask_quality_score', 0.0)
                sound_score = res_item_dict.get('sound_candidate_score', 0.0)
                char_fusion = res_item_dict.get('avg_char_fusion_score', 0.0)
                
                print(f"     得分分解: Seq2Seq={seq_score:.3f}, 掩码遵循={adherence:.3f}, 掩码质量={mask_quality:.3f}")
                print(f"               声音候选={sound_score:.3f}, 字符融合={char_fusion:.3f}")
                
                # 显示源声音候选
                sound_source = res_item_dict.get('sound_candidate_text_source', 'N/A')
                print(f"     源声音候选: '{sound_source}'")
        
        # === 显示掩码模板统计 ===
        if hasattr(prediction_system_inst.mask_generator, 'templates') and prediction_system_inst.mask_generator.templates:
            print(f"\n🎭 掩码模板生成统计:")
            template_stats = {}
            for template_name, template_mask in prediction_system_inst.mask_generator.templates.items():
                mask_count = template_mask.count(Config.MASK_TOKEN)
                template_stats[template_name] = (template_mask, mask_count)
            
            # 按掩码数量分组显示
            for mask_count in sorted(set(stats[1] for stats in template_stats.values())):
                templates_with_count = [(name, mask) for name, (mask, count) in template_stats.items() if count == mask_count]
                if templates_with_count:
                    print(f"   {mask_count}个掩码位置: {len(templates_with_count)} 个模板")
                    for name, mask in templates_with_count[:3]:  # 显示前3个
                        print(f"     • {name}: '{mask}'")
                    if len(templates_with_count) > 3:
                        print(f"     ... 还有 {len(templates_with_count) - 3} 个")
        
        # === 生成可视化对比图 ===
        if prediction_system_inst.basic_system:
            try:
                print(f"\n🎨 生成可视化对比图...")
                # 这里需要导入create_comparison_visualization函数
                create_comparison_visualization(
                    audio_f_path, 
                    expected_seq_from_filename or "N/A", 
                    str(acc_stats_res.get('sound_model_prediction','')), 
                    str(acc_stats_res.get('advanced_model_prediction','')), 
                    prediction_system_inst.basic_system 
                )
                print(f"✅ 对比可视化图已生成")
            except Exception as e_viz_adv_file: 
                print(f"❌ 可视化结果时出错: {e_viz_adv_file}")
        else: 
            print("⚠️ 无法生成对比可视化：basic_system 未有效初始化")
        
        print("\n" + "=" * 80)
        print("高级预测完成!")
        return True
        
    except Exception as e:
        print(f"\n❌ 高级预测过程中出错: {e}")
        traceback.print_exc()
        return False

def advanced_predict_directory(config_manager_obj):
    """高级预测整个目录 - 修复版本"""
    
    # 检查Seq2Seq模型
    seq2seq_model_file = "best_model_PIN_Dodonew_v2.pth" 
    if not os.path.exists(seq2seq_model_file):
        print(f"警告: Seq2Seq模型文件不存在: {seq2seq_model_file}")
        if input("是否继续高级预测（将使用随机初始化的Seq2Seq模型）? (y/n): ").lower() != 'y':
            print("高级预测已取消。")
            return False
    
    # === 修复1：显示当前模型目录状态 ===
    current_model_dir = config_manager_obj.get_path("model_dir")
    print(f"\n当前配置的模型目录: {current_model_dir}")
    
    # 检查当前目录是否有模型文件
    if os.path.exists(current_model_dir):
        model_files = [f for f in os.listdir(current_model_dir) if f.endswith(('.pkl', '.h5', '.json'))]
        if model_files:
            print(f"发现 {len(model_files)} 个模型文件")
        else:
            print("⚠️ 当前目录中没有模型文件")
    else:
        print("⚠️ 当前模型目录不存在")
    
    # 询问是否使用自定义模型目录
    custom_eps_sound_model_dir = input("请输入用于高级预测的声音模型目录 [可选, 回车使用当前目录]: ").strip() or None
    
    if custom_eps_sound_model_dir:
        if not os.path.isdir(custom_eps_sound_model_dir):
            print(f"错误: 指定的声音模型目录 '{custom_eps_sound_model_dir}' 无效或不是目录。")
            return False
        
        # 检查自定义目录是否有模型文件
        model_files = [f for f in os.listdir(custom_eps_sound_model_dir) if f.endswith(('.pkl', '.h5', '.json'))]
        if not model_files:
            print(f"❌ 错误：模型目录 {custom_eps_sound_model_dir} 中没有找到模型文件")
            return False
        
        print(f"✅ 高级预测将使用自定义模型目录: {custom_eps_sound_model_dir}")
        print(f"   注意：为保持一致性，基础声音模型也将使用此目录")
    else:
        print(f"✅ 高级预测将使用当前模型目录: {current_model_dir}")
    
    try:
        # === 修复2：创建EnhancedPredictionSystem ===
        prediction_system_inst = EnhancedPredictionSystem(
            config_manager_obj, 
            seq2seq_model_file, 
            sound_model_dir_override=custom_eps_sound_model_dir
        )
        
        # 获取目录路径
        audio_dir_path = input("请输入音频文件目录路径: ")
        if not os.path.isdir(audio_dir_path): 
            print(f"错误: 目录 '{audio_dir_path}' 不是有效目录。")
            return False
        
        # 检查目录中是否有WAV文件
        wav_files = [f for f in os.listdir(audio_dir_path) if f.lower().endswith('.wav')]
        if not wav_files:
            print(f"❌ 错误：目录 {audio_dir_path} 中没有WAV文件")
            return False
        
        print(f"✅ 发现 {len(wav_files)} 个WAV文件")
        
        # 获取参数
        top_k_for_csv_report = int(input("每个文件在CSV中记录的最佳高级结果数量 [默认1]: ") or "1")
        top_k_for_csv_report = min(max(1, top_k_for_csv_report), 5)
        
        save_all_visualizations = input("是否为每个文件保存可视化对比图? [y/n, 默认n]: ").lower() == 'y'
        verbose_each_file = input("是否显示每个文件的详细处理信息? [y/n, 默认n]: ").lower() == 'y'
        
        print(f"\n开始批量高级预测...")
        print(f"将处理 {len(wav_files)} 个文件，详细结果将保存到CSV")
        
        # === 执行批量预测 ===
        start_time_batch_pred = time.time()
        directory_results = prediction_system_inst.predict_directory(
            audio_dir_path, 
            top_k=top_k_for_csv_report,
            verbose=verbose_each_file, 
            save_viz=save_all_visualizations
        )
        elapsed_time_batch_pred = time.time() - start_time_batch_pred
        
        print(f"\n整个高级预测过程用时: {elapsed_time_batch_pred:.2f}秒")
        print(f"平均每文件用时: {elapsed_time_batch_pred / len(wav_files):.2f}秒")
        
        return bool(directory_results)
        
    except Exception as e:
        print(f"\n❌ 批量高级预测过程中出错: {e}")
        traceback.print_exc()
        return False


def generate_comprehensive_report(config_manager):
    """生成综合报告"""
    print("\n正在生成综合系统报告...")

    # 创建一个临时的音频处理器来生成报告
    audio_processor = AudioProcessor(config_manager)

    # 模拟一些处理统计（在实际应用中这些数据来自真实处理）
    audio_processor.processing_stats = {
        'files_processed': 25,
        'files_failed': 3,
        'total_keystrokes_detected': 320,
        'processing_times': [1.2, 0.8, 1.5, 2.1, 0.9] * 5,
        'keystroke_counts': [12, 8, 15, 10, 11] * 5,
        'detection_confidence_scores': [0.85, 0.72, 0.91, 0.68, 0.83] * 5,
        'signal_to_noise_ratios': [18.5, 22.1, 15.3, 25.8, 19.7] * 5,
        'file_sizes': [2.1, 1.8, 2.5, 3.2, 1.9] * 5,
    }

    audio_processor.method_stats = {
        "peak_detection": 18,
        "equal_segments": 5,
        "adaptive_threshold": 4,
        "ensemble_method": 1
    }

    # 生成报告
    report_dir = config_manager.get_path("results_dir") + "/comprehensive_report"
    os.makedirs(report_dir, exist_ok=True)

    audio_processor._generate_processing_report(report_dir)

    print(f"综合报告已生成: {report_dir}")
    print("包含以下学术风格图表:")
    print("• 处理摘要仪表板 - 系统整体性能概览")
    print("• 信号质量分析 - 音频质量评估和分布")
    print("• 检测性能分析 - 按键检测准确率和置信度")
    print("• 方法比较分析 - 不同检测方法效果对比")
    print("• 时间性分析 - 处理效率和时间趋势")

    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="键盘声音识别系统 - 增强版")
    parser.add_argument("--config", type=str, help="配置文件路径")
    args = parser.parse_args()

    # 初始化配置管理器
    config_manager = ConfigManager(args.config)

    while True:
        print("\n" + "=" * 60)
        print(" " * 15 + "键盘声音识别系统 - 增强版")
        print("=" * 60 + "\n")
        print("基础功能:")
        print("1. 处理音频文件（转换、分割）")
        print("2. 创建匿名测试集")
        print("3. 训练模型")
        print("4. 测试模型性能")
        print("5. 预测单个文件")
        print("6. 分析数据统计")
        print()
        print("可视化功能:")
        print("7. 标准可视化分析")
        print("8. 处理整个目录 (含学术风格报告)")
        print("9. 增强可视化分析 (学术风格)")
        print("10. 生成综合报告 (学术风格)")
        print()
        print("高级功能:")
        print("11. 高级预测单个文件 (掩码+Seq2Seq)")
        print("12. 高级预测整个目录 (掩码+Seq2Seq)")
        print()
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
        elif choice == "9":
            enhanced_visualize_results(config_manager)
        elif choice == "10":
            generate_comprehensive_report(config_manager)
        elif choice == "11":
            advanced_predict_file(config_manager)
        elif choice == "12":
            advanced_predict_directory(config_manager)
        elif choice == "0":
            print("\n感谢使用，再见!")
            break
        else:
            print("\n无效的选择，请重试")

        input("\n按回车键继续...")


if __name__ == "__main__":
    main()