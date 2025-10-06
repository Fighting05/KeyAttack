import os
import shutil
import argparse


def extract_wav_files(input_dir, output_dir):
    """
    将指定目录下的所有WAV文件提取（复制）到另一个目录

    参数:
        input_dir: 包含WAV文件的输入目录
        output_dir: 保存WAV文件的输出目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        except Exception as e:
            print(f"创建输出目录失败: {e}")
            return

    # 获取所有WAV文件
    try:
        all_files = os.listdir(input_dir)
        wav_files = [f for f in all_files if f.lower().endswith('.wav')]
        total_files = len(wav_files)

        if total_files == 0:
            print(f"在目录 {input_dir} 中没有找到WAV文件")
            return

        print(f"找到 {total_files} 个WAV文件，开始提取...")

        # 复制每个文件
        for i, file in enumerate(wav_files, 1):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            print(f"[{i}/{total_files}] 正在复制: {file}")

            try:
                shutil.copy2(input_path, output_path)
            except Exception as e:
                print(f"复制文件 {file} 失败: {e}")

        print(f"完成！已将 {total_files} 个WAV文件提取到 {output_dir}")

    except Exception as e:
        print(f"处理目录时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取目录中的WAV文件到另一个目录")
    parser.add_argument("-i", "--input", required=True, help="包含WAV文件的输入目录")
    parser.add_argument("-o", "--output", required=True, help="保存WAV文件的输出目录")

    args = parser.parse_args()

    extract_wav_files(args.input, args.output)