import os
from pydub import AudioSegment
import argparse


def convert_m4a_to_wav(input_dir, output_dir=None):
    """
    将指定目录下的所有m4a文件转换为wav格式

    参数:
        input_dir: 包含m4a文件的输入目录
        output_dir: 保存wav文件的输出目录，默认与输入目录相同
    """
    # 如果没有指定输出目录，使用输入目录python m4a.py -i 1171 -o 117
    if output_dir is None:
        output_dir = input_dir

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有m4a文件
    m4a_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.m4a')]
    total_files = len(m4a_files)

    if total_files == 0:
        print(f"在目录 {input_dir} 中没有找到m4a文件")
        return

    print(f"找到 {total_files} 个m4a文件，开始转换...")

    # 转换每个文件
    for i, file in enumerate(m4a_files, 1):
        input_path = os.path.join(input_dir, file)
        output_file = os.path.splitext(file)[0] + '.wav'
        output_path = os.path.join(output_dir, output_file)

        print(f"[{i}/{total_files}] 正在转换: {file} -> {output_file}")

        # 使用pydub加载m4a并导出为wav
        audio = AudioSegment.from_file(input_path, format="m4a")
        audio.export(output_path, format="wav")

    print(f"完成！已将 {total_files} 个文件从m4a转换为wav格式")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量将m4a音频文件转换为wav格式")
    parser.add_argument("-i", "--input", required=True, help="包含m4a文件的输入目录")
    parser.add_argument("-o", "--output", help="保存wav文件的输出目录（可选）")

    args = parser.parse_args()

    convert_m4a_to_wav(args.input, args.output)