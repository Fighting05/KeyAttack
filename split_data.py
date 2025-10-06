# split_data.py
import os
import shutil
import random
import argparse
from pathlib import Path
import time


def split_data(data_dir="data", train_dir="train", test_dir="test",
               train_ratio=0.8, file_ext=".wav", copy_mode=True, seed=42):
    """
    将数据目录中的文件随机划分为训练集和测试集

    参数:
        data_dir: 源数据目录
        train_dir: 训练数据输出目录
        test_dir: 测试数据输出目录
        train_ratio: 训练集比例 (0-1之间)
        file_ext: 文件扩展名筛选
        copy_mode: True为复制文件，False为移动文件
        seed: 随机种子

    返回:
        tuple: (训练文件数, 测试文件数)
    """
    # 标准化路径
    data_dir = Path(data_dir)
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    # 确保目录存在
    if not data_dir.exists():
        print(f"错误: 数据目录 '{data_dir}' 不存在")
        return 0, 0

    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # 获取所有指定扩展名的文件
    files = [f for f in data_dir.glob(f"*{file_ext}")]

    if not files:
        print(f"警告: 在 '{data_dir}' 中未找到 '{file_ext}' 文件")
        return 0, 0

    # 设置随机种子并打乱文件顺序
    random.seed(seed)
    random.shuffle(files)

    # 计算分割点
    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # 处理文件
    train_copied = 0
    test_copied = 0

    # 复制或移动训练文件
    print(f"\n处理训练文件 ({len(train_files)} 个):")
    for i, file in enumerate(train_files):
        try:
            dest = train_dir / file.name
            if copy_mode:
                shutil.copy2(file, dest)
                print(f"  复制文件 [{i + 1}/{len(train_files)}]: {file.name}")
            else:
                shutil.move(file, dest)
                print(f"  移动文件 [{i + 1}/{len(train_files)}]: {file.name}")
            train_copied += 1
        except Exception as e:
            print(f"  处理 {file.name} 时出错: {str(e)}")

    # 复制或移动测试文件
    print(f"\n处理测试文件 ({len(test_files)} 个):")
    for i, file in enumerate(test_files):
        try:
            dest = test_dir / file.name
            if copy_mode:
                shutil.copy2(file, dest)
                print(f"  复制文件 [{i + 1}/{len(test_files)}]: {file.name}")
            else:
                shutil.move(file, dest)
                print(f"  移动文件 [{i + 1}/{len(test_files)}]: {file.name}")
            test_copied += 1
        except Exception as e:
            print(f"  处理 {file.name} 时出错: {str(e)}")

    # 打印统计信息
    print("\n分割数据完成!")
    print(f"源数据目录: {data_dir}")
    print(f"总文件数: {len(files)}")
    print(f"训练集: {train_copied}/{len(train_files)} 文件 ({train_copied / len(files) * 100:.1f}%) -> {train_dir}")
    print(f"测试集: {test_copied}/{len(test_files)} 文件 ({test_copied / len(files) * 100:.1f}%) -> {test_dir}")

    return train_copied, test_copied


def main():
    """主函数"""
    # 设置参数
    config = {
        "data_dir": "r9000p",
        "train_dir": "r9000p_train",
        "test_dir": "r9000p_test",
        "train_ratio": 0.8,
        "file_ext": ".wav",
        "copy_mode": True,
        "seed": 42
    }

    # 确认操作提示
    if not config["copy_mode"]:
        confirm = input("警告: 使用移动模式将从源目录移除文件。是否继续? (y/n): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return

    start_time = time.time()

    # 执行数据划分
    train_count, test_count = split_data(
        data_dir=config["data_dir"],
        train_dir=config["train_dir"],
        test_dir=config["test_dir"],
        train_ratio=config["train_ratio"],
        file_ext=config["file_ext"],
        copy_mode=config["copy_mode"],
        seed=config["seed"]
    )

    end_time = time.time()
    duration = end_time - start_time

    if train_count + test_count > 0:
        print(f"\n用时: {duration:.2f}秒")


if __name__ == "__main__":
    main()