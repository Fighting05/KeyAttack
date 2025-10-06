# config_manager.py
import os
import json

# # -*- coding: utf-8 -*-
# import sys
# import os

# # 在文件开头添加这几行，解决Windows中文编码问题
# if sys.platform.startswith('win'):
#     import codecs
#     sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
#     sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
class ConfigManager:
    """配置管理类

    负责加载、保存和访问配置信息
    """

    def __init__(self, config_path=None):

        self.config_path = config_path
        self.config = self._get_default_config()

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            print(f"使用默认配置")
            self._ensure_dirs()
            self.save_config(config_path or "config.json")

    def _get_default_config(self):
        """获取默认配置"""
        return {
            "sample_rate": 44100,
            "frame_length": 1024,
            "hop_length": 256,
            "mfcc_coef": 40,
            "min_segment_length": 0.05,
            "max_segment_length": 0.3,
            "energy_threshold_percentile": 85,
            "silence_threshold": 0.03,
            "min_silence_duration": 0.08,
            "bandpass_filter": {
                "lowcut": 100,
                "highcut": 8000,
                "order": 4
            },
            "feature_extraction": {
                "use_mfcc": True,
                "use_spectral": True,
                "use_temporal": True,
                "use_wavelet": True,
                "use_chroma": True
            },
            "models": {
                "traditional": {
                    "use_rf": True,
                    "use_gb": True,
                    "use_svm": False,
                    "ensemble_weight": 0.3
                },
                "deep_learning": {
                    "use_cnn": True,
                    "use_lstm": True,
                    "ensemble_weight": 0.7,
                    "batch_size": 32,
                    "epochs": 100,
                    "patience": 10
                }
            },
            "sequence_model": {
                "use_ngram": True,
                "use_hmm": True,
                "ngram_weight": 0.3,
                "ngram_order": 3
            },
            "paths": {
                "model_dir": "models",
                "data_dir": "data",
                "results_dir": "results",
                "feature_cache": "features.pkl",
                "config_file": "config.json",
                "train_dir": "train",
                "test_dir": "test",
                "original_test_dir": "original_test"
            },
            "audio": {
                "window_size": 0.02,
                "energy_threshold": 5.0,
                "min_time_between_keys": 0.05,
                "segment_before_peak": 0.05,
                "segment_after_peak": 0.15
            },
            "expected_sequence_length": 10
        }

    def _ensure_dirs(self):
        """确保所有必要的目录存在"""
        for key, path in self.config["paths"].items():
            if isinstance(path, str) and not path.endswith('.pkl') and not path.endswith('.json'):
                os.makedirs(path, exist_ok=True)
                print(f"创建目录: {path}")

    def load_config(self, config_path):
        """从文件加载配置

        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
                print(f"从 {config_path} 加载配置")
                self._ensure_dirs()
        except Exception as e:
            print(f"加载配置文件时出错: {e}")

    def save_config(self, config_path=None):
        """保存配置到文件

        Args:
            config_path: 保存配置的路径，若为None则使用初始化时的路径
        """
        save_path = config_path or self.config_path or "config.json"
        try:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                print(f"配置已保存至 {save_path}")
        except Exception as e:
            print(f"保存配置文件时出错: {e}")

    def get(self, key, default=None):
        """获取配置值

        Args:
            key: 配置键，可以使用点号分隔的嵌套键
            default: 默认值，当键不存在时返回

        Returns:
            配置值或默认值
        """
        if "." in key:
            # 处理嵌套键，如 "models.traditional.use_rf"
            parts = key.split(".")
            current = self.config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        else:
            return self.config.get(key, default)

    def set(self, key, value):
        """设置配置值

        Args:
            key: 配置键，可以使用点号分隔的嵌套键
            value: 要设置的值
        """
        if "." in key:
            # 处理嵌套键
            parts = key.split(".")
            current = self.config
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.config[key] = value

    def get_path(self, key):
        """获取配置中的路径并确保目录存在

        Args:
            key: 路径键名

        Returns:
            路径字符串
        """
        path = self.get(f"paths.{key}")
        if path and not path.endswith('.pkl') and not path.endswith('.json'):
            os.makedirs(path, exist_ok=True)
        return path


