# keystroke_model.py
import os
import numpy as np
import pickle
import json
import time
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import torch


class KeystrokeModelTrainer:
    """高级模型训练类，支持深度学习和传统机器学习模型
    """

    def __init__(self, config_manager):
        """初始化模型训练器

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager
        self.models = {}
        self.scaler = StandardScaler()
        self.initialize_models()

    def initialize_models(self):
        """初始化所有支持的模型"""
        # 传统机器学习模型
        if self.config.get("models.traditional.use_rf", True):
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )

        if self.config.get("models.traditional.use_gb", True):
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )

        # 深度学习模型在训练时创建

    def train(self, X, y):
        """训练所有选择的模型

        Args:
            X: 特征矩阵
            y: 标签向量

        Returns:
            dict: 训练好的模型字典
        """
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)

        # 确保模型目录存在
        model_dir = self.config.get_path("model_dir")

        # 保存标准化器
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        print("\n开始训练传统机器学习模型...")
        # 训练传统机器学习模型
        for name, model in self.models.items():
            if name not in ['cnn', 'lstm']:  # 跳过深度学习模型
                print(f"训练 {name} 模型...")
                model.fit(X_scaled, y)
                print(f"{name} 模型训练完成")

                # 保存模型
                with open(os.path.join(model_dir, f'{name}.pkl'), 'wb') as f:
                    pickle.dump(model, f)

        print("\n开始训练深度学习模型...")
        # 训练深度学习模型
        if self.config.get("models.deep_learning.use_cnn", True):
            # 准备深度学习数据
            classes = np.unique(y)
            class_to_idx = {str(cls): i for i, cls in enumerate(classes)}  # 确保键是字符串
            y_idx = np.array([class_to_idx[str(cls)] for cls in y])

            # 保存类别索引映射
            with open(os.path.join(model_dir, 'class_indices.json'), 'w') as f:
                json.dump(class_to_idx, f)

            # 转换为分类格式
            y_cat = to_categorical(y_idx)

            # 为CNN准备输入数据
            X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

            # 创建CNN模型
            cnn_model = self.create_cnn_model(input_shape=X_cnn.shape[1:], num_classes=len(classes))

            try:
                os.makedirs(model_dir, exist_ok=True)

                # 使用绝对路径和新的Keras格式
                model_save_path = os.path.abspath(os.path.join(model_dir, 'cnn_model.keras'))
                print(f"模型将保存至绝对路径: {model_save_path}")
            except Exception as e:
                print(f"创建模型目录时出错: {e}")
                # 使用当前目录的相对路径
                model_save_path = "./cnn_model.keras"
                print(f"改为保存至当前目录: {model_save_path}")

            callbacks = [
                EarlyStopping(
                    monitor='loss',
                    patience=self.config.get("models.deep_learning.patience", 10),
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    model_save_path,
                    save_best_only=True,
                    monitor='loss'
                )
            ]

            print("训练CNN模型...")
            # 训练模型
            history = cnn_model.fit(
                X_cnn, y_cat,
                epochs=self.config.get("models.deep_learning.epochs", 100),
                batch_size=self.config.get("models.deep_learning.batch_size", 32),
                callbacks=callbacks,
                verbose=2
            )
            print("CNN模型训练完成")

            # 保存训练历史
            history_dict = {
                'loss': history.history['loss'],
                'accuracy': history.history['accuracy']
            }

            results_dir = self.config.get_path("results_dir")
            with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
                json.dump(history_dict, f)

            # 可视化训练历史
            self.plot_training_history(history)

            # 保存CNN模型
            cnn_model.save(os.path.join(model_dir, 'cnn_model.h5'))
            self.models['cnn'] = cnn_model

        # 添加LSTM模型训练
        if self.config.get("models.deep_learning.use_lstm", True):
            # 使用与CNN相同的准备数据步骤
            print("训练LSTM模型...")

            # 为LSTM准备序列数据
            # 这里我们使用与CNN相同的数据格式，但实际项目中可能需要调整
            X_lstm = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

            # 创建LSTM模型
            lstm_model = self.create_lstm_model(input_shape=X_lstm.shape[1:], num_classes=len(classes))

            # 定义回调函数
            callbacks = [
                EarlyStopping(
                    monitor='loss',
                    patience=self.config.get("models.deep_learning.patience", 10),
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    os.path.join(model_dir, 'lstm_model.h5'),
                    save_best_only=True,
                    monitor='loss'
                )
            ]

            # 训练模型
            history = lstm_model.fit(
                X_lstm, y_cat,
                epochs=self.config.get("models.deep_learning.epochs", 100),
                batch_size=self.config.get("models.deep_learning.batch_size", 32),
                callbacks=callbacks,
                verbose=2
            )
            print("LSTM模型训练完成")

            # 保存LSTM模型
            lstm_model.save(os.path.join(model_dir, 'lstm_model.h5'))
            self.models['lstm'] = lstm_model

        return self.models

    def create_cnn_model(self, input_shape, num_classes):
        """创建一个1D CNN模型用于音频特征分类

        Args:
            input_shape: 输入形状
            num_classes: 类别数量

        Returns:
            keras模型
        """
        model = Sequential([
            # 第一个卷积层
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),

            # 第二个卷积层
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),

            # 第三个卷积层
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),

            # 全连接层
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        # 编译模型
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        return model

    def create_lstm_model(self, input_shape, num_classes):
        """创建一个LSTM模型用于序列特征分类

        Args:
            input_shape: 输入形状
            num_classes: 类别数量

        Returns:
            keras模型
        """
        model = Sequential([
            # LSTM层
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.25),

            # 第二个LSTM层
            Bidirectional(LSTM(64)),
            Dropout(0.25),

            # 全连接层
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        # 编译模型
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        return model

    def plot_training_history(self, history):
        """可视化模型训练历史

        Args:
            history: keras训练历史对象
        """
        results_dir = self.config.get_path("results_dir")
        plt.figure(figsize=(12, 4))

        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 绘制准确率
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_history.png'))
        plt.close()
        print(f"训练历史图表已保存至 {os.path.join(results_dir, 'training_history.png')}")

    def evaluate_model(self, X_test, y_test, model_name=None):
        """评估指定模型或所有模型的性能

        Args:
            X_test: 测试特征
            y_test: 测试标签
            model_name: 模型名称，如果为None则评估所有模型

        Returns:
            dict: 评估结果
        """
        # 特征标准化
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        if model_name and model_name in self.models:
            models_to_eval = {model_name: self.models[model_name]}
        else:
            models_to_eval = self.models

        for name, model in models_to_eval.items():
            print(f"\n评估 {name} 模型...")

            # 深度学习模型和传统机器学习模型的处理不同
            if name in ['cnn', 'lstm']:
                # 加载类别索引
                model_dir = self.config.get_path("model_dir")
                with open(os.path.join(model_dir, 'class_indices.json'), 'r') as f:
                    class_to_idx = json.load(f)

                classes = list(class_to_idx.keys())
                idx_to_class = {v: k for k, v in class_to_idx.items()}

                # 转换测试标签
                y_test_idx = np.array([class_to_idx[str(cls)] for cls in y_test])
                y_test_cat = to_categorical(y_test_idx)

                # 准备测试数据
                if name == 'cnn' or name == 'lstm':
                    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

                    # 预测
                    y_probs = model.predict(X_test_reshaped)
                    y_pred = np.argmax(y_probs, axis=1)

                    # 转换回原始标签
                    y_pred_classes = np.array([int(idx_to_class[idx]) for idx in y_pred])

                    # 计算准确率
                    accuracy = np.mean(y_pred_classes == y_test)

                    # 计算混淆矩阵
                    cm = confusion_matrix(y_test, y_pred_classes)

                    # 生成分类报告
                    report = classification_report(y_test, y_pred_classes, output_dict=True)

                    results[name] = {
                        'accuracy': accuracy,
                        'confusion_matrix': cm,
                        'classification_report': report
                    }

                    print(f"{name} 准确率: {accuracy:.4f}")
                    print("\n分类报告:")
                    print(classification_report(y_test, y_pred_classes))
            else:
                # 传统机器学习模型
                y_pred = model.predict(X_test_scaled)
                accuracy = np.mean(y_pred == y_test)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                results[name] = {
                    'accuracy': accuracy,
                    'confusion_matrix': cm,
                    'classification_report': report
                }

                print(f"{name} 准确率: {accuracy:.4f}")
                print("\n分类报告:")
                print(classification_report(y_test, y_pred))

            # 可视化混淆矩阵
            self.plot_confusion_matrix(cm, name)

        return results

    def plot_confusion_matrix(self, cm, model_name):
        """可视化混淆矩阵

        Args:
            cm: 混淆矩阵
            model_name: 模型名称
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()

        # 添加数值标签
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        results_dir = self.config.get_path("results_dir")
        plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model_name}.png'))
        plt.close()
        print(f"混淆矩阵已保存至 {os.path.join(results_dir, f'confusion_matrix_{model_name}.png')}")

    def load_models(self):
        """加载所有已训练的模型

        Returns:
            dict: 加载的模型字典
        """
        model_dir = self.config.get_path("model_dir")
        models = {}

        # 加载传统机器学习模型
        for model_name in ['random_forest', 'gradient_boosting']:
            model_path = os.path.join(model_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                    print(f"已加载 {model_name} 模型")

        # 加载CNN模型
        cnn_path = os.path.join(model_dir, 'cnn_model.h5')
        if os.path.exists(cnn_path):
            try:
                models['cnn'] = load_model(cnn_path)
                print(f"已加载 CNN 模型")
            except Exception as e:
                print(f"警告: 无法加载CNN模型: {e}")

        # 加载LSTM模型
        lstm_path = os.path.join(model_dir, 'lstm_model.h5')
        if os.path.exists(lstm_path):
            try:
                models['lstm'] = load_model(lstm_path)
                print(f"已加载 LSTM 模型")
            except Exception as e:
                print(f"警告: 无法加载LSTM模型: {e}")

        # 加载类别索引映射
        class_indices_path = os.path.join(model_dir, 'class_indices.json')
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                self.class_indices = json.load(f)
                # 创建逆映射
                self.idx_to_class = {int(v): k for k, v in self.class_indices.items()}
                print(f"已加载类别索引映射")

        # 加载缩放器
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                print(f"已加载特征缩放器")

        self.models = models
        return models

    def predict(self, features, ensemble=True):
        """使用加载的模型进行预测

        Args:
            features: 特征向量或矩阵
            ensemble: 是否使用集成方法

        Returns:
            预测的类别
        """
        if not self.models:
            print("错误: 没有加载任何模型，请先加载模型")
            return None

        # 确保特征是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 应用特征缩放
        features_scaled = self.scaler.transform(features)

        # 单个样本的预测结果
        predictions = {}

        # 对每个模型进行预测
        for name, model in self.models.items():
            if name in ['cnn', 'lstm']:
                # 深度学习模型
                features_reshaped = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
                pred_probs = model.predict(features_reshaped)
                pred_classes = np.argmax(pred_probs, axis=1)

                # 转换回原始标签
                if hasattr(self, 'idx_to_class'):
                    predictions[name] = [int(self.idx_to_class[idx]) for idx in pred_classes]
                else:
                    predictions[name] = pred_classes
            else:
                # 传统机器学习模型
                predictions[name] = model.predict(features_scaled)

        if not ensemble or len(self.models) == 1:
            # 使用单个模型（优先使用深度学习模型）
            if 'cnn' in predictions:
                return predictions['cnn']
            elif 'lstm' in predictions:
                return predictions['lstm']
            else:
                # 使用第一个可用的模型
                return next(iter(predictions.values()))
        else:
            # 集成预测
            # 获取所有预测结果
            all_preds = []
            for model_name, preds in predictions.items():
                all_preds.extend(preds)

            # 对于每个样本，进行投票
            ensemble_predictions = []
            for i in range(features.shape[0]):
                sample_preds = [predictions[model_name][i] for model_name in predictions]
                # 投票（取众数）
                counter = Counter(sample_preds)
                ensemble_pred = counter.most_common(1)[0][0]
                ensemble_predictions.append(ensemble_pred)

            return np.array(ensemble_predictions)


class SequenceModeling:
    """序列建模组件，使用N-gram模型和HMM增强预测结果"""

    def __init__(self, config_manager):
        """初始化序列建模组件

        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager
        self.ngram_model = None
        self.hmm_model = None

    def train_ngram_model(self, sequences):
        """训练N-gram语言模型

        Args:
            sequences: 训练序列列表

        Returns:
            训练好的N-gram模型
        """
        from collections import defaultdict

        order = self.config.get("sequence_model.ngram_order", 3)
        model = defaultdict(Counter)

        # 构建N-gram频率
        for sequence in sequences:
            # 添加开始和结束标记
            padded_seq = [' '] * (order - 1) + list(str(sequence)) + [' ']

            for i in range(len(padded_seq) - order + 1):
                context = tuple(padded_seq[i:i + order - 1])
                next_char = padded_seq[i + order - 1]
                model[context][next_char] += 1

        # 将频率转换为概率
        ngram_model = {}
        for context, counter in model.items():
            total = sum(counter.values())
            ngram_model[context] = {char: count / total for char, count in counter.items()}

        self.ngram_model = ngram_model

        # 保存模型
        model_dir = self.config.get_path("model_dir")
        with open(os.path.join(model_dir, 'ngram_model.pkl'), 'wb') as f:
            pickle.dump(ngram_model, f)

        return ngram_model

    def score_sequence(self, sequence, smoothing=0.1):
        """使用N-gram模型计算序列概率

        Args:
            sequence: 要评分的序列
            smoothing: 平滑参数

        Returns:
            序列的对数概率
        """
        if not self.ngram_model:
            return 0

        order = self.config.get("sequence_model.ngram_order", 3)
        log_prob = 0

        # 添加开始和结束标记
        padded_seq = [' '] * (order - 1) + list(str(sequence)) + [' ']

        for i in range(len(padded_seq) - order + 1):
            context = tuple(padded_seq[i:i + order - 1])
            next_char = padded_seq[i + order - 1]

            # 获取条件概率，使用平滑处理
            if context in self.ngram_model and next_char in self.ngram_model[context]:
                prob = self.ngram_model[context][next_char]
            else:
                # 拉普拉斯平滑
                prob = smoothing / (smoothing * len(self.get_vocab()))

            log_prob += np.log(prob)

        return log_prob

    def get_vocab(self):
        """获取N-gram模型的词汇表

        Returns:
            词汇集合
        """
        if not self.ngram_model:
            return set()

        vocab = set()
        for context_dict in self.ngram_model.values():
            vocab.update(context_dict.keys())

        return vocab

    def correct_sequence(self, predicted_sequence, top_k=3):
        """使用N-gram模型纠正预测序列

        Args:
            predicted_sequence: 初始预测序列
            top_k: 返回的最佳变体数量

        Returns:
            纠正后的序列
        """
        if not self.ngram_model or not predicted_sequence:
            return predicted_sequence

        # 生成可能的序列变体（允许一定数量的替换）
        variants = self.generate_variants(predicted_sequence, top_k)

        # 计算每个变体的得分
        scored_variants = [(variant, self.score_sequence(variant)) for variant in variants]

        # 选择得分最高的
        best_variant = max(scored_variants, key=lambda x: x[1])

        return best_variant[0]

    def generate_variants(self, sequence, top_k=3):
        """生成序列的可能变体

        Args:
            sequence: 原始序列
            top_k: 返回的最佳变体数量

        Returns:
            变体列表
        """
        if not self.ngram_model:
            return [sequence]

        # 获取词汇表
        vocab = self.get_vocab()
        vocab = [v for v in vocab if v not in [' ', '']]

        variants = [sequence]

        # 为每个位置生成替换字符
        for i in range(len(sequence)):
            prefix = sequence[:i]
            suffix = sequence[i + 1:]

            for char in vocab:
                if char != sequence[i]:
                    variant = prefix + char + suffix
                    variants.append(variant)

        # 计算每个变体的得分
        scored_variants = [(variant, self.score_sequence(variant)) for variant in variants]
        scored_variants.sort(key=lambda x: x[1], reverse=True)

        # 返回得分最高的top_k个变体
        return [v for v, _ in scored_variants[:top_k]]

    def load_model(self):
        """加载保存的N-gram模型

        Returns:
            bool: 是否成功加载
        """
        model_dir = self.config.get_path("model_dir")
        ngram_path = os.path.join(model_dir, 'ngram_model.pkl')

        if os.path.exists(ngram_path):
            try:
                with open(ngram_path, 'rb') as f:
                    self.ngram_model = pickle.load(f)
                print(f"已加载N-gram模型")
                return True
            except Exception as e:
                print(f"加载N-gram模型失败: {e}")
                return False
        else:
            print(f"N-gram模型文件不存在: {ngram_path}")
            return False