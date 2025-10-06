# import os
# import torch
# import numpy as np
# import librosa
# import soundfile as sf
# from pathlib import Path
# import matplotlib.pyplot as plt
# import json
# import time
# import pickle
# from collections import Counter, defaultdict
# from scipy.special import softmax
# import itertools
# import traceback

# # 导入必要的模块
# from config_manager import ConfigManager
# from audio_processing import AudioProcessor
# from feature_extraction import FeatureExtractor
# from keystroke_model import KeystrokeModelTrainer

# # 全局键盘映射
# KEYS = '1234567890qwertyuiopasdfghjklzxcvbnm'


# # Seq2Seq模型配置
# class Config:
#     """Seq2Seq模型配置类"""
#     SOS_TOKEN = "< SOS >"
#     EOS_TOKEN = "<EOS>"
#     PAD_TOKEN = "<PAD>"
#     MASK_TOKEN = "￥"
#     vocab = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN] + [chr(i) for i in range(32, 127)] + [MASK_TOKEN]
#     char2idx = {c: i for i, c in enumerate(vocab)}
#     idx2char = {i: c for i, c in enumerate(vocab)}
#     VOCAB_SIZE = len(vocab)
#     EMBED_DIM = 256
#     HIDDEN_DIM = 512
#     NUM_LAYERS = 2
#     DROPOUT = 0.5
#     BATCH_SIZE = 64
#     TEACHER_FORCING_RATIO = 0.5
#     BEAM_WIDTH = 90
#     MAX_LEN = 30
#     PATIENCE = 5
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Seq2Seq模型组件定义
# class Attention(torch.nn.Module):
#     """注意力机制模块"""

#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.attn = torch.nn.Linear(hidden_dim * 2, hidden_dim)
#         self.v = torch.nn.Linear(hidden_dim, 1)

#     def forward(self, hidden, encoder_outputs):
#         seq_len = encoder_outputs.shape[0]
#         hidden = hidden.repeat(seq_len, 1, 1)
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
#         attention = self.v(energy).squeeze(2)
#         return torch.softmax(attention, dim=0)


# class Encoder(torch.nn.Module):
#     """Seq2Seq编码器"""

#     def __init__(self):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)
#         self.lstm = torch.nn.LSTM(Config.EMBED_DIM, Config.HIDDEN_DIM,
#                                   Config.NUM_LAYERS, dropout=Config.DROPOUT)

#     def forward(self, src, src_len):
#         embedded = self.embedding(src)
#         packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=False)
#         outputs, (hidden, cell) = self.lstm(packed)
#         outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
#         return outputs, hidden, cell


# class Decoder(torch.nn.Module):
#     """Seq2Seq解码器"""

#     def __init__(self):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)
#         self.attention = Attention(Config.HIDDEN_DIM)
#         self.lstm = torch.nn.LSTM(Config.EMBED_DIM + Config.HIDDEN_DIM, Config.HIDDEN_DIM,
#                                   Config.NUM_LAYERS, dropout=Config.DROPOUT)
#         self.fc = torch.nn.Linear(Config.HIDDEN_DIM * 2, Config.VOCAB_SIZE)

#     def forward(self, input, hidden, cell, encoder_outputs):
#         input = input.unsqueeze(0)
#         embedded = self.embedding(input)
#         attn_weights = self.attention(hidden[-1], encoder_outputs)
#         context = (attn_weights.unsqueeze(2) * encoder_outputs).sum(dim=0)
#         lstm_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)
#         output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
#         prediction = self.fc(torch.cat((output.squeeze(0), context), dim=1))
#         return prediction, hidden, cell


# class Seq2Seq(torch.nn.Module):
#     """Seq2Seq模型"""

#     def __init__(self):
#         super().__init__()
#         self.encoder = Encoder().to(Config.DEVICE)
#         self.decoder = Decoder().to(Config.DEVICE)

#     def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
#         batch_size = src.shape[1]
#         trg_len = trg.shape[0]

#         encoder_outputs, hidden, cell = self.encoder(src, src_len)

#         inputs = trg[0]
#         outputs = torch.zeros(trg_len, batch_size, Config.VOCAB_SIZE).to(Config.DEVICE)

#         for t in range(1, trg_len):
#             output, hidden, cell = self.decoder(inputs, hidden, cell, encoder_outputs)
#             outputs[t] = output
#             teacher_force = torch.rand(1).item() < teacher_forcing_ratio
#             top1 = output.argmax(1)
#             inputs = trg[t] if teacher_force else top1

#         return outputs

#     def beam_decode(self, src, src_len, max_len=Config.MAX_LEN, beam_width=Config.BEAM_WIDTH, alpha=0.6):
#         """优化的波束搜索解码，带长度惩罚"""
#         encoder_outputs, hidden, cell = self.encoder(src, src_len)

#         # 初始序列 [序列, 累计得分, 得分列表, 最后隐藏状态, 最后细胞状态]
#         sequences = [
#             [[Config.char2idx[Config.SOS_TOKEN]], 0.0, [], hidden, cell]
#         ]
#         # 已完成的序列
#         completed_sequences = []

#         # 开始波束搜索
#         for _ in range(max_len):
#             candidates = []

#             # 扩展当前序列
#             for seq, score, scores_list, h, c in sequences:
#                 # 如果序列已结束
#                 if seq[-1] == Config.char2idx[Config.EOS_TOKEN]:
#                     completed_sequences.append([seq, score, scores_list, h, c])
#                     continue

#                 # 通过解码器获取下一个预测
#                 input_token = torch.tensor([seq[-1]], device=Config.DEVICE)
#                 output, new_h, new_c = self.decoder(input_token, h, c, encoder_outputs)

#                 # 获取Top-k预测
#                 log_probs = torch.log_softmax(output, dim=1)
#                 topk_probs, topk_ids = log_probs.topk(beam_width)

#                 # 将新的候选序列添加到候选列表
#                 for i in range(topk_ids.size(1)):
#                     token_id = topk_ids[0][i].item()
#                     prob = topk_probs[0][i].item()

#                     new_seq = seq + [token_id]
#                     new_score = score + prob
#                     new_scores_list = scores_list + [prob]

#                     # 添加长度惩罚
#                     lp = ((5 + len(new_seq)) ** alpha) / ((5 + 1) ** alpha)
#                     normalized_score = new_score / lp

#                     candidates.append([new_seq, new_score, new_scores_list, new_h, new_c, normalized_score])

#             # 如果没有候选序列，结束搜索
#             if not candidates:
#                 break

#             # 按归一化得分排序，只保留前beam_width个
#             sequences = sorted(candidates, key=lambda x: x[5], reverse=True)[:beam_width]
#             sequences = [[seq, score, scores_list, h, c] for seq, score, scores_list, h, c, _ in sequences]

#         # 合并已完成序列和未完成序列
#         all_sequences = completed_sequences + sequences

#         # 按得分排序
#         all_sequences = sorted(all_sequences, key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)

#         # 返回得分最高的序列
#         return all_sequences[0] if all_sequences else None

#     def beam_search_multiple(self, src, src_len, max_len=Config.MAX_LEN, beam_width=Config.BEAM_WIDTH, num_results=10):
#         """返回多个波束搜索结果"""
#         encoder_outputs, hidden, cell = self.encoder(src, src_len)

#         # 初始序列 [序列, 累计得分, 得分列表, 最后隐藏状态, 最后细胞状态]
#         sequences = [
#             [[Config.char2idx[Config.SOS_TOKEN]], 0.0, [], hidden, cell]
#         ]
#         # 已完成的序列
#         completed_sequences = []

#         # 开始波束搜索
#         for _ in range(max_len):
#             candidates = []

#             # 扩展当前序列
#             for seq, score, scores_list, h, c in sequences:
#                 # 如果序列已结束
#                 if seq[-1] == Config.char2idx[Config.EOS_TOKEN]:
#                     completed_sequences.append([seq, score, scores_list, h, c])
#                     continue

#                 # 通过解码器获取下一个预测
#                 input_token = torch.tensor([seq[-1]], device=Config.DEVICE)
#                 output, new_h, new_c = self.decoder(input_token, h, c, encoder_outputs)

#                 # 获取Top-k预测
#                 log_probs = torch.log_softmax(output, dim=1)
#                 topk_probs, topk_ids = log_probs.topk(beam_width)

#                 # 将新的候选序列添加到候选列表
#                 for i in range(topk_ids.size(1)):
#                     token_id = topk_ids[0][i].item()
#                     prob = topk_probs[0][i].item()

#                     new_seq = seq + [token_id]
#                     new_score = score + prob
#                     new_scores_list = scores_list + [prob]

#                     candidates.append([new_seq, new_score, new_scores_list, new_h, new_c, new_score / len(new_seq)])

#             # 如果没有候选序列，结束搜索
#             if not candidates:
#                 break

#             # 按得分排序，只保留前beam_width个
#             sequences = sorted(candidates, key=lambda x: x[5], reverse=True)[:beam_width]
#             sequences = [[seq, score, scores_list, h, c] for seq, score, scores_list, h, c, _ in sequences]

#         # 合并已完成序列和未完成序列
#         all_sequences = completed_sequences + sequences

#         # 按归一化得分排序
#         all_sequences = sorted(all_sequences, key=lambda x: x[1] / len(x[0]), reverse=True)

#         # 转换序列为文本并返回多个结果
#         unique_results = []
#         seen_texts = set()

#         for seq, score, scores_list, _, _ in all_sequences:
#             # 提取文本
#             output_text = ''.join([Config.idx2char[i] for i in seq
#                                    if i not in {Config.char2idx[Config.SOS_TOKEN],
#                                                 Config.char2idx[Config.EOS_TOKEN],
#                                                 Config.char2idx[Config.PAD_TOKEN]}])

#             # 添加到结果中，保证唯一性
#             if output_text and output_text not in seen_texts:
#                 seen_texts.add(output_text)
#                 unique_results.append([seq, score / len(seq), scores_list, output_text])
#                 if len(unique_results) >= num_results:
#                     break

#         return unique_results


# class ProbabilityAnalyzer:
#     """概率分析工具类"""

#     @staticmethod
#     def probability_entropy(probs):
#         """计算概率分布的信息熵"""
#         return -np.sum(probs * np.log(probs + 1e-9))

#     @staticmethod
#     def top_k_certainty(probs, k=3):
#         """计算前K个预测的总概率"""
#         sorted_indices = np.argsort(probs)[-k:]
#         return np.sum(probs[sorted_indices])

#     @staticmethod
#     def probability_contrast(probs):
#         """计算概率对比度（最高概率与第二高概率的差距）"""
#         sorted_probs = np.sort(probs)
#         if len(sorted_probs) >= 2:
#             return sorted_probs[-1] - sorted_probs[-2]
#         return 0

#     @staticmethod
#     def normalize_probabilities(probs, temperature=1.0):
#         """使用温度重新归一化概率分布"""
#         if temperature == 1.0:
#             return probs

#         # 使用温度缩放概率，降低温度会使分布更加尖锐
#         log_probs = np.log(probs + 1e-9) / temperature
#         return softmax(log_probs)

#     @staticmethod
#     def analyze_position(probs, idx_to_class):
#         """分析按键位置的概率特征"""
#         entropy = ProbabilityAnalyzer.probability_entropy(probs)
#         top_indices = np.argsort(probs)[-3:][::-1]
#         top_probs = probs[top_indices]

#         # 将索引转换为字符
#         top_chars = []
#         for idx in top_indices:
#             if str(idx) in idx_to_class:
#                 char = idx_to_class[str(idx)]
#             else:
#                 char = str(idx)
#             top_chars.append(char)

#         # 计算对比度 (最高概率与第二高概率的差距)
#         contrast = ProbabilityAnalyzer.probability_contrast(probs)

#         # 计算前3个预测的总概率
#         top3_certainty = ProbabilityAnalyzer.top_k_certainty(probs, 3)

#         return {
#             'top_chars': top_chars,
#             'top_probs': top_probs.tolist(),
#             'entropy': float(entropy),
#             'contrast': float(contrast),
#             'top3_certainty': float(top3_certainty)
#         }


# class MaskGenerator:
#     """增强的掩码生成器"""

#     def __init__(self, mask_token=Config.MASK_TOKEN, max_mask_ratio=0.6):
#         """初始化掩码生成器

#         Args:
#             mask_token: 掩码标记
#             max_mask_ratio: 最大掩码比例，超过此比例的掩码将被过滤
#         """
#         self.mask_token = mask_token
#         self.max_mask_ratio = max_mask_ratio
#         self.templates = {}

#     def _check_mask_ratio(self, mask):
#         """检查掩码中掩码标记的比例是否在允许范围内

#         Args:
#             mask: 掩码字符串或列表

#         Returns:
#             bool: 掩码是否通过检查
#         """
#         if isinstance(mask, list):
#             mask_count = mask.count(self.mask_token)
#             total = len(mask)
#         else:
#             mask_count = mask.count(self.mask_token)
#             total = len(mask)

#         mask_ratio = mask_count / total
#         return mask_ratio <= self.max_mask_ratio

#     def generate_masks(self, position_info, base_prediction=None, sequential_analysis=True):
#         """生成多种掩码模板，不限制数量"""
#         if not position_info:
#             return []

#         masks = []
#         self.templates = {}

#         # 1. 基于置信度的基础掩码
#         for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
#             mask = self._create_confidence_mask(position_info, threshold)
#             mask_str = ''.join(mask)
#             self.templates[f"基础掩码 (阈值={threshold:.1f})"] = mask_str
#             masks.append(mask_str)

#         # 2. 基于熵的掩码
#         for threshold in [0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]:
#             mask = self._create_entropy_mask(position_info, threshold)
#             mask_str = ''.join(mask)
#             self.templates[f"熵掩码 (阈值={threshold:.1f})"] = mask_str
#             masks.append(mask_str)

#         # 3. 基于对比度的掩码
#         for threshold in [0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
#             mask = self._create_contrast_mask(position_info, threshold)
#             mask_str = ''.join(mask)
#             self.templates[f"对比度掩码 (阈值={threshold:.1f})"] = mask_str
#             masks.append(mask_str)

#         # 4. 保留前N个确定位置的掩码
#         for keep_count in range(1, min(len(position_info), 8)):
#             mask = self._create_topn_mask(position_info, keep_count)
#             mask_str = ''.join(mask)
#             self.templates[f"Top-{keep_count}掩码"] = mask_str
#             masks.append(mask_str)

#         # 5. 混合策略掩码
#         for threshold in [0.7, 0.6, 0.5, 0.4]:
#             hybrid_mask = self._create_hybrid_mask(position_info, threshold)
#             mask_str = ''.join(hybrid_mask)
#             self.templates[f"混合策略掩码 (阈值={threshold:.1f})"] = mask_str
#             masks.append(mask_str)

#         # 6. 序列模式掩码
#         if sequential_analysis:
#             pattern_mask = self._analyze_sequence_patterns(position_info)
#             if pattern_mask:
#                 mask_str = ''.join(pattern_mask)
#                 self.templates["序列模式掩码"] = mask_str
#                 masks.append(mask_str)

#         # 7. 概率组合掩码
#         combination_masks = self._create_combination_masks(position_info, max_uncertain=4)
#         for i, mask_str in enumerate(combination_masks):
#             self.templates[f"组合掩码-{i + 1}"] = mask_str
#             masks.append(mask_str)

#         # 8. 完全穷举掩码 (不依赖概率分布)
#         enumeration_masks = self._create_pure_enumeration_masks(position_info, base_prediction)
#         for i, mask_str in enumerate(enumeration_masks):
#             self.templates[f"纯穷举掩码-{i + 1}"] = mask_str
#             masks.append(mask_str)

#         # 9. 基于熵的多候选掩码
#         multi_candidate_masks = self._create_entropy_based_candidates(position_info)
#         for i, mask_str in enumerate(multi_candidate_masks):
#             self.templates[f"熵候选掩码-{i + 1}"] = mask_str
#             masks.append(mask_str)

#         # 过滤掩码中全是掩码标记的情况
#         filtered_masks = []
#         for mask in masks:
#             if mask.count(self.mask_token) < len(mask):  # 掩码不能全是掩码标记
#                 filtered_masks.append(mask)

#         # 删除重复的掩码
#         unique_masks = []
#         seen = set()
#         for mask in filtered_masks:
#             if mask not in seen:
#                 seen.add(mask)
#                 unique_masks.append(mask)

#         # 为每个掩码计算质量得分
#         scored_masks = []
#         for mask in unique_masks:
#             score = self._calculate_mask_quality(mask, position_info)
#             scored_masks.append((mask, score))

#         # 按掩码数量和得分排序
#         scored_masks.sort(key=lambda x: (x[0].count(self.mask_token), -x[1]))

#         # 返回所有掩码 - 不限制数量
#         return scored_masks

#     def _create_confidence_mask(self, position_info, threshold):
#         """基于置信度创建掩码，同时确保掩码率不超过阈值"""
#         mask = []
#         for pos in position_info:
#             if pos['top_probs'][0] >= threshold:
#                 mask.append(pos['top_chars'][0])
#             else:
#                 mask.append(self.mask_token)

#         # 检查掩码率
#         mask_count = mask.count(self.mask_token)
#         max_allowed_masks = int(len(mask) * self.max_mask_ratio)

#         # 如果掩码太多，填充一些低置信度但相对较高的位置
#         if mask_count > max_allowed_masks:
#             # 获取所有掩码位置
#             masked_indices = [i for i, c in enumerate(mask) if c == self.mask_token]
#             # 按照置信度排序这些位置
#             sorted_indices = sorted(masked_indices,
#                                     key=lambda i: position_info[i]['top_probs'][0],
#                                     reverse=True)
#             # 填充掩码过多的部分
#             for i in sorted_indices[:mask_count - max_allowed_masks]:
#                 mask[i] = position_info[i]['top_chars'][0]

#         return mask

#     def _create_entropy_mask(self, position_info, threshold):
#         """基于熵创建掩码，同时确保掩码率不超过阈值"""
#         mask = []
#         for pos in position_info:
#             if pos['entropy'] <= threshold:
#                 mask.append(pos['top_chars'][0])
#             else:
#                 mask.append(self.mask_token)

#         # 检查掩码率
#         mask_count = mask.count(self.mask_token)
#         max_allowed_masks = int(len(mask) * self.max_mask_ratio)

#         # 如果掩码太多，填充一些熵较低的位置
#         if mask_count > max_allowed_masks:
#             # 获取所有掩码位置
#             masked_indices = [i for i, c in enumerate(mask) if c == self.mask_token]
#             # 按照熵值排序这些位置
#             sorted_indices = sorted(masked_indices,
#                                     key=lambda i: position_info[i]['entropy'])
#             # 填充掩码过多的部分
#             for i in sorted_indices[:mask_count - max_allowed_masks]:
#                 mask[i] = position_info[i]['top_chars'][0]

#         return mask

#     def _create_contrast_mask(self, position_info, threshold):
#         """基于对比度创建掩码，同时确保掩码率不超过阈值"""
#         mask = []
#         for pos in position_info:
#             if pos['contrast'] >= threshold:
#                 mask.append(pos['top_chars'][0])
#             else:
#                 mask.append(self.mask_token)

#         # 检查掩码率
#         mask_count = mask.count(self.mask_token)
#         max_allowed_masks = int(len(mask) * self.max_mask_ratio)

#         # 如果掩码太多，填充一些对比度较高的位置
#         if mask_count > max_allowed_masks:
#             # 获取所有掩码位置
#             masked_indices = [i for i, c in enumerate(mask) if c == self.mask_token]
#             # 按照对比度排序这些位置
#             sorted_indices = sorted(masked_indices,
#                                     key=lambda i: position_info[i]['contrast'],
#                                     reverse=True)
#             # 填充掩码过多的部分
#             for i in sorted_indices[:mask_count - max_allowed_masks]:
#                 mask[i] = position_info[i]['top_chars'][0]

#         return mask

#     def _create_topn_mask(self, position_info, n):
#         """创建仅保留前N个高置信度预测的掩码"""
#         # 按置信度排序位置
#         sorted_pos = sorted(enumerate(position_info),
#                             key=lambda x: x[1]['top_probs'][0],
#                             reverse=True)

#         # 创建全掩码模板
#         mask = [self.mask_token] * len(position_info)

#         # 计算需要填充的最小位置数，确保掩码率不超过阈值
#         min_fill = int(len(position_info) * (1 - self.max_mask_ratio))
#         fill_count = max(n, min_fill)  # 取n和min_fill中的较大值

#         # 填充前N个位置，但确保至少填充min_fill个位置
#         for i in range(min(fill_count, len(sorted_pos))):
#             pos_idx, pos_info = sorted_pos[i]
#             mask[pos_idx] = pos_info['top_chars'][0]

#         return mask

#     def _create_hybrid_mask(self, position_info, threshold=0.6):
#         """创建混合策略掩码，结合多种特征并确保掩码率不超过阈值"""
#         mask = []
#         for pos in position_info:
#             # 结合多种特征进行决策
#             score = 0.4 * pos['top_probs'][0] + 0.2 * (1 - pos['entropy'] / 3) + 0.2 * pos['contrast'] + 0.2 * pos[
#                 'top3_certainty']

#             if score >= threshold:  # 动态阈值
#                 mask.append(pos['top_chars'][0])
#             else:
#                 mask.append(self.mask_token)

#         # 检查掩码率
#         mask_count = mask.count(self.mask_token)
#         max_allowed_masks = int(len(mask) * self.max_mask_ratio)

#         # 如果掩码太多，填充一些综合分数较高的位置
#         if mask_count > max_allowed_masks:
#             # 获取所有掩码位置
#             masked_indices = [i for i, c in enumerate(mask) if c == self.mask_token]

#             # 计算每个位置的综合分数
#             scores = []
#             for i in masked_indices:
#                 pos = position_info[i]
#                 score = 0.4 * pos['top_probs'][0] + 0.2 * (1 - pos['entropy'] / 3) + 0.2 * pos['contrast'] + 0.2 * pos[
#                     'top3_certainty']
#                 scores.append((i, score))

#             # 按分数排序
#             sorted_indices = [i for i, _ in sorted(scores, key=lambda x: x[1], reverse=True)]

#             # 填充掩码过多的部分
#             for i in sorted_indices[:mask_count - max_allowed_masks]:
#                 mask[i] = position_info[i]['top_chars'][0]

#         return mask

#     def _analyze_sequence_patterns(self, position_info):
#         """分析序列模式，识别可能的数字序列"""
#         if len(position_info) < 4:  # 太短无法分析
#             return None

#         # 尝试检测数字序列模式 (如递增、递减、重复等)
#         top_chars = [pos['top_chars'][0] for pos in position_info]

#         # 检查是否全是数字
#         if all(c.isdigit() for c in top_chars):
#             # 检查递增/递减模式
#             is_increasing = True
#             is_decreasing = True

#             for i in range(1, len(top_chars)):
#                 if int(top_chars[i]) != int(top_chars[i - 1]) + 1:
#                     is_increasing = False
#                 if int(top_chars[i]) != int(top_chars[i - 1]) - 1:
#                     is_decreasing = False

#             # 如果检测到模式，创建更高置信度的掩码
#             if is_increasing or is_decreasing:
#                 # 基础掩码
#                 mask = self._create_confidence_mask(position_info, 0.6)

#                 # 修复可能的错误
#                 for i in range(1, len(mask) - 1):
#                     if mask[i] == self.mask_token:
#                         # 如果前后都不是掩码，且符合递增/递减模式，则填充
#                         if mask[i - 1] != self.mask_token and mask[i + 1] != self.mask_token:
#                             if is_increasing and int(mask[i + 1]) - int(mask[i - 1]) == 2:
#                                 mask[i] = str(int(mask[i - 1]) + 1)
#                             elif is_decreasing and int(mask[i - 1]) - int(mask[i + 1]) == 2:
#                                 mask[i] = str(int(mask[i - 1]) - 1)

#                 return mask

#         return None

#     def _create_combination_masks(self, position_info, max_uncertain=4):
#         """创建考虑第二高概率的组合掩码"""
#         # 找出不确定的位置 (第二高概率也较高)
#         uncertain_positions = []
#         for i, pos in enumerate(position_info):
#             if len(pos['top_probs']) > 1 and pos['top_probs'][1] > 0.3:
#                 uncertain_positions.append(i)

#         # 如果不确定位置太多，只保留概率差最小的几个
#         if len(uncertain_positions) > max_uncertain:
#             pos_with_diffs = [(i, position_info[i]['top_probs'][0] - position_info[i]['top_probs'][1])
#                               for i in uncertain_positions]
#             pos_with_diffs.sort(key=lambda x: x[1])  # 按概率差排序
#             uncertain_positions = [pos for pos, _ in pos_with_diffs[:max_uncertain]]

#         # 如果没有不确定位置，返回空列表
#         if not uncertain_positions:
#             return []

#         # 创建基础掩码
#         base_mask = self._create_confidence_mask(position_info, 0.5)

#         # 生成组合
#         combination_masks = []

#         # 对于每个不确定位置，尝试第一和第二高概率的选择
#         for combination in itertools.product([0, 1], repeat=len(uncertain_positions)):
#             candidate = base_mask.copy()
#             for i, pos_idx in enumerate(uncertain_positions):
#                 # 使用第一或第二高的预测
#                 candidate[pos_idx] = position_info[pos_idx]['top_chars'][min(combination[i],
#                                                                              len(position_info[pos_idx][
#                                                                                      'top_chars']) - 1)]

#             # 检查掩码率
#             mask_str = ''.join(candidate)
#             if self._check_mask_ratio(mask_str):
#                 combination_masks.append(mask_str)

#         return combination_masks

#     def _calculate_mask_quality(self, mask, position_info, base_prediction=None):
#         """计算掩码质量得分"""
#         if len(mask) != len(position_info):
#             return 0

#         confidence_sum = 0
#         mask_count = 0
#         base_match_score = 0

#         for i, char in enumerate(mask):
#             if char != self.mask_token:
#                 # 计算与基础预测的匹配度
#                 if base_prediction and i < len(base_prediction):
#                     if char == base_prediction[i]:
#                         base_match_score += 1

#                 # 计算概率得分
#                 try:
#                     char_idx = position_info[i]['top_chars'].index(char)
#                     prob = position_info[i]['top_probs'][char_idx]
#                     confidence_sum += prob
#                     mask_count += 1
#                 except ValueError:
#                     confidence_sum += 0.1
#                     mask_count += 1

#         # 计算平均置信度
#         avg_confidence = confidence_sum / max(1, mask_count)

#         # 计算基础预测匹配得分
#         if base_prediction and len(base_prediction) > 0:
#             base_match_ratio = base_match_score / len(base_prediction)
#         else:
#             base_match_ratio = 0

#         # 计算掩码比例得分
#         mask_ratio = mask_count / len(mask)
#         mask_ratio_score = 0.5 + 0.5 * mask_ratio  # 0.5-1.0的得分范围

#         # 加权平均 - 增加与基础预测匹配的权重
#         total_score = 0.4 * avg_confidence + 0.3 * mask_ratio_score + 0.3 * base_match_ratio

#         return total_score

#     def _create_pure_enumeration_masks(self, position_info, base_prediction=None):
#         """完全不依赖概率分布的穷举掩码策略

#         Args:
#             position_info: 位置概率信息
#             base_prediction: 基础声音模型的预测结果
#         """
#         masks = []
#         seq_length = len(position_info)

#         # 使用基础声音模型预测作为基准
#         if base_prediction and len(base_prediction) == seq_length:
#             default_chars = list(base_prediction)
#         else:
#             # 如果没有基础预测，使用概率最高字符
#             default_chars = [pos['top_chars'][0] for pos in position_info]

#         # 单掩码 - 每个位置分别掩码
#         for i in range(seq_length):
#             mask = default_chars.copy()
#             mask[i] = self.mask_token
#             masks.append(''.join(mask))

#         # 双掩码 - 每两个位置组合掩码
#         if seq_length >= 2:
#             for i in range(seq_length):
#                 for j in range(i + 1, seq_length):
#                     mask = default_chars.copy()
#                     mask[i] = self.mask_token
#                     mask[j] = self.mask_token
#                     masks.append(''.join(mask))

#         # 三掩码 - 每三个位置组合掩码
#         if seq_length >= 3:
#             for i in range(seq_length):
#                 for j in range(i + 1, seq_length):
#                     for k in range(j + 1, seq_length):
#                         mask = default_chars.copy()
#                         mask[i] = self.mask_token
#                         mask[j] = self.mask_token
#                         mask[k] = self.mask_token
#                         masks.append(''.join(mask))

#         return masks

#     def _create_entropy_based_candidates(self, position_info):
#         """基于熵值为每个位置保留不同数量的候选"""
#         base_masks = []

#         # 准备基础掩码 - 所有位置用最高概率字符填充
#         base_mask = [pos['top_chars'][0] for pos in position_info]

#         # 对每个位置基于熵决定候选数量
#         for i, pos in enumerate(position_info):
#             entropy = pos['entropy']

#             # 动态决定候选数量
#             if entropy > 2.5:
#                 candidates = min(4, len(pos['top_chars']))  # 非常高熵，保留4个
#             elif entropy > 2.0:
#                 candidates = min(3, len(pos['top_chars']))  # 高熵，保留3个
#             elif entropy > 1.5:
#                 candidates = min(2, len(pos['top_chars']))  # 中等熵，保留2个
#             else:
#                 continue  # 低熵位置不生成候选

#             # 为每个额外候选创建掩码
#             for j in range(1, candidates):
#                 if j < len(pos['top_chars']):
#                     new_mask = base_mask.copy()
#                     new_mask[i] = pos['top_chars'][j]
#                     base_masks.append(''.join(new_mask))

#         return base_masks

#     def _create_context_aware_mask(self, position_info):
#         """生成考虑上下文关系的掩码模板"""
#         mask = []

#         for i, pos in enumerate(position_info):
#             # 基本置信度
#             base_confidence = pos['top_probs'][0]
#             context_boost = 0
#             alt_char_selected = False

#             # 分析相邻位置模式
#             if i > 0 and i < len(position_info) - 1:
#                 prev_char = position_info[i - 1]['top_chars'][0]
#                 curr_char = pos['top_chars'][0]
#                 next_char = position_info[i + 1]['top_chars'][0]

#                 # 数字序列检测
#                 if all(c.isdigit() for c in [prev_char, curr_char, next_char]):
#                     # 检查递增/递减模式
#                     if int(curr_char) == int(prev_char) + 1 and int(next_char) == int(curr_char) + 1:
#                         context_boost += 0.2  # 递增序列
#                     elif int(curr_char) == int(prev_char) - 1 and int(next_char) == int(curr_char) - 1:
#                         context_boost += 0.2  # 递减序列

#                     # 检查替代字符是否能形成更好的序列
#                     for alt_idx, alt_char in enumerate(pos['top_chars'][1:], 1):
#                         if not alt_char.isdigit():
#                             continue

#                         alt_prob = pos['top_probs'][alt_idx]
#                         prob_diff = pos['top_probs'][0] - alt_prob

#                         # 如果第二高概率不低于首选太多，且能形成更好序列
#                         if prob_diff < 0.2 and alt_char.isdigit():
#                             if int(alt_char) == int(prev_char) + 1 and int(next_char) == int(alt_char) + 1:
#                                 mask.append(alt_char)
#                                 alt_char_selected = True
#                                 break

#             if not alt_char_selected:
#                 # 结合上下文和基础置信度
#                 adjusted_confidence = base_confidence + context_boost

#                 if adjusted_confidence >= 0.55:
#                     mask.append(pos['top_chars'][0])
#                 else:
#                     mask.append(self.mask_token)

#         return mask


# class EnhancedPredictionSystem:
#     """增强的预测系统"""

#     def __init__(self, config_manager, seq2seq_model_path="seq_best_model.pth"):
#         """初始化预测系统

#         Args:
#             config_manager: 配置管理器实例
#             seq2seq_model_path: Seq2Seq模型文件路径
#         """
#         self.config = config_manager
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # 初始化组件
#         self.audio_processor = AudioProcessor(config_manager)
#         self.feature_extractor = FeatureExtractor(config_manager)
#         self.model_trainer = KeystrokeModelTrainer(config_manager)

#         # 初始化辅助工具
#         self.probability_analyzer = ProbabilityAnalyzer()
#         self.mask_generator = MaskGenerator(mask_token=Config.MASK_TOKEN, max_mask_ratio=0.6)

#         # 加载声音和序列模型
#         self.load_sound_models()
#         self.seq2seq_model = self.load_seq2seq_model(seq2seq_model_path)

#         # 加载对比用的普通预测系统
#         from keystroke_recognition import KeystrokeRecognitionSystem
#         self.basic_system = KeystrokeRecognitionSystem(config_path=config_manager.config_path)

#         # 统计信息
#         self.stats = {
#             'processed_files': 0,
#             'successful_predictions': 0,
#             'total_accuracy': 0,
#             'correct_chars': 0,
#             'total_chars': 0,
#             'basic_correct_chars': 0,
#             'advanced_correct_chars': 0,
#             'total_improvement': 0,
#             'improved_files': 0,
#             'start_time': time.time()
#         }

#     def load_sound_models(self):
#         """加载声音识别模型"""
#         model_dir = self.config.get_path("model_dir")
#         self.sound_models = {}

#         print(f"加载声音识别模型...")

#         # 检查模型目录
#         if not os.path.exists(model_dir):
#             print(f"警告: 模型目录 {model_dir} 不存在")
#             os.makedirs(model_dir, exist_ok=True)
#             return

#         # 加载CNN模型和LSTM模型
#         try:
#             # 尝试使用keras加载
#             try:
#                 import keras
#                 cnn_path = os.path.join(model_dir, 'cnn_model.h5')
#                 if os.path.exists(cnn_path):
#                     self.sound_models['cnn'] = keras.models.load_model(cnn_path)
#                     print(f"已加载 CNN 模型")

#                 lstm_path = os.path.join(model_dir, 'lstm_model.h5')
#                 if os.path.exists(lstm_path):
#                     self.sound_models['lstm'] = keras.models.load_model(lstm_path)
#                     print(f"已加载 LSTM 模型")
#             except Exception as e1:
#                 print(f"使用keras加载模型失败: {e1}")

#                 # 尝试使用tensorflow加载
#                 try:
#                     import tensorflow as tf
#                     cnn_path = os.path.join(model_dir, 'cnn_model.h5')
#                     if os.path.exists(cnn_path):
#                         self.sound_models['cnn'] = tf.keras.models.load_model(cnn_path)
#                         print(f"已加载 CNN 模型")

#                     lstm_path = os.path.join(model_dir, 'lstm_model.h5')
#                     if os.path.exists(lstm_path):
#                         self.sound_models['lstm'] = tf.keras.models.load_model(lstm_path)
#                         print(f"已加载 LSTM 模型")
#                 except Exception as e2:
#                     print(f"使用tensorflow加载模型失败: {e2}")
#         except Exception as e:
#             print(f"加载深度学习模型时出错: {e}")

#         # 加载类别索引映射
#         class_indices_path = os.path.join(model_dir, 'class_indices.json')
#         if os.path.exists(class_indices_path):
#             try:
#                 with open(class_indices_path, 'r') as f:
#                     self.class_indices = json.load(f)
#                     # 创建逆映射
#                     self.idx_to_class = {v: k for k, v in self.class_indices.items()}
#                 print(f"已加载类别索引映射")
#             except Exception as e:
#                 print(f"加载类别索引映射时出错: {e}")

#         # 加载缩放器
#         scaler_path = os.path.join(model_dir, 'scaler.pkl')
#         if os.path.exists(scaler_path):
#             try:
#                 with open(scaler_path, 'rb') as f:
#                     self.scaler = pickle.load(f)
#                     self.model_trainer.scaler = self.scaler
#                 print(f"已加载特征缩放器")
#             except Exception as e:
#                 print(f"加载特征缩放器时出错: {e}")

#         print(f"成功加载了 {len(self.sound_models)} 个声音模型")

#     def load_seq2seq_model(self, model_path):
#         """加载Seq2Seq模型

#         Args:
#             model_path: 模型文件路径

#         Returns:
#             加载的Seq2Seq模型
#         """
#         print(f"加载Seq2Seq模型: {model_path}")

#         if not os.path.exists(model_path):
#             print(f"警告: Seq2Seq模型文件不存在: {model_path}")
#             print(f"将创建新的Seq2Seq模型")
#             model = Seq2Seq()
#             return model

#         try:
#             # 创建模型实例
#             model = Seq2Seq()

#             # 加载模型权重
#             checkpoint = torch.load(model_path, map_location=self.device)
#             model.load_state_dict(checkpoint)

#             print(f"成功加载Seq2Seq模型")
#             return model
#         except Exception as e:
#             print(f"加载Seq2Seq模型失败: {e}")
#             print(f"将创建新的Seq2Seq模型")
#             return Seq2Seq()

#     def extract_keystroke_probabilities(self, audio_file_path):
#         """从音频文件中提取按键概率分布，优化版本

#         Args:
#             audio_file_path: 音频文件路径

#         Returns:
#             tuple: (位置信息列表, 检测到的按键数量)
#         """
#         try:
#             # 加载音频
#             y, sr = self.audio_processor.load_audio(audio_file_path)

#             # 尝试从文件名猜测预期的按键数量
#             expected_length = None
#             filename = os.path.basename(audio_file_path)
#             digit_part = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
#             if digit_part:
#                 expected_length = len(digit_part)
#                 print(f"从文件名猜测的预期按键数量: {expected_length}")

#             # 检测按键 - 使用改进的检测方法
#             segments, segment_times = self.audio_processor.isolate_keystrokes_ensemble(
#                 y, sr, expected_length
#             )

#             if not segments:
#                 print("未检测到按键")
#                 return None, 0

#             # 无需额外基于时间间隔过滤，因为isolate_keystrokes_ensemble已经考虑了这一点
#             filtered_segments = segments
#             filtered_times = segment_times

#             # 提取每个按键的概率分布
#             position_info = []

#             for i, segment in enumerate(filtered_segments):
#                 # 提取特征
#                 features = self.feature_extractor.extract_features(segment, sr)
#                 features = features.reshape(1, -1)
#                 features_scaled = self.model_trainer.scaler.transform(features)

#                 # 获取所有可用模型的概率分布
#                 model_probs = {}

#                 # 1. CNN模型概率
#                 if 'cnn' in self.sound_models:
#                     try:
#                         # 使用与训练相同的形状 [batch_size, features, 1]
#                         features_reshaped = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1],
#                                                                     1)
#                         cnn_probs = self.sound_models['cnn'].predict(features_reshaped, verbose=0)[0]
#                         model_probs['cnn'] = cnn_probs
#                     except Exception as e:
#                         print(f"CNN模型预测出错: {e}")

#                 # 2. LSTM模型概率 - 修复维度问题
#                 if 'lstm' in self.sound_models:
#                     try:
#                         # 使用与训练完全相同的形状变换 [batch_size, features, 1]
#                         # 注意这里与CNN形状相同而不是转置维度
#                         lstm_input = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
#                         lstm_probs = self.sound_models['lstm'].predict(lstm_input, verbose=0)[0]
#                         model_probs['lstm'] = lstm_probs
#                     except Exception as e:
#                         print(f"LSTM模型预测出错，跳过: {e}")
#                         traceback.print_exc()

#                 # 3. 随机森林模型
#                 if hasattr(self, 'random_forest') and self.random_forest:
#                     rf_pred = self.random_forest.predict_proba(features_scaled)[0]
#                     rf_probs = np.zeros(len(self.idx_to_class))
#                     for i, prob in enumerate(rf_pred):
#                         class_idx = int(self.random_forest.classes_[i])
#                         rf_probs[class_idx] = prob
#                     model_probs['rf'] = rf_probs

#                 # 4. 梯度提升模型
#                 if hasattr(self, 'gradient_boosting') and self.gradient_boosting:
#                     gb_pred = self.gradient_boosting.predict_proba(features_scaled)[0]
#                     gb_probs = np.zeros(len(self.idx_to_class))
#                     for i, prob in enumerate(gb_pred):
#                         class_idx = int(self.gradient_boosting.classes_[i])
#                         gb_probs[class_idx] = prob
#                     model_probs['gb'] = gb_probs

#                 # 4. 整合概率分布
#                 if len(model_probs) > 0:
#                     # 设置模型权重
#                     weights = {
#                         'cnn': 0.6,
#                         'lstm': 0.4,
#                         # 添加其他模型权重
#                     }

#                     # 初始化综合概率分布
#                     num_classes = len(model_probs['cnn']) if 'cnn' in model_probs else 0
#                     combined_probs = np.zeros(num_classes)
#                     total_weight = 0

#                     # 加权平均
#                     for model_name, probs in model_probs.items():
#                         if model_name in weights:
#                             combined_probs += weights[model_name] * probs
#                             total_weight += weights[model_name]

#                     # 归一化
#                     if total_weight > 0:
#                         combined_probs /= total_weight

#                     # 使用整合后的概率分布
#                     probs = combined_probs
#                 else:
#                     # 如果没有可用模型，使用CNN模型概率
#                     probs = model_probs.get('cnn', np.zeros(len(self.idx_to_class)))

#                 # 分析概率分布
#                 top_indices = np.argsort(probs)[-3:][::-1]
#                 top_probs = probs[top_indices]

#                 # 将索引转换为字符
#                 top_chars = []
#                 for idx in top_indices:
#                     if hasattr(self, 'idx_to_class') and self.idx_to_class:
#                         if str(idx) in self.idx_to_class:
#                             char = self.idx_to_class[str(idx)]
#                         else:
#                             char = str(idx)
#                     else:
#                         char = str(idx)
#                     top_chars.append(char)

#                 # 计算各种特征
#                 entropy = -np.sum(probs * np.log(probs + 1e-9))
#                 contrast = 0
#                 if len(top_probs) >= 2:
#                     contrast = float(top_probs[0] - top_probs[1])
#                 top3_certainty = float(np.sum(top_probs))

#                 # 保存位置信息
#                 position_info.append({
#                     'position': i,
#                     'top_chars': top_chars,
#                     'top_probs': top_probs.tolist(),
#                     'entropy': float(entropy),
#                     'contrast': contrast,
#                     'top3_certainty': top3_certainty
#                 })

#             print(f"提取了 {len(filtered_segments)} 个按键的概率分布")
#             return position_info, len(filtered_segments)

#         except Exception as e:
#             print(f"获取按键概率分布时出错: {e}")
#             traceback.print_exc()
#             return None, 0

#     def _predict_with_single_mask(self, mask, position_info):
#         """使用单个掩码进行Seq2Seq预测"""
#         try:
#             # 准备输入
#             input_ids = [Config.char2idx.get(c, Config.char2idx[Config.MASK_TOKEN]) for c in mask]
#             src = torch.tensor(input_ids).unsqueeze(1).to(self.device)
#             src_len = torch.tensor([len(input_ids)]).to(self.device)

#             # 使用波束搜索解码
#             beam_result = self.seq2seq_model.beam_decode(src, src_len, beam_width=Config.BEAM_WIDTH)

#             if beam_result:
#                 sequence, score, scores_list, _, _ = beam_result

#                 # 提取文本
#                 output_text = ''.join([Config.idx2char[i] for i in sequence
#                                        if i not in {Config.char2idx[Config.SOS_TOKEN],
#                                                     Config.char2idx[Config.EOS_TOKEN],
#                                                     Config.char2idx[Config.PAD_TOKEN]}])

#                 return {
#                     'text': output_text,
#                     'seq_score': score / len(sequence),
#                 }

#             return None
#         except Exception as e:
#             print(f"处理掩码 {mask} 时出错: {e}")
#             return None

#     def _predict_results_by_mask_count(self, mask, position_info):
#         """根据掩码数量生成相应数量的预测结果"""
#         mask_count = mask.count(Config.MASK_TOKEN)

#         # 根据掩码数量决定生成结果数
#         if mask_count == 1:
#             num_results = 8
#         elif mask_count == 2:
#             num_results = 24
#         elif mask_count == 3:
#             num_results = 72
#         else:
#             num_results = max(4, mask_count * 3)  # 默认至少4个结果

#         results = []

#         # 准备输入
#         input_ids = [Config.char2idx.get(c, Config.char2idx[Config.MASK_TOKEN]) for c in mask]
#         src = torch.tensor(input_ids).unsqueeze(1).to(self.device)
#         src_len = torch.tensor([len(input_ids)]).to(self.device)

#         # 使用波束搜索解码
#         beam_search_width = min(num_results * 2, Config.BEAM_WIDTH)

#         # 执行波束搜索，收集多个结果
#         beam_results = self.seq2seq_model.beam_search_multiple(
#             src, src_len,
#             max_len=Config.MAX_LEN,
#             beam_width=beam_search_width,
#             num_results=num_results
#         )

#         # 处理结果
#         if beam_results:
#             for seq, score, scores_list, output_text in beam_results:
#                 # 计算掩码匹配得分
#                 mask_match_score = 0
#                 mask_match_count = 0

#                 for i, (a, b) in enumerate(zip(mask, output_text)):
#                     if a != Config.MASK_TOKEN and i < len(output_text):
#                         mask_match_count += 1
#                         if a == b:
#                             mask_match_score += 1

#                 if mask_match_count > 0:
#                     mask_match_score /= mask_match_count

#                 # 综合得分
#                 combined_score = 0.7 * score + 0.3 * mask_match_score

#                 results.append({
#                     'text': output_text,
#                     'mask': mask,
#                     'seq_score': score,
#                     'mask_match_score': mask_match_score,
#                     'combined_score': combined_score
#                 })

#         return results

#     def _evaluate_accuracy(self, expected_sequence, predicted_sequence):
#         """计算准确率 - 考虑长度不匹配情况"""
#         if not expected_sequence or not predicted_sequence:
#             return 0.0

#         min_len = min(len(expected_sequence), len(predicted_sequence))
#         if min_len == 0:
#             return 0.0

#         correct_chars = sum(1 for i in range(min_len) if expected_sequence[i] == predicted_sequence[i])

#         # 使用预期序列长度作为分母
#         return correct_chars / len(expected_sequence)

#     def predict_with_enhanced_masks(self, audio_file_path, top_k=30, verbose=True, compare_basic=True):
#         """使用增强的掩码预测方法，并与基础声音模型对比

#         Args:
#             audio_file_path: 音频文件路径
#             top_k: 返回的最佳结果数量
#             verbose: 是否显示详细信息
#             compare_basic: 是否与基础声音模型对比

#         Returns:
#             dict: 包含高级预测和基础预测的结果字典
#         """
#         if verbose:
#             print(f"对文件进行高级预测: {audio_file_path}")


#         # 1. 先进行基础声音模型预测（如果需要对比）
#         basic_prediction = None
#         if compare_basic:
#             try:
#                 print(f"\n正在使用基础声音模型进行预测...")
#                 basic_prediction = self.basic_system.predict_from_file(audio_file_path, verbose=False)
#                 print(f"基础声音模型预测结果: {basic_prediction}")
#             except Exception as e:
#                 print(f"基础声音模型预测失败: {e}")

#         # 2. 提取按键概率特征并继续高级预测
#         position_info, num_keystrokes = self.extract_keystroke_probabilities(audio_file_path)

#         if position_info is None or num_keystrokes == 0:
#             print("无法提取按键概率，无法进行预测")
#             return {'advanced': [], 'basic': basic_prediction, 'improvement': 0, 'accuracy_stats': {}}

#         # 使用增强的掩码生成器创建掩码
#         if verbose:
#             print(f"生成掩码模板...")

#         scored_masks = self.mask_generator.generate_masks(
#             position_info,
#             base_prediction=basic_prediction  # 传递基础预测
#         )

#         if verbose:
#             print(f"生成了 {len(scored_masks)} 个掩码:")
#             for i, (mask, score) in enumerate(scored_masks[:10]):  # 只显示前10个
#                 template_name = next((name for name, template in self.mask_generator.templates.items()
#                                       if template == mask), f"掩码 {i + 1}")
#                 print(f"  {template_name}: {mask} (得分: {score:.4f})")

#         if not scored_masks:
#             print("未能生成有效掩码，无法进行预测")
#             return {'advanced': [], 'basic': basic_prediction, 'improvement': 0, 'accuracy_stats': {}}

#         # 准备存储所有预测结果
#         all_results = []

#         # 对每个掩码进行预测
#         mask_counter = 0
#         for mask, mask_score in scored_masks:
#             try:
#                 mask_counter += 1
#                 if verbose and mask_counter <= 5:  # 只为前5个掩码显示详细信息
#                     template_name = next((name for name, template in self.mask_generator.templates.items()
#                                           if template == mask), "未命名掩码")
#                     print(f"\n处理掩码{mask_counter}/{len(scored_masks)}: {template_name} - {mask}")

#                 # 根据掩码的掩码数量生成不同数量的结果
#                 mask_results = self._predict_results_by_mask_count(mask, position_info)

#                 if mask_results:
#                     # 添加掩码得分
#                     for result in mask_results:
#                         result['mask_score'] = mask_score

#                         # 计算综合得分
#                         result['combined_score'] = (
#                                 0.5 * result['seq_score'] +
#                                 0.3 * result['mask_match_score'] +
#                                 0.2 * mask_score
#                         )

#                         # 添加掩码模板名称
#                         result['template_name'] = next(
#                             (name for name, template in self.mask_generator.templates.items()
#                              if template == mask),
#                             f"掩码-{mask_counter}"
#                         )

#                     all_results.extend(mask_results)

#                     if verbose and mask_counter <= 5:
#                         print(f"  生成了 {len(mask_results)} 个预测结果")

#             except Exception as e:
#                 print(f"处理掩码 {mask} 时出错: {e}")
#                 traceback.print_exc()

#         # 对结果排序
#         all_results = sorted(all_results, key=lambda x: x['combined_score'], reverse=True)

#         # 去除重复
#         seen = set()
#         filtered_results = []
#         for result in all_results:
#             if result['text'] not in seen:
#                 seen.add(result['text'])
#                 filtered_results.append(result)

#         # 输出部分结果
#         if verbose and filtered_results:
#             print(f"\n预测生成完成! 总共 {len(filtered_results)} 个唯一结果")
#             print("\n最佳10个预测结果:")
#             for i, result in enumerate(filtered_results[:10]):
#                 print(
#                     f"{i + 1}. {result['text']} (得分: {result['combined_score']:.4f}, 模板: {result['template_name']})")

#         # 3. 准确率评估
#         accuracy_stats = {
#             'sound_model': 0,
#             'pure_seq2seq': 0,
#             'combined_model': 0,
#             'sound_model_prediction': "",
#             'pure_seq2seq_prediction': "",
#             'combined_model_prediction': "",
#             'best_model': ""
#         }

#         # 从文件名提取预期序列 - 确保此变量存在
#         expected_sequence = ''.join(c for c in os.path.splitext(os.path.basename(audio_file_path))[0] if c.isdigit())

#         if expected_sequence:
#             # 1. 声音模型准确率
#             if basic_prediction:
#                 accuracy_stats['sound_model_prediction'] = basic_prediction
#                 accuracy_stats['sound_model'] = self._evaluate_accuracy(expected_sequence, basic_prediction)

#             # 2. 纯Seq2Seq模型准确率 (全掩码)
#             full_mask = Config.MASK_TOKEN * len(position_info)
#             seq2seq_result = self._predict_with_single_mask(full_mask, position_info)

#             if seq2seq_result:
#                 seq2seq_text = seq2seq_result['text']
#                 accuracy_stats['pure_seq2seq_prediction'] = seq2seq_text
#                 accuracy_stats['pure_seq2seq'] = self._evaluate_accuracy(expected_sequence, seq2seq_text)

#             # 3. 组合模型 - 选择准确率更高的模型
#             sound_accuracy = accuracy_stats['sound_model']
#             seq2seq_accuracy = accuracy_stats['pure_seq2seq']
#             advanced_mask_accuracy = 0

#             if filtered_results:
#                 advanced_text = filtered_results[0]['text']
#                 advanced_mask_accuracy = self._evaluate_accuracy(expected_sequence, advanced_text)

#             # 选择三种模型中准确率最高的作为组合模型结果
#             if sound_accuracy >= seq2seq_accuracy and sound_accuracy >= advanced_mask_accuracy:
#                 # 声音模型最好
#                 accuracy_stats['combined_model_prediction'] = basic_prediction
#                 accuracy_stats['combined_model'] = sound_accuracy
#             elif seq2seq_accuracy >= sound_accuracy and seq2seq_accuracy >= advanced_mask_accuracy:
#                 # 纯Seq2Seq最好
#                 accuracy_stats['combined_model_prediction'] = seq2seq_text
#                 accuracy_stats['combined_model'] = seq2seq_accuracy
#             else:
#                 # 掩码+Seq2Seq最好
#                 accuracy_stats['combined_model_prediction'] = advanced_text
#                 accuracy_stats['combined_model'] = advanced_mask_accuracy

#             # 添加打印所有三种模型的准确率
#             if verbose:
#                 print("\n准确率对比:")
#                 print(f"预期序列: {expected_sequence}")
#                 print(f"声音模型: {basic_prediction} - 准确率: {accuracy_stats['sound_model']:.2%}")
#                 print(
#                     f"纯Seq2Seq: {accuracy_stats['pure_seq2seq_prediction']} - 准确率: {accuracy_stats['pure_seq2seq']:.2%}")
#                 print(f"掩码+Seq2Seq: {filtered_results[0]['text']} - 准确率: {advanced_mask_accuracy:.2%}")
#                 print(f"最佳模型: {accuracy_stats['best_model']} - 准确率: {accuracy_stats['combined_model']:.2%}")

#         # 4. 计算提升率
#         improvement = 0
#         if basic_prediction and filtered_results:
#             if expected_sequence:
#                 # 计算基础预测准确率
#                 basic_accuracy = accuracy_stats['sound_model']

#                 # 计算高级预测准确率
#                 advanced_accuracy = accuracy_stats['combined_model']

#                 # 计算相对提升
#                 if basic_accuracy > 0:
#                     improvement = (advanced_accuracy - basic_accuracy) / basic_accuracy * 100
#                 else:
#                     improvement = float('inf') if advanced_accuracy > 0 else 0

#                 if verbose:
#                     if improvement != float('inf'):
#                         print(f"相对于声音模型的准确率提升: {improvement:.2f}%")
#                     else:
#                         print(f"相对于声音模型的准确率提升: 无限 (基础模型准确率为0)")

#         # 添加统计信息
#         self.stats['processed_files'] += 1
#         if filtered_results:
#             self.stats['successful_predictions'] += 1

#         return {
#             'advanced': filtered_results[:top_k],
#             'basic': basic_prediction,
#             'improvement': improvement,
#             'accuracy_stats': accuracy_stats
#         }

#     def predict_directory(self, dir_path, top_k=30, verbose=True, save_viz=False):
#         """对目录中的所有音频文件进行预测

#         Args:
#             dir_path: 目录路径
#             top_k: 每个文件返回的最佳结果数量
#             verbose: 是否显示详细信息
#             save_viz: 是否保存可视化结果

#         Returns:
#             预测结果字典
#         """
#         if not os.path.exists(dir_path):
#             print(f"错误: 目录不存在: {dir_path}")
#             return {}

#         if not os.path.isdir(dir_path):
#             print(f"错误: {dir_path} 不是目录")
#             return {}

#         # 查找目录中的所有WAV文件
#         wav_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.wav')]
#         if not wav_files:
#             print(f"错误: 目录 {dir_path} 中没有WAV文件")
#             return {}

#         print(f"在目录 {dir_path} 中找到 {len(wav_files)} 个WAV文件")

#         # 准备结果保存
#         # 使用更具描述性的CSV文件名，包含时间戳
#         results_file = os.path.join(dir_path, f"advanced_prediction_results_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        
#         # CSV表头 - 确保与后面写入的数据列对应
#         csv_headers = [
#             "文件名", "预期序列",
#             "声音模型预测", "声音模型准确率(字符)", "声音模型准确率(序列)", # 新增序列准确率
#             "纯Seq2Seq预测", "纯Seq2Seq准确率(字符)", "纯Seq2Seq准确率(序列)", # 新增序列准确率
#             "高级模型预测", "高级模型准确率(字符)", "高级模型准确率(序列)", # 新增序列准确率
#             "提升率(字符级,%)", "最佳高级模板名称", "最佳高级掩码"
#         ]
#         with open(results_file, 'w', encoding='utf-8', newline='') as f: # newline='' 推荐用于csv
#             import csv # 导入csv模块
#             writer = csv.writer(f)
#             writer.writerow(csv_headers)


#         # 统计信息初始化 (针对本次目录运行)
#         current_run_stats = defaultdict(float) # 使用defaultdict简化初始化
#         current_run_stats['start_time'] = time.time()
#         current_run_stats['best_model_counts'] = defaultdict(int)


#         # 处理每个文件
#         all_file_results_map = {} # 存储每个文件的详细预测结果字典

#         print("\n开始批量预测...")

#         for i, filename in enumerate(wav_files):
#             file_path = os.path.join(dir_path, filename)
#             print(f"\n[{i + 1}/{len(wav_files)}] 处理文件: {filename}")

#             expected_sequence = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
#             if expected_sequence:
#                 print(f"  从文件名提取的预期序列: '{expected_sequence}'")
#             else:
#                 print(f"  警告: 文件 '{filename}' 未能提取到预期序列。准确率将不计算。")

#             try:
#                 # 调用核心预测函数，获取包含所有准确率信息的字典
#                 # 注意: predict_with_enhanced_masks 需要返回一个包含 'accuracy_stats' 的字典，
#                 # 其中 accuracy_stats 应包含类似:
#                 # 'sound_model_prediction', 'sound_model_char_accuracy', 'sound_model_sequence_accuracy',
#                 # 'pure_seq2seq_prediction', 'pure_seq2seq_char_accuracy', 'pure_seq2seq_sequence_accuracy',
#                 # 'advanced_model_prediction', 'advanced_model_char_accuracy', 'advanced_model_sequence_accuracy'
#                 # 以及 'improvement_char_level'
#                 results_dict = self.predict_with_enhanced_masks(
#                     file_path,
#                     top_k=top_k, # top_k 用于 predict_with_enhanced_masks 内部决定返回多少高级候选
#                     verbose=verbose, # 控制 predict_with_enhanced_masks 内部的打印
#                     compare_basic=True
#                 )

#                 acc_stats = results_dict.get('accuracy_stats', {})

#                 # 安全获取预测文本和准确率值，提供默认值
#                 sound_pred_text = acc_stats.get('sound_model_prediction', "N/A")
#                 sound_char_acc = float(acc_stats.get('sound_model_char_accuracy', 0.0))
#                 sound_seq_acc = float(acc_stats.get('sound_model_sequence_accuracy', 0.0))

#                 pure_s2s_pred_text = acc_stats.get('pure_seq2seq_prediction', "N/A")
#                 pure_s2s_char_acc = float(acc_stats.get('pure_seq2seq_char_accuracy', 0.0))
#                 pure_s2s_seq_acc = float(acc_stats.get('pure_seq2seq_sequence_accuracy', 0.0))
                
#                 advanced_pred_text = "N/A"
#                 best_adv_template_name = "N/A"
#                 best_adv_mask_str = "N/A"
                
#                 if results_dict.get('advanced'): # 检查是否有高级预测结果
#                     advanced_pred_text = results_dict['advanced'][0].get('text', "N/A")
#                     best_adv_template_name = results_dict['advanced'][0].get('template_name', 'N/A')
#                     best_adv_mask_str = results_dict['advanced'][0].get('mask', 'N/A')
                
#                 # 如果 acc_stats 中有 advanced_model 的准确率，则使用它，否则设为0
#                 adv_char_acc = float(acc_stats.get('advanced_model_char_accuracy', 0.0))
#                 adv_seq_acc = float(acc_stats.get('advanced_model_sequence_accuracy', 0.0))


#                 # 计算相对于声音模型的字符级提升率 (确保类型安全)
#                 improvement_char_level_val = results_dict.get('improvement_char_level', 0.0)
#                 improvement_display_str = "0.00%"
#                 improvement_csv_val = "0.00"

#                 if isinstance(improvement_char_level_val, (int, float)):
#                     if improvement_char_level_val == float('inf'):
#                         improvement_display_str = "inf%"
#                         improvement_csv_val = "inf"
#                     elif improvement_char_level_val != 0.0:
#                         improvement_display_str = f"{improvement_char_level_val:.2f}%"
#                         improvement_csv_val = f"{improvement_char_level_val:.2f}"
                
#                 # 打印当前文件的简要结果 (替换原来的报错print)
#                 print(
#                     f"  最佳高级预测: '{advanced_pred_text}' "
#                     f"(字符准确率: {adv_char_acc:.2%}, 序列准确率: {adv_seq_acc:.0%}, "
#                     f"相较声音模型字符级提升: {improvement_display_str})"
#                 )

#                 # 保存到CSV
#                 with open(results_file, 'a', encoding='utf-8', newline='') as f:
#                     writer = csv.writer(f)
#                     writer.writerow([
#                         filename, expected_sequence,
#                         sound_pred_text, f"{sound_char_acc:.4f}", f"{sound_seq_acc:.0f}",
#                         pure_s2s_pred_text, f"{pure_s2s_char_acc:.4f}", f"{pure_s2s_seq_acc:.0f}",
#                         advanced_pred_text, f"{adv_char_acc:.4f}", f"{adv_seq_acc:.0f}",
#                         improvement_csv_val, # 存数值或'inf'
#                         best_adv_template_name,
#                         best_adv_mask_str
#                     ])
                
#                 all_file_results_map[filename] = results_dict # 存储完整结果供返回

#                 # 更新总体统计数据 (仅当有预期序列时)
#                 if expected_sequence:
#                     current_run_stats['total_files_with_expected_sequence'] += 1
#                     seq_len = len(expected_sequence)
#                     if seq_len > 0:
#                         current_run_stats['total_expected_chars'] += seq_len
                        
#                         current_run_stats['sound_model_total_correct_chars'] += sound_char_acc * seq_len
#                         current_run_stats['sound_model_total_correct_sequences'] += sound_seq_acc
                        
#                         current_run_stats['pure_seq2seq_total_correct_chars'] += pure_s2s_char_acc * seq_len
#                         current_run_stats['pure_seq2seq_total_correct_sequences'] += pure_s2s_seq_acc

#                         current_run_stats['advanced_model_total_correct_chars'] += adv_char_acc * seq_len
#                         current_run_stats['advanced_model_total_correct_sequences'] += adv_seq_acc
                        
#                         # 更新 best_model_counts (基于字符准确率)
#                         # 这个逻辑需要 predict_with_enhanced_masks 返回一个明确的 'best_model_type' 指示
#                         # 或者我们在这里重新判断。为简单起见，假设高级模型是我们的目标。
#                         # 如果高级模型准确率最高，则记为高级模型胜出。
#                         if adv_char_acc >= sound_char_acc and adv_char_acc >= pure_s2s_char_acc:
#                             if adv_char_acc > sound_char_acc or adv_char_acc > pure_s2s_char_acc : # 严格优于至少一个
#                                 current_run_stats['best_model_counts']['advanced'] += 1
#                             else: # 打平
#                                 current_run_stats['best_model_counts']['tie'] += 1
#                         elif sound_char_acc >= pure_s2s_char_acc:
#                              current_run_stats['best_model_counts']['sound'] += 1
#                         else:
#                              current_run_stats['best_model_counts']['pure_seq2seq'] += 1


#                 # 可视化
#                 if save_viz and self.basic_system: # 确保 basic_system 可用
#                     try:
#                         create_comparison_visualization(
#                             file_path,
#                             expected_sequence or "N/A",
#                             sound_pred_text,
#                             advanced_pred_text,
#                             self.basic_system # 传递 KeystrokeRecognitionSystem 实例
#                         )
#                         print(f"  对比可视化已保存。")
#                     except Exception as e_viz:
#                         print(f"  可视化出错 for {filename}: {e_viz}")
            
#             except Exception as e_file_proc:
#                 print(f"处理文件 {filename} 时发生严重错误: {e_file_proc}")
#                 traceback.print_exc()
#                 with open(results_file, 'a', encoding='utf-8', newline='') as f:
#                     writer = csv.writer(f)
#                     # 写入错误行到CSV
#                     error_row = [filename, expected_sequence] + ["ERROR"] * 3 + ["0.0000"] * 6 + ["0.00", "N/A", "N/A"]
#                     writer.writerow(error_row)

#         # ----- 循环结束后，计算并打印总体统计信息 -----
#         elapsed_time_run = time.time() - current_run_stats['start_time']
#         print("\n====== 目录预测统计信息 (本次运行) ======")
        
#         total_valid_files_for_stats = current_run_stats['total_files_with_expected_sequence']
#         total_exp_chars_for_stats = current_run_stats['total_expected_chars']

#         print(f"已处理文件总数 (有预期序列的): {int(total_valid_files_for_stats)}")
#         if total_valid_files_for_stats == 0:
#             print("没有文件带有可用于统计的预期序列。无法计算总体准确率。")
#             return all_file_results_map

#         print(f"总预期字符数: {int(total_exp_chars_for_stats)}")
#         print(f"处理时间: {elapsed_time_run:.2f}秒 (平均: {elapsed_time_run/max(1, len(wav_files)):.2f}秒/文件)")

#         # 计算各模型总体准确率
#         sm_char_acc_overall = (current_run_stats['sound_model_total_correct_chars'] / total_exp_chars_for_stats) if total_exp_chars_for_stats > 0 else 0
#         sm_seq_acc_overall = (current_run_stats['sound_model_total_correct_sequences'] / total_valid_files_for_stats)
#         print(f"\n声音模型总体: 字符准确率={sm_char_acc_overall:.2%}, 序列准确率={sm_seq_acc_overall:.2%}")

#         ps_char_acc_overall = (current_run_stats['pure_seq2seq_total_correct_chars'] / total_exp_chars_for_stats) if total_exp_chars_for_stats > 0 else 0
#         ps_seq_acc_overall = (current_run_stats['pure_seq2seq_total_correct_sequences'] / total_valid_files_for_stats)
#         print(f"纯Seq2Seq总体: 字符准确率={ps_char_acc_overall:.2%}, 序列准确率={ps_seq_acc_overall:.2%}")

#         am_char_acc_overall = (current_run_stats['advanced_model_total_correct_chars'] / total_exp_chars_for_stats) if total_exp_chars_for_stats > 0 else 0
#         am_seq_acc_overall = (current_run_stats['advanced_model_total_correct_sequences'] / total_valid_files_for_stats)
#         print(f"高级模型总体: 字符准确率={am_char_acc_overall:.2%}, 序列准确率={am_seq_acc_overall:.2%}")

#         # 计算字符级提升
#         if sm_char_acc_overall > 0 and am_char_acc_overall > sm_char_acc_overall:
#             char_improvement_overall = (am_char_acc_overall - sm_char_acc_overall) / sm_char_acc_overall * 100
#             print(f"高级模型相较于声音模型的字符准确率提升: {char_improvement_overall:.2f}%")
#         elif am_char_acc_overall > 0 and sm_char_acc_overall == 0:
#             print(f"高级模型相较于声音模型的字符准确率提升: ∞ (声音模型准确率为0)")
        
#         print(f"\n模型胜出统计 (基于字符准确率):")
#         for model_type, count in current_run_stats['best_model_counts'].items():
#              print(f"  {model_type}: {count} 次 ({count / total_valid_files_for_stats:.2%})")


#         print(f"\n详细结果已保存至: {results_file}")
#         return all_file_results_map


# # 对比可视化函数
# def create_comparison_visualization(file_path, expected, basic, advanced, system):
#     """创建对比可视化

#     Args:
#         file_path: 音频文件路径
#         expected: 预期序列
#         basic: 基础预测结果
#         advanced: 高级预测结果
#         system: KeystrokeRecognitionSystem实例
#     """
#     import matplotlib.pyplot as plt
#     from datetime import datetime
#     import numpy as np

#     # 加载音频
#     y, sr = system.audio_processor.load_audio(file_path)

#     # 检测按键
#     segments, segment_times, energy = system.audio_processor.detect_keystrokes(y, sr)

#     # 创建可视化
#     plt.figure(figsize=(15, 8))

#     # 1. 显示波形
#     plt.subplot(3, 1, 1)
#     times = np.linspace(0, len(y) / sr, len(y))
#     plt.plot(times, y)
#     plt.title("音频波形")

#     # 标记检测到的按键
#     for i, (start, end) in enumerate(segment_times):
#         plt.axvspan(start, end, alpha=0.2, color='red')
#         plt.text((start + end) / 2, plt.ylim()[1] * 0.9, f"{i + 1}",
#                  horizontalalignment='center', fontsize=10)

#     # 2. 显示基础预测
#     plt.subplot(3, 1, 2)
#     plt.plot(times, y)
#     plt.title("基础声音模型预测")

#     # 标记检测到的按键并添加基础预测标签
#     for i, ((start, end), pred) in enumerate(zip(segment_times, basic[:len(segment_times)])):
#         plt.axvspan(start, end, alpha=0.2, color='blue')
#         plt.text((start + end) / 2, plt.ylim()[1] * 0.9, pred,
#                  horizontalalignment='center', fontsize=12, color='blue')

#     # 3. 显示高级预测
#     plt.subplot(3, 1, 3)
#     plt.plot(times, y)
#     plt.title("高级模型预测 (掩码+Seq2Seq)")

#     # 标记检测到的按键并添加高级预测标签
#     for i, ((start, end), pred) in enumerate(zip(segment_times, advanced[:len(segment_times)])):
#         plt.axvspan(start, end, alpha=0.2, color='green')
#         plt.text((start + end) / 2, plt.ylim()[1] * 0.9, pred,
#                  horizontalalignment='center', fontsize=12, color='green')

#     # 添加标题信息
#     plt.suptitle(f"预测对比: {'预期: ' + expected if expected else '未知预期序列'}\n"
#                  f"基础预测: {basic} | 高级预测: {advanced}", fontsize=14)

#     plt.tight_layout()

#     # 保存图像
#     results_dir = system.config_manager.get_path("results_dir")
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     save_path = os.path.join(results_dir, f'prediction_comparison_{timestamp}.png')
#     plt.savefig(save_path)
#     plt.close()

#     print(f"对比可视化已保存至: {save_path}")


# def advanced_predict_file(config_manager):
#     """高级按键预测函数（单个文件），包含基础模型对比

#     Args:
#         config_manager: 配置管理器实例

#     Returns:
#         bool: 是否成功
#     """
#     # 检查Seq2Seq模型文件
#     seq2seq_model_path = "seq_best_model.pth"
#     if not os.path.exists(seq2seq_model_path):
#         print(f"警告: Seq2Seq模型文件不存在: {seq2seq_model_path}")
#         use_seq2seq = input("是否继续高级预测（将使用随机初始化的Seq2Seq模型）? (y/n): ")
#         if use_seq2seq.lower() != 'y':
#             print("高级预测已取消")
#             return False

#     # 初始化增强预测系统
#     prediction_system = EnhancedPredictionSystem(config_manager, seq2seq_model_path)

#     # 获取音频文件路径
#     file_path = input("请输入音频文件路径: ")
#     if not os.path.exists(file_path):
#         print(f"错误: 文件不存在: {file_path}")
#         return False

#     # 获取其他参数
#     top_k = int(input("返回的最佳结果数量 [默认30]: ") or "30")

#     # 进行预测
#     print("\n开始高级预测...")
#     start_time = time.time()
#     results = prediction_system.predict_with_enhanced_masks(
#         file_path,
#         top_k=top_k,
#         verbose=True,
#         compare_basic=True
#     )
#     elapsed_time = time.time() - start_time

#     # 输出结果
#     if results['advanced']:
#         print(f"\n预测完成! (用时: {elapsed_time:.2f}秒)")

#         # 提取预期序列
#         expected_sequence = ''.join(c for c in os.path.splitext(os.path.basename(file_path))[0] if c.isdigit())

#         # 三种预测结果
#         basic_prediction = results['accuracy_stats']['sound_model_prediction'] or ""
#         seq2seq_prediction = results['accuracy_stats']['pure_seq2seq_prediction'] or ""
#         combined_prediction = results['advanced'][0]['text']

#         # 三种准确率
#         basic_accuracy = results['accuracy_stats']['sound_model']
#         seq2seq_accuracy = results['accuracy_stats']['pure_seq2seq']
#         combined_accuracy = results['accuracy_stats']['combined_model']

#         print("\n预测结果对比:")
#         if expected_sequence:
#             print(f"预期序列:     {expected_sequence}")
#         print(f"声音模型预测: {basic_prediction}" + (f" (准确率: {basic_accuracy:.2%})" if expected_sequence else ""))
#         print(f"纯Seq2Seq预测: {seq2seq_prediction}" + (
#             f" (准确率: {seq2seq_accuracy:.2%})" if expected_sequence else ""))
#         print(f"组合模型预测: {combined_prediction}" + (
#             f" (准确率: {combined_accuracy:.2%})" if expected_sequence else ""))

#         if expected_sequence:
#             # 计算提升
#             if basic_accuracy > 0:
#                 vs_sound = (combined_accuracy - basic_accuracy) / basic_accuracy * 100
#                 print(f"相对于声音模型提升: {vs_sound:.2f}%")
#             if seq2seq_accuracy > 0:
#                 vs_seq2seq = (combined_accuracy - seq2seq_accuracy) / seq2seq_accuracy * 100
#                 print(f"相对于纯Seq2Seq提升: {vs_seq2seq:.2f}%")

#         # 输出详细结果
#         print("\n组合模型预测详情:")
#         for i, result in enumerate(results['advanced'][:10]):  # 只显示前10个
#             print(f"{i + 1}. {result['text']} (综合得分: {result['combined_score']:.4f})")
#             print(f"   模板类型: {result['template_name']}")
#             print(f"   掩码: {result['mask']}")
#             print(f"   掩码率: {result['mask'].count(Config.MASK_TOKEN) / len(result['mask']):.2%}")
#             if 'seq_score' in result:
#                 print(f"   序列得分: {result['seq_score']:.4f}, "
#                       f"掩码匹配度: {result['mask_match_score']:.4f}, "
#                       f"掩码质量: {result['mask_score']:.4f}")

#         # 可视化（添加对比可视化）
#         try:
#             create_comparison_visualization(
#                 file_path,
#                 expected_sequence if expected_sequence else "",
#                 basic_prediction,
#                 combined_prediction,
#                 prediction_system.basic_system
#             )
#         except Exception as e:
#             print(f"可视化结果时出错: {e}")

#         return True
#     else:
#         print("未能生成预测结果")
#         return False


# def advanced_predict_directory(config_manager):
#     """对目录中的所有音频文件进行高级预测

#     Args:
#         config_manager: 配置管理器实例

#     Returns:
#         bool: 是否成功
#     """
#     # 检查Seq2Seq模型文件
#     seq2seq_model_path = "seq_best_model.pth"
#     if not os.path.exists(seq2seq_model_path):
#         print(f"警告: Seq2Seq模型文件不存在: {seq2seq_model_path}")
#         use_seq2seq = input("是否继续高级预测（将使用随机初始化的Seq2Seq模型）? (y/n): ")
#         if use_seq2seq.lower() != 'y':
#             print("高级预测已取消")
#             return False

#     # 初始化增强预测系统
#     prediction_system = EnhancedPredictionSystem(config_manager, seq2seq_model_path)

#     # 获取目录路径
#     dir_path = input("请输入音频文件目录路径: ")
#     if not os.path.exists(dir_path):
#         print(f"错误: 目录不存在: {dir_path}")
#         return False

#     if not os.path.isdir(dir_path):
#         print(f"错误: {dir_path} 不是目录")
#         return False

#     # 获取其他参数
#     top_k = int(input("每个文件返回的最佳结果数量 [默认30]: ") or "30")
#     save_viz = input("是否保存可视化结果 [y/n, 默认n]: ").lower() == 'y'
#     verbose = input("是否显示详细信息 [y/n, 默认n]: ").lower() == 'y'

#     # 进行预测
#     start_time = time.time()
#     results = prediction_system.predict_directory(
#         dir_path,
#         top_k=top_k,
#         verbose=verbose,
#         save_viz=save_viz
#     )
#     elapsed_time = time.time() - start_time

#     print(f"\n整个预测过程用时: {elapsed_time:.2f}秒")
#     return bool(results)


# if __name__ == "__main__":
#     # 单独运行时使用
#     from config_manager import ConfigManager

#     config = ConfigManager()
#     advanced_predict_file(config)


# import os
# import torch
# import numpy as np
# import librosa
# import soundfile as sf
# from pathlib import Path
# import matplotlib.pyplot as plt
# import json
# import time
# import pickle
# from collections import Counter, defaultdict
# from scipy.special import softmax
# import itertools
# import traceback

# # 导入必要的模块
# from config_manager import ConfigManager
# from audio_processing import AudioProcessor
# from feature_extraction import FeatureExtractor
# from keystroke_model import KeystrokeModelTrainer

# # 全局键盘映射
# KEYS = '1234567890qwertyuiopasdfghjklzxcvbnm'


# # Seq2Seq模型配置
# class Config:
#     """Seq2Seq模型配置类"""
#     SOS_TOKEN = "<SOS>"
#     EOS_TOKEN = "<EOS>"
#     PAD_TOKEN = "<PAD>"
#     MASK_TOKEN = "￥"
#     vocab = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN] + [chr(i) for i in range(32, 127)] + [MASK_TOKEN]
#     char2idx = {c: i for i, c in enumerate(vocab)}
#     idx2char = {i: c for i, c in enumerate(vocab)}
#     VOCAB_SIZE = len(vocab)
#     EMBED_DIM = 256
#     HIDDEN_DIM = 512
#     NUM_LAYERS = 2
#     DROPOUT = 0.5
#     BATCH_SIZE = 64
#     TEACHER_FORCING_RATIO = 0.5
#     BEAM_WIDTH = 50
#     MAX_LEN = 20
#     PATIENCE = 5

#     @classmethod
#     def get_device(cls):
#         try:
#             if torch.cuda.is_available():
#                 test_tensor = torch.randn(1).cuda()
#                 del test_tensor
#                 return torch.device('cuda')
#         except Exception as e:
#             print(f"CUDA不可用，使用CPU: {e}")
#         return torch.device('cpu')

#     DEVICE = None


# if Config.DEVICE is None:
#     Config.DEVICE = Config.get_device()


# # Seq2Seq模型组件定义
# class Attention(torch.nn.Module):
#     """注意力机制模块"""

#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.attn = torch.nn.Linear(hidden_dim * 2, hidden_dim)
#         self.v = torch.nn.Linear(hidden_dim, 1)

#     def forward(self, hidden, encoder_outputs):
#         seq_len = encoder_outputs.shape[0]
#         hidden = hidden.repeat(seq_len, 1, 1)
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
#         attention = self.v(energy).squeeze(2)
#         return torch.softmax(attention, dim=0)


# class Encoder(torch.nn.Module):
#     """Seq2Seq编码器"""

#     def __init__(self):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)
#         self.lstm = torch.nn.LSTM(Config.EMBED_DIM, Config.HIDDEN_DIM,
#                                   Config.NUM_LAYERS, dropout=Config.DROPOUT)

#     def forward(self, src, src_len):
#         embedded = self.embedding(src)
#         packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=False)
#         outputs, (hidden, cell) = self.lstm(packed)
#         outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
#         return outputs, hidden, cell


# class Decoder(torch.nn.Module):
#     """Seq2Seq解码器"""

#     def __init__(self):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)
#         self.attention = Attention(Config.HIDDEN_DIM)
#         self.lstm = torch.nn.LSTM(Config.EMBED_DIM + Config.HIDDEN_DIM, Config.HIDDEN_DIM,
#                                   Config.NUM_LAYERS, dropout=Config.DROPOUT)
#         self.fc = torch.nn.Linear(Config.HIDDEN_DIM * 2, Config.VOCAB_SIZE)

#     def forward(self, input, hidden, cell, encoder_outputs):
#         input = input.unsqueeze(0)
#         embedded = self.embedding(input)
#         attn_weights = self.attention(hidden[-1], encoder_outputs)
#         context = (attn_weights.unsqueeze(2) * encoder_outputs).sum(dim=0)
#         lstm_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)
#         output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
#         prediction = self.fc(torch.cat((output.squeeze(0), context), dim=1))
#         return prediction, hidden, cell


# class Seq2Seq(torch.nn.Module):
#     """Seq2Seq模型"""

#     def __init__(self):
#         super().__init__()
#         self.encoder = Encoder().to(Config.DEVICE)
#         self.decoder = Decoder().to(Config.DEVICE)

#     def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
#         batch_size = src.shape[1]
#         trg_len = trg.shape[0]

#         encoder_outputs, hidden, cell = self.encoder(src, src_len)

#         inputs = trg[0]
#         outputs = torch.zeros(trg_len, batch_size, Config.VOCAB_SIZE).to(Config.DEVICE)

#         for t in range(1, trg_len):
#             output, hidden, cell = self.decoder(inputs, hidden, cell, encoder_outputs)
#             outputs[t] = output
#             teacher_force = torch.rand(1).item() < teacher_forcing_ratio
#             top1 = output.argmax(1)
#             inputs = trg[t] if teacher_force else top1

#         return outputs

#     def beam_decode(self, src, src_len, max_len=Config.MAX_LEN, beam_width=Config.BEAM_WIDTH, alpha=0.6):
#         with torch.no_grad():
#             encoder_outputs, hidden, cell = self.encoder(src, src_len)

#             sequences = [
#                 [[Config.char2idx[Config.SOS_TOKEN]], 0.0, [], hidden, cell]
#             ]
#             completed_sequences = []

#             for _ in range(max_len):
#                 candidates = []

#                 for seq, score, scores_list, h, c in sequences:
#                     if seq[-1] == Config.char2idx[Config.EOS_TOKEN]:
#                         completed_sequences.append([seq, score, scores_list, h, c])
#                         continue

#                     input_token = torch.tensor([seq[-1]], device=Config.DEVICE)
#                     output, new_h, new_c = self.decoder(input_token, h, c, encoder_outputs)

#                     log_probs = torch.log_softmax(output, dim=1)
#                     topk_probs, topk_ids = log_probs.topk(beam_width)

#                     for i in range(topk_ids.size(1)):
#                         token_id = topk_ids[0][i].item()
#                         prob = topk_probs[0][i].item()

#                         new_seq = seq + [token_id]
#                         new_score = score + prob
#                         new_scores_list = scores_list + [prob]

#                         lp = ((5 + len(new_seq)) ** alpha) / ((5 + 1) ** alpha)
#                         normalized_score = new_score / lp

#                         candidates.append([new_seq, new_score, new_scores_list, new_h, new_c, normalized_score])

#                 if not candidates:
#                     break

#                 sequences = sorted(candidates, key=lambda x: x[5], reverse=True)[:beam_width]
#                 sequences = [[seq, score, scores_list, h, c] for seq, score, scores_list, h, c, _ in sequences]

#             all_sequences = completed_sequences + sequences
#             all_sequences = sorted(all_sequences, key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)

#             if all_sequences:
#                 best_seq = all_sequences[0]
#                 return best_seq[0], best_seq[1] / (len(best_seq[0]) ** alpha), best_seq[2]
#             return None

#     def beam_search_multiple(self, src, src_len, max_len=Config.MAX_LEN, beam_width=Config.BEAM_WIDTH, num_results=10):
#         with torch.no_grad():
#             encoder_outputs, hidden, cell = self.encoder(src, src_len)

#             sequences = [
#                 [[Config.char2idx[Config.SOS_TOKEN]], 0.0, [], hidden, cell]
#             ]
#             completed_sequences = []

#             for _ in range(max_len):
#                 candidates = []

#                 for seq, score, scores_list, h, c in sequences:
#                     if seq[-1] == Config.char2idx[Config.EOS_TOKEN]:
#                         completed_sequences.append([seq, score, scores_list, h, c])
#                         continue

#                     input_token = torch.tensor([seq[-1]], device=Config.DEVICE)
#                     output, new_h, new_c = self.decoder(input_token, h, c, encoder_outputs)

#                     log_probs = torch.log_softmax(output, dim=1)
#                     topk_probs, topk_ids = log_probs.topk(beam_width)

#                     for i in range(topk_ids.size(1)):
#                         token_id = topk_ids[0][i].item()
#                         prob = topk_probs[0][i].item()

#                         new_seq = seq + [token_id]
#                         new_score = score + prob
#                         new_scores_list = scores_list + [prob]

#                         candidates.append([new_seq, new_score, new_scores_list, new_h, new_c, new_score / len(new_seq)])

#                 if not candidates:
#                     break

#                 sequences = sorted(candidates, key=lambda x: x[5], reverse=True)[:beam_width]
#                 sequences = [[seq, score, scores_list, h, c] for seq, score, scores_list, h, c, _ in sequences]

#             all_sequences = completed_sequences + sequences
#             all_sequences = sorted(all_sequences, key=lambda x: x[1] / len(x[0]), reverse=True)

#             unique_results = []
#             seen_texts = set()

#             for seq, score, scores_list, _, _ in all_sequences:
#                 output_text = ''.join([Config.idx2char[i] for i in seq
#                                        if i not in {Config.char2idx[Config.SOS_TOKEN],
#                                                     Config.char2idx[Config.EOS_TOKEN],
#                                                     Config.char2idx[Config.PAD_TOKEN]}])

#                 if output_text and output_text not in seen_texts:
#                     seen_texts.add(output_text)
#                     unique_results.append([seq, score / len(seq), scores_list, output_text])
#                     if len(unique_results) >= num_results:
#                         break

#             return unique_results


# class ProbabilityAnalyzer:
#     """概率分析工具类"""

#     @staticmethod
#     def probability_entropy(probs):
#         return -np.sum(probs * np.log(probs + 1e-9))

#     @staticmethod
#     def top_k_certainty(probs, k=3):
#         sorted_indices = np.argsort(probs)[-k:]
#         return np.sum(probs[sorted_indices])

#     @staticmethod
#     def probability_contrast(probs):
#         sorted_probs = np.sort(probs)
#         if len(sorted_probs) >= 2:
#             return sorted_probs[-1] - sorted_probs[-2]
#         return 0

#     @staticmethod
#     def normalize_probabilities(probs, temperature=1.0):
#         if temperature == 1.0:
#             return probs
#         log_probs = np.log(probs + 1e-9) / temperature
#         return softmax(log_probs)

#     @staticmethod
#     def analyze_position(probs, idx_to_class):
#         entropy = ProbabilityAnalyzer.probability_entropy(probs)
#         top_indices = np.argsort(probs)[-3:][::-1]
#         top_probs = probs[top_indices]

#         top_chars = []
#         for idx in top_indices:
#             if str(idx) in idx_to_class:
#                 char = idx_to_class[str(idx)]
#             else:
#                 char = str(idx)
#             top_chars.append(char)

#         contrast = ProbabilityAnalyzer.probability_contrast(probs)
#         top3_certainty = ProbabilityAnalyzer.top_k_certainty(probs, 3)

#         return {
#             'top_chars': top_chars,
#             'top_probs': top_probs.tolist(),
#             'entropy': float(entropy),
#             'contrast': float(contrast),
#             'top3_certainty': float(top3_certainty)
#         }


# class MaskGenerator:
#     """增强的掩码生成器"""

#     def __init__(self, mask_token=Config.MASK_TOKEN, max_mask_ratio=0.6, max_masks=15):
#         self.mask_token = mask_token
#         self.max_mask_ratio = max_mask_ratio
#         self.max_masks = max_masks
#         self.templates = {}

#     def _check_mask_ratio(self, mask):
#         if isinstance(mask, list):
#             mask_count = mask.count(self.mask_token)
#             total = len(mask)
#         else:
#             mask_count = mask.count(self.mask_token)
#             total = len(mask)

#         if total == 0:
#             return False

#         mask_ratio = mask_count / total
#         return mask_ratio <= self.max_mask_ratio

#     def generate_masks(self, position_info, base_prediction=None):
#         masks = set()
#         self.templates = {}

#         for threshold in [0.8, 0.6, 0.4]:
#             mask = self._create_confidence_mask(position_info, threshold)
#             mask_str = ''.join(mask)
#             self.templates[f"基础掩码 (阈值={threshold:.1f})"] = mask_str
#             masks.add(mask_str)

#         for threshold in [1.0, 2.0]:
#             mask = self._create_entropy_mask(position_info, threshold)
#             mask_str = ''.join(mask)
#             self.templates[f"熵掩码 (阈值={threshold:.1f})"] = mask_str
#             masks.add(mask_str)

#         for keep_count in [1, len(position_info) // 2, len(position_info) - 1]:
#             if keep_count > 0 and keep_count < len(position_info):
#                 mask = self._create_topn_mask(position_info, keep_count)
#                 mask_str = ''.join(mask)
#                 self.templates[f"Top-{keep_count}掩码"] = mask_str
#                 masks.add(mask_str)

#         if base_prediction and len(base_prediction) == len(position_info):
#             for i in range(len(base_prediction)):
#                 mask = list(base_prediction)
#                 mask[i] = self.mask_token
#                 mask_str = ''.join(mask)
#                 self.templates[f"单字符掩码-{i+1}"] = mask_str
#                 masks.add(mask_str)

#         scored_masks = []
#         for mask_str in masks:
#             if mask_str.count(self.mask_token) > 0 and mask_str.count(self.mask_token) < len(mask_str):
#                 if self._check_mask_ratio(mask_str):
#                     score = self._calculate_mask_quality(mask_str, position_info, base_prediction)
#                     scored_masks.append((mask_str, score))

#         scored_masks.sort(key=lambda x: (x[0].count(self.mask_token), -x[1]))
#         return scored_masks[:self.max_masks]

#     def _create_confidence_mask(self, position_info, threshold):
#         mask = []
#         for pos in position_info:
#             if pos.get('top_probs') and pos['top_probs'][0] >= threshold:
#                 mask.append(pos['top_chars'][0])
#             else:
#                 mask.append(self.mask_token)
#         return mask

#     def _create_entropy_mask(self, position_info, threshold):
#         mask = []
#         for pos in position_info:
#             if pos.get('entropy', float('inf')) <= threshold:
#                 mask.append(pos['top_chars'][0])
#             else:
#                 mask.append(self.mask_token)
#         return mask

#     def _create_topn_mask(self, position_info, n):
#         sorted_pos = sorted(enumerate(position_info),
#                             key=lambda x: x[1].get('top_probs', [0])[0],
#                             reverse=True)

#         mask = [self.mask_token] * len(position_info)
#         for i in range(min(n, len(sorted_pos))):
#             pos_idx, pos_info = sorted_pos[i]
#             mask[pos_idx] = pos_info['top_chars'][0]

#         return mask

#     def _calculate_mask_quality(self, mask, position_info, base_prediction=None):
#         if len(mask) != len(position_info):
#             return 0

#         confidence_sum = 0
#         mask_count = 0
#         base_match_score = 0

#         for i, char in enumerate(mask):
#             if char != self.mask_token:
#                 if base_prediction and i < len(base_prediction) and char == base_prediction[i]:
#                     base_match_score += 1

#                 try:
#                     if position_info[i].get('top_chars') and char in position_info[i]['top_chars']:
#                         char_idx = position_info[i]['top_chars'].index(char)
#                         prob = position_info[i]['top_probs'][char_idx]
#                         confidence_sum += prob
#                     else:
#                         confidence_sum += 0.1
#                     mask_count += 1
#                 except (ValueError, IndexError):
#                     confidence_sum += 0.1
#                     mask_count += 1

#         avg_confidence = confidence_sum / max(1, mask_count)
#         base_match_ratio = base_match_score / max(1, mask_count)
#         mask_ratio = mask_count / len(mask)
#         mask_ratio_score = 0.5 + 0.5 * mask_ratio

#         total_score = 0.4 * avg_confidence + 0.3 * mask_ratio_score + 0.3 * base_match_ratio
#         return total_score


# class EnhancedPredictionSystem:
#     """增强的预测系统"""

#     def __init__(self, config_manager, seq2seq_model_path="seq_best_model.pth", sound_model_dir_override=None):
#         self.config = config_manager
#         self.device = Config.DEVICE

#         self.audio_processor = AudioProcessor(config_manager)
#         self.feature_extractor = FeatureExtractor(config_manager)
#         self.model_trainer = KeystrokeModelTrainer(config_manager)

#         self.probability_analyzer = ProbabilityAnalyzer()
#         self.mask_generator = MaskGenerator(mask_token=Config.MASK_TOKEN, max_mask_ratio=0.6)

#         self.load_sound_models(sound_model_dir_override)
#         self.seq2seq_model = self.load_seq2seq_model(seq2seq_model_path)

#         from keystroke_recognition import KeystrokeRecognitionSystem
#         self.basic_system = KeystrokeRecognitionSystem(config_manager=config_manager)

#         self.stats = {
#             'processed_files': 0,
#             'successful_predictions': 0,
#             'total_accuracy': 0,
#             'correct_chars': 0,
#             'total_chars': 0,
#             'basic_correct_chars': 0,
#             'advanced_correct_chars': 0,
#             'total_improvement': 0,
#             'improved_files': 0,
#             'start_time': time.time()
#         }

#     def load_sound_models(self, sound_model_dir_override=None):
#         model_dir = sound_model_dir_override or self.config.get_path("model_dir")
#         self.sound_models = {}

#         print(f"加载声音识别模型...")

#         if not os.path.exists(model_dir):
#             print(f"警告: 模型目录 {model_dir} 不存在")
#             return

#         try:
#             try:
#                 import keras
#                 cnn_path = os.path.join(model_dir, 'cnn_model.h5')
#                 if os.path.exists(cnn_path):
#                     self.sound_models['cnn'] = keras.models.load_model(cnn_path)
#                     print(f"已加载 CNN 模型")

#                 lstm_path = os.path.join(model_dir, 'lstm_model.h5')
#                 if os.path.exists(lstm_path):
#                     self.sound_models['lstm'] = keras.models.load_model(lstm_path)
#                     print(f"已加载 LSTM 模型")
#             except Exception as e1:
#                 try:
#                     import tensorflow as tf
#                     cnn_path = os.path.join(model_dir, 'cnn_model.h5')
#                     if os.path.exists(cnn_path):
#                         self.sound_models['cnn'] = tf.keras.models.load_model(cnn_path)
#                         print(f"已加载 CNN 模型")

#                     lstm_path = os.path.join(model_dir, 'lstm_model.h5')
#                     if os.path.exists(lstm_path):
#                         self.sound_models['lstm'] = tf.keras.models.load_model(lstm_path)
#                         print(f"已加载 LSTM 模型")
#                 except Exception as e2:
#                     print(f"加载深度学习模型失败: {e2}")

#         except Exception as e:
#             print(f"加载深度学习模型时出错: {e}")

#         class_indices_path = os.path.join(model_dir, 'class_indices.json')
#         if os.path.exists(class_indices_path):
#             try:
#                 with open(class_indices_path, 'r') as f:
#                     self.class_indices = json.load(f)
#                     self.idx_to_class = {v: k for k, v in self.class_indices.items()}
#                 print(f"已加载类别索引映射")
#             except Exception as e:
#                 print(f"加载类别索引映射时出错: {e}")

#         scaler_path = os.path.join(model_dir, 'scaler.pkl')
#         if os.path.exists(scaler_path):
#             try:
#                 with open(scaler_path, 'rb') as f:
#                     self.scaler = pickle.load(f)
#                     self.model_trainer.scaler = self.scaler
#                 print(f"已加载特征缩放器")
#             except Exception as e:
#                 print(f"加载特征缩放器时出错: {e}")

#         print(f"成功加载了 {len(self.sound_models)} 个声音模型")

#     def load_seq2seq_model(self, model_path):
#         print(f"加载Seq2Seq模型: {model_path}")

#         model = Seq2Seq().to(Config.DEVICE)

#         if not os.path.exists(model_path):
#             print(f"警告: Seq2Seq模型文件不存在: {model_path}")
#             alternative_paths = ["seq2seq_model.pth", "models/seq2seq_model.pth", "seq_model.pth"]

#             model_found = False
#             for alt_path in alternative_paths:
#                 if os.path.exists(alt_path):
#                     print(f"找到替代模型文件: {alt_path}")
#                     model_path = alt_path
#                     model_found = True
#                     break

#             if not model_found:
#                 print(f"将创建新的Seq2Seq模型（随机初始化）")
#                 return model

#         try:
#             checkpoint = torch.load(model_path, map_location=self.device)

#             if isinstance(checkpoint, dict):
#                 if 'model_state_dict' in checkpoint:
#                     model.load_state_dict(checkpoint['model_state_dict'])
#                 elif 'state_dict' in checkpoint:
#                     model.load_state_dict(checkpoint['state_dict'])
#                 else:
#                     model.load_state_dict(checkpoint)
#             else:
#                 model.load_state_dict(checkpoint)

#             model.eval()
#             print(f"成功加载Seq2Seq模型")
#             return model
#         except Exception as e:
#             print(f"加载Seq2Seq模型失败: {e}")
#             return model

#     def extract_keystroke_probabilities(self, audio_file_path):
#         try:
#             y, sr = self.audio_processor.load_audio(audio_file_path)

#             expected_length = None
#             filename = os.path.basename(audio_file_path)
#             digit_part = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
#             if digit_part:
#                 expected_length = len(digit_part)
#                 print(f"从文件名猜测的预期按键数量: {expected_length}")

#             segments, segment_times = self.audio_processor.isolate_keystrokes_ensemble(
#                 y, sr, expected_length
#             )

#             if not segments:
#                 print("未检测到按键")
#                 return None, 0

#             position_info = []

#             for i, segment in enumerate(segments):
#                 features = self.feature_extractor.extract_features(segment, sr)
#                 features = features.reshape(1, -1)
#                 features_scaled = self.model_trainer.scaler.transform(features)

#                 model_probs = {}

#                 if 'cnn' in self.sound_models:
#                     try:
#                         features_reshaped = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
#                         cnn_probs = self.sound_models['cnn'].predict(features_reshaped, verbose=0)[0]
#                         model_probs['cnn'] = cnn_probs
#                     except Exception as e:
#                         print(f"CNN模型预测出错: {e}")

#                 if 'lstm' in self.sound_models:
#                     try:
#                         lstm_input = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
#                         lstm_probs = self.sound_models['lstm'].predict(lstm_input, verbose=0)[0]
#                         model_probs['lstm'] = lstm_probs
#                     except Exception as e:
#                         print(f"LSTM模型预测出错: {e}")

#                 if len(model_probs) > 0:
#                     weights = {'cnn': 0.6, 'lstm': 0.4}

#                     num_classes = len(list(model_probs.values())[0])
#                     combined_probs = np.zeros(num_classes)
#                     total_weight = 0

#                     for model_name, probs in model_probs.items():
#                         if model_name in weights:
#                             combined_probs += weights[model_name] * probs
#                             total_weight += weights[model_name]

#                     if total_weight > 0:
#                         combined_probs /= total_weight

#                     probs = combined_probs
#                 else:
#                     num_classes = len(self.idx_to_class) if hasattr(self, 'idx_to_class') else 10
#                     probs = np.ones(num_classes) / num_classes

#                 analyzed_position = self.probability_analyzer.analyze_position(probs, self.idx_to_class)
#                 analyzed_position['position'] = i
#                 position_info.append(analyzed_position)

#             print(f"提取了 {len(segments)} 个按键的概率分布")
#             return position_info, len(segments)

#         except Exception as e:
#             print(f"获取按键概率分布时出错: {e}")
#             traceback.print_exc()
#             return None, 0

#     def _predict_with_single_mask(self, mask, position_info):
#         try:
#             input_ids = [Config.char2idx.get(c, Config.char2idx[Config.MASK_TOKEN]) for c in mask]
#             src = torch.tensor(input_ids).unsqueeze(1).to(self.device)
#             src_len = torch.tensor([len(input_ids)]).to(self.device)

#             beam_result = self.seq2seq_model.beam_decode(src, src_len, beam_width=Config.BEAM_WIDTH)

#             if beam_result:
#                 sequence, score, scores_list = beam_result
#                 output_text = ''.join([Config.idx2char[i] for i in sequence
#                                        if i not in {Config.char2idx[Config.SOS_TOKEN],
#                                                     Config.char2idx[Config.EOS_TOKEN],
#                                                     Config.char2idx[Config.PAD_TOKEN]}])

#                 return {
#                     'text': output_text,
#                     'seq_score': score,
#                 }

#             return None
#         except Exception as e:
#             print(f"处理掩码 {mask} 时出错: {e}")
#             return None

#     def _predict_results_by_mask_count(self, mask, position_info):
#         mask_count = mask.count(Config.MASK_TOKEN)

#         if mask_count == 1:
#             num_results = 8
#         elif mask_count == 2:
#             num_results = 12
#         elif mask_count == 3:
#             num_results = 18
#         else:
#             num_results = max(4, mask_count * 3)

#         input_ids = [Config.char2idx.get(c, Config.char2idx[Config.MASK_TOKEN]) for c in mask]
#         src = torch.tensor(input_ids).unsqueeze(1).to(self.device)
#         src_len = torch.tensor([len(input_ids)]).to(self.device)

#         beam_search_width = min(num_results * 2, Config.BEAM_WIDTH)

#         beam_results = self.seq2seq_model.beam_search_multiple(
#             src, src_len,
#             max_len=Config.MAX_LEN,
#             beam_width=beam_search_width,
#             num_results=num_results
#         )

#         results = []
#         if beam_results:
#             for seq, score, scores_list, output_text in beam_results:
#                 mask_match_score = 0
#                 mask_match_count = 0

#                 for i, (a, b) in enumerate(zip(mask, output_text)):
#                     if a != Config.MASK_TOKEN and i < len(output_text):
#                         mask_match_count += 1
#                         if a == b:
#                             mask_match_score += 1

#                 if mask_match_count > 0:
#                     mask_match_score /= mask_match_count

#                 combined_score = 0.7 * score + 0.3 * mask_match_score

#                 results.append({
#                     'text': output_text,
#                     'mask': mask,
#                     'seq_score': score,
#                     'mask_match_score': mask_match_score,
#                     'combined_score': combined_score
#                 })

#         return results

#     def _evaluate_accuracy(self, expected_sequence, predicted_sequence):
#         if not expected_sequence or not predicted_sequence:
#             return 0.0

#         min_len = min(len(expected_sequence), len(predicted_sequence))
#         if min_len == 0:
#             return 0.0

#         correct_chars = sum(1 for i in range(min_len) if expected_sequence[i] == predicted_sequence[i])
#         return correct_chars / len(expected_sequence)

#     def predict_with_enhanced_masks(self, audio_file_path, top_k=30, verbose=True, compare_basic=True):
#         if verbose:
#             print(f"对文件进行高级预测: {audio_file_path}")

#         basic_prediction = None
#         if compare_basic:
#             try:
#                 print(f"\n正在使用基础声音模型进行预测...")
#                 basic_prediction = self.basic_system.predict_from_file(audio_file_path, verbose=False)
#                 print(f"基础声音模型预测结果: {basic_prediction}")
#             except Exception as e:
#                 print(f"基础声音模型预测失败: {e}")

#         position_info, num_keystrokes = self.extract_keystroke_probabilities(audio_file_path)

#         if position_info is None or num_keystrokes == 0:
#             print("无法提取按键概率，无法进行预测")
#             return {'advanced': [], 'basic': basic_prediction, 'improvement': 0, 'accuracy_stats': {}}

#         scored_masks = self.mask_generator.generate_masks(position_info, base_prediction=basic_prediction)

#         if verbose:
#             print(f"生成了 {len(scored_masks)} 个掩码:")
#             for i, (mask, score) in enumerate(scored_masks[:10]):
#                 template_name = next((name for name, template in self.mask_generator.templates.items()
#                                       if template == mask), f"掩码 {i + 1}")
#                 print(f"  {template_name}: {mask} (得分: {score:.4f})")

#         if not scored_masks:
#             print("未能生成有效掩码，无法进行预测")
#             return {'advanced': [], 'basic': basic_prediction, 'improvement': 0, 'accuracy_stats': {}}

#         all_results = []

#         mask_counter = 0
#         for mask, mask_score in scored_masks:
#             try:
#                 mask_counter += 1
#                 if verbose and mask_counter <= 5:
#                     template_name = next((name for name, template in self.mask_generator.templates.items()
#                                           if template == mask), "未命名掩码")
#                     print(f"\n处理掩码{mask_counter}/{len(scored_masks)}: {template_name} - {mask}")

#                 mask_results = self._predict_results_by_mask_count(mask, position_info)

#                 if mask_results:
#                     for result in mask_results:
#                         result['mask_score'] = mask_score
#                         result['combined_score'] = (
#                                 0.5 * result['seq_score'] +
#                                 0.3 * result['mask_match_score'] +
#                                 0.2 * mask_score
#                         )
#                         result['template_name'] = next(
#                             (name for name, template in self.mask_generator.templates.items()
#                              if template == mask),
#                             f"掩码-{mask_counter}"
#                         )

#                     all_results.extend(mask_results)

#                     if verbose and mask_counter <= 5:
#                         print(f"  生成了 {len(mask_results)} 个预测结果")

#             except Exception as e:
#                 print(f"处理掩码 {mask} 时出错: {e}")

#         all_results = sorted(all_results, key=lambda x: x['combined_score'], reverse=True)

#         seen = set()
#         filtered_results = []
#         for result in all_results:
#             if result['text'] not in seen:
#                 seen.add(result['text'])
#                 filtered_results.append(result)

#         if verbose and filtered_results:
#             print(f"\n预测生成完成! 总共 {len(filtered_results)} 个唯一结果")
#             print("\n最佳10个预测结果:")
#             for i, result in enumerate(filtered_results[:10]):
#                 print(
#                     f"{i + 1}. {result['text']} (得分: {result['combined_score']:.4f}, 模板: {result['template_name']})")

#         accuracy_stats = {
#             'sound_model': 0,
#             'pure_seq2seq': 0,
#             'combined_model': 0,
#             'sound_model_prediction': "",
#             'pure_seq2seq_prediction': "",
#             'combined_model_prediction': "",
#             'best_model': ""
#         }

#         expected_sequence = ''.join(c for c in os.path.splitext(os.path.basename(audio_file_path))[0] if c.isdigit())

#         if expected_sequence:
#             if basic_prediction:
#                 accuracy_stats['sound_model_prediction'] = basic_prediction
#                 accuracy_stats['sound_model'] = self._evaluate_accuracy(expected_sequence, basic_prediction)

#             full_mask = Config.MASK_TOKEN * len(position_info)
#             seq2seq_result = self._predict_with_single_mask(full_mask, position_info)

#             if seq2seq_result:
#                 seq2seq_text = seq2seq_result['text']
#                 accuracy_stats['pure_seq2seq_prediction'] = seq2seq_text
#                 accuracy_stats['pure_seq2seq'] = self._evaluate_accuracy(expected_sequence, seq2seq_text)

#             sound_accuracy = accuracy_stats['sound_model']
#             seq2seq_accuracy = accuracy_stats['pure_seq2seq']
#             advanced_mask_accuracy = 0

#             if filtered_results:
#                 advanced_text = filtered_results[0]['text']
#                 advanced_mask_accuracy = self._evaluate_accuracy(expected_sequence, advanced_text)

#             if sound_accuracy >= seq2seq_accuracy and sound_accuracy >= advanced_mask_accuracy:
#                 accuracy_stats['combined_model_prediction'] = basic_prediction
#                 accuracy_stats['combined_model'] = sound_accuracy
#                 accuracy_stats['best_model'] = "声音模型"
#             elif seq2seq_accuracy >= sound_accuracy and seq2seq_accuracy >= advanced_mask_accuracy:
#                 accuracy_stats['combined_model_prediction'] = seq2seq_text
#                 accuracy_stats['combined_model'] = seq2seq_accuracy
#                 accuracy_stats['best_model'] = "纯Seq2seq"
#             else:
#                 accuracy_stats['combined_model_prediction'] = advanced_text
#                 accuracy_stats['combined_model'] = advanced_mask_accuracy
#                 accuracy_stats['best_model'] = "掩码+Seq2seq"

#             if verbose:
#                 print("\n准确率对比:")
#                 print(f"预期序列: {expected_sequence}")
#                 print(f"声音模型: {basic_prediction} - 准确率: {accuracy_stats['sound_model']:.2%}")
#                 print(f"纯Seq2Seq: {accuracy_stats['pure_seq2seq_prediction']} - 准确率: {accuracy_stats['pure_seq2seq']:.2%}")
#                 print(f"掩码+Seq2Seq: {filtered_results[0]['text'] if filtered_results else 'N/A'} - 准确率: {advanced_mask_accuracy:.2%}")
#                 print(f"最佳模型: {accuracy_stats['best_model']} - 准确率: {accuracy_stats['combined_model']:.2%}")

#         improvement = 0
#         if basic_prediction and filtered_results:
#             if expected_sequence:
#                 basic_accuracy = accuracy_stats['sound_model']
#                 advanced_accuracy = accuracy_stats['combined_model']

#                 if basic_accuracy > 0:
#                     improvement = (advanced_accuracy - basic_accuracy) / basic_accuracy * 100
#                 else:
#                     improvement = float('inf') if advanced_accuracy > 0 else 0

#                 if verbose:
#                     if improvement != float('inf'):
#                         print(f"相对于声音模型的准确率提升: {improvement:.2f}%")
#                     else:
#                         print(f"相对于声音模型的准确率提升: 无限")

#         self.stats['processed_files'] += 1
#         if filtered_results:
#             self.stats['successful_predictions'] += 1

#         return {
#             'advanced': filtered_results[:top_k],
#             'basic': basic_prediction,
#             'improvement': improvement,
#             'accuracy_stats': accuracy_stats
#         }

#     def predict_directory(self, dir_path, top_k=30, verbose=True, save_viz=False):
#         if not os.path.exists(dir_path):
#             print(f"错误: 目录不存在: {dir_path}")
#             return {}

#         if not os.path.isdir(dir_path):
#             print(f"错误: {dir_path} 不是目录")
#             return {}

#         wav_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.wav')]
#         if not wav_files:
#             print(f"错误: 目录 {dir_path} 中没有WAV文件")
#             return {}

#         print(f"在目录 {dir_path} 中找到 {len(wav_files)} 个WAV文件")

#         results_file = os.path.join(dir_path, f"advanced_prediction_results_{time.strftime('%Y%m%d-%H%M%S')}.csv")

#         csv_headers = [
#             "文件名", "预期序列",
#             "声音模型预测", "声音模型准确率",
#             "纯Seq2Seq预测", "纯Seq2Seq准确率",
#             "高级模型预测", "高级模型准确率",
#             "提升率(%)", "最佳高级模板名称", "最佳高级掩码"
#         ]

#         with open(results_file, 'w', encoding='utf-8', newline='') as f:
#             import csv
#             writer = csv.writer(f)
#             writer.writerow(csv_headers)

#         current_run_stats = defaultdict(float)
#         current_run_stats['start_time'] = time.time()
#         current_run_stats['best_model_counts'] = defaultdict(int)

#         all_file_results_map = {}

#         print("\n开始批量预测...")

#         for i, filename in enumerate(wav_files):
#             file_path =            os.path.join(dir_path, filename)
#             print(f"\n[{i + 1}/{len(wav_files)}] 处理文件: {filename}")

#             expected_sequence = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
#             if expected_sequence:
#                 print(f"  从文件名提取的预期序列: '{expected_sequence}'")
#             else:
#                 print(f"  警告: 文件 '{filename}' 未能提取到预期序列。")

#             try:
#                 results_dict = self.predict_with_enhanced_masks(
#                     file_path,
#                     top_k=top_k,
#                     verbose=verbose,
#                     compare_basic=True
#                 )

#                 acc_stats = results_dict.get('accuracy_stats', {})

#                 sound_pred_text = acc_stats.get('sound_model_prediction', "N/A")
#                 sound_acc = float(acc_stats.get('sound_model', 0.0))

#                 pure_s2s_pred_text = acc_stats.get('pure_seq2seq_prediction', "N/A")
#                 pure_s2s_acc = float(acc_stats.get('pure_seq2seq', 0.0))

#                 advanced_pred_text = "N/A"
#                 best_template_name = "N/A"
#                 best_mask_str = "N/A"

#                 if results_dict.get('advanced'):
#                     advanced_pred_text = results_dict['advanced'][0].get('text', 'N/A')
#                     best_template_name = results_dict['advanced'][0].get('template_name', 'N/A')
#                     best_mask_str = results_dict['advanced'][0].get('mask', 'N/A')

#                 improvement_val = results_dict.get('improvement', 0.0)
#                 improvement_csv = "0.00"

#                 if isinstance(improvement_val, (int, float)):
#                     if improvement_val == float('inf'):
#                         improvement_csv = "inf"
#                     elif improvement_val != 0.0:
#                         improvement_csv = f"{improvement_val:.2f}"

#                 print(f"  最佳高级预测: '{advanced_pred_text}' (准确率: {acc_stats.get('combined_model', 0.0):.2%}, 提升: {improvement_csv}%)")

#                 with open(results_file, 'a', encoding='utf-8', newline='') as f:
#                     writer = csv.writer(f)
#                     writer.writerow([
#                         filename, expected_sequence,
#                         sound_pred_text, f"{sound_acc:.4f}",
#                         pure_s2s_pred_text, f"{pure_s2s_acc:.4f}",
#                         advanced_pred_text, f"{acc_stats.get('combined_model', 0.0):.4f}",
#                         improvement_csv,
#                         best_template_name,
#                         best_mask_str
#                     ])

#                 all_file_results_map[filename] = results_dict

#                 if expected_sequence:
#                     current_run_stats['total_files_with_expected_sequence'] += 1
#                     seq_len = len(expected_sequence)
#                     if seq_len > 0:
#                         current_run_stats['total_expected_chars'] += seq_len
#                         current_run_stats['sound_model_total_correct_chars'] += sound_acc * seq_len
#                         current_run_stats['pure_seq2seq_total_correct_chars'] += pure_s2s_acc * seq_len
#                         current_run_stats['advanced_model_total_correct_chars'] += acc_stats.get('combined_model', 0.0) * seq_len

#                         best_model = acc_stats.get('best_model', 'unknown')
#                         current_run_stats['best_model_counts'][best_model] += 1

#                 if save_viz and self.basic_system:
#                     try:
#                         create_comparison_visualization(
#                             file_path,
#                             expected_sequence or "N/A",
#                             sound_pred_text,
#                             advanced_pred_text,
#                             self.basic_system
#                         )
#                         print(f"  对比可视化已保存。")
#                     except Exception as e_viz:
#                         print(f"  可视化出错 for {filename}: {e_viz}")

#             except Exception as e_file_proc:
#                 print(f"处理文件 {filename} 时发生严重错误: {e_file_proc}")
#                 traceback.print_exc()
#                 with open(results_file, 'a', encoding='utf-8', newline='') as f:
#                     writer = csv.writer(f)
#                     error_row = [filename, expected_sequence] + ["ERROR"] * 3 + ["0.0000"] * 3 + ["0.00", "N/A", "N/A"]
#                     writer.writerow(error_row)

#         elapsed_time_run = time.time() - current_run_stats['start_time']
#         print("\n====== 目录预测统计信息 ======")

#         total_valid_files = current_run_stats['total_files_with_expected_sequence']
#         total_exp_chars = current_run_stats['total_expected_chars']

#         print(f"已处理文件总数 (有预期序列的): {int(total_valid_files)}")
#         if total_valid_files == 0:
#             print("没有文件带有可用于统计的预期序列。")
#             return all_file_results_map

#         print(f"总预期字符数: {int(total_exp_chars)}")
#         print(f"处理时间: {elapsed_time_run:.2f}秒 (平均: {elapsed_time_run / max(1, len(wav_files)):.2f}秒/文件)")

#         sm_acc_overall = (current_run_stats['sound_model_total_correct_chars'] / total_exp_chars) if total_exp_chars > 0 else 0
#         ps_acc_overall = (current_run_stats['pure_seq2seq_total_correct_chars'] / total_exp_chars) if total_exp_chars > 0 else 0
#         am_acc_overall = (current_run_stats['advanced_model_total_correct_chars'] / total_exp_chars) if total_exp_chars > 0 else 0

#         print(f"\n声音模型总体准确率: {sm_acc_overall:.2%}")
#         print(f"纯Seq2Seq总体准确率: {ps_acc_overall:.2%}")
#         print(f"高级模型总体准确率: {am_acc_overall:.2%}")

#         if sm_acc_overall > 0 and am_acc_overall > sm_acc_overall:
#             char_improvement_overall = (am_acc_overall - sm_acc_overall) / sm_acc_overall * 100
#             print(f"高级模型相较于声音模型的准确率提升: {char_improvement_overall:.2f}%")
#         elif am_acc_overall > 0 and sm_acc_overall == 0:
#             print(f"高级模型相较于声音模型的准确率提升: ∞ (声音模型准确率为0)")

#         print(f"\n模型胜出统计:")
#         for model_type, count in current_run_stats['best_model_counts'].items():
#             print(f"  {model_type}: {count} 次 ({count / total_valid_files:.2%})")

#         print(f"\n详细结果已保存至: {results_file}")
#         return all_file_results_map


# # 对比可视化函数
# def create_comparison_visualization(file_path, expected, basic, advanced, system):
#     """创建对比可视化"""
#     import matplotlib.pyplot as plt
#     from datetime import datetime
#     import numpy as np

#     try:
#         y, sr = system.audio_processor.load_audio(file_path)

#         segments, segment_times, energy = system.audio_processor.detect_keystrokes(y, sr)

#         plt.figure(figsize=(15, 8))

#         # 1. 显示波形
#         plt.subplot(3, 1, 1)
#         times = np.linspace(0, len(y) / sr, len(y))
#         plt.plot(times, y)
#         plt.title("音频波形")

#         for i, (start, end) in enumerate(segment_times):
#             plt.axvspan(start, end, alpha=0.2, color='red')
#             plt.text((start + end) / 2, plt.ylim()[1] * 0.9, f"{i + 1}",
#                      horizontalalignment='center', fontsize=10)

#         # 2. 显示基础预测
#         plt.subplot(3, 1, 2)
#         plt.plot(times, y)
#         plt.title("基础声音模型预测")

#         for i, ((start, end), pred) in enumerate(zip(segment_times, basic[:len(segment_times)])):
#             plt.axvspan(start, end, alpha=0.2, color='blue')
#             plt.text((start + end) / 2, plt.ylim()[1] * 0.9, pred,
#                      horizontalalignment='center', fontsize=12, color='blue')

#         # 3. 显示高级预测
#         plt.subplot(3, 1, 3)
#         plt.plot(times, y)
#         plt.title("高级模型预测 (掩码+Seq2Seq)")

#         for i, ((start, end), pred) in enumerate(zip(segment_times, advanced[:len(segment_times)])):
#             plt.axvspan(start, end, alpha=0.2, color='green')
#             plt.text((start + end) / 2, plt.ylim()[1] * 0.9, pred,
#                      horizontalalignment='center', fontsize=12, color='green')

#         plt.suptitle(f"预测对比: {'预期: ' + expected if expected else '未知预期序列'}\n"
#                      f"基础预测: {basic} | 高级预测: {advanced}", fontsize=14)

#         plt.tight_layout()

#         results_dir = system.config_manager.get_path("results_dir")
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         save_path = os.path.join(results_dir, f'prediction_comparison_{timestamp}.png')
#         plt.savefig(save_path)
#         plt.close()

#         print(f"对比可视化已保存至: {save_path}")
#     except Exception as e:
#         print(f"创建可视化时出错: {e}")


# def advanced_predict_file(config_manager):
#     """高级按键预测函数（单个文件）"""
#     seq2seq_model_path = "seq_best_model.pth"
#     if not os.path.exists(seq2seq_model_path):
#         print(f"警告: Seq2Seq模型文件不存在: {seq2seq_model_path}")
#         use_seq2seq = input("是否继续高级预测（将使用随机初始化的Seq2Seq模型）? (y/n): ")
#         if use_seq2seq.lower() != 'y':
#             print("高级预测已取消")
#             return False

#     prediction_system = EnhancedPredictionSystem(config_manager, seq2seq_model_path)

#     file_path = input("请输入音频文件路径: ")
#     if not os.path.exists(file_path):
#         print(f"错误: 文件不存在: {file_path}")
#         return False

#     top_k = int(input("返回的最佳结果数量 [默认30]: ") or "30")

#     print("\n开始高级预测...")
#     start_time = time.time()
#     results = prediction_system.predict_with_enhanced_masks(
#         file_path,
#         top_k=top_k,
#         verbose=True,
#         compare_basic=True
#     )
#     elapsed_time = time.time() - start_time

#     if results['advanced']:
#         print(f"\n预测完成! (用时: {elapsed_time:.2f}秒)")

#         expected_sequence = ''.join(c for c in os.path.splitext(os.path.basename(file_path))[0] if c.isdigit())

#         basic_prediction = results['accuracy_stats']['sound_model_prediction'] or ""
#         seq2seq_prediction = results['accuracy_stats']['pure_seq2seq_prediction'] or ""
#         combined_prediction = results['accuracy_stats']['combined_model_prediction'] or ""

#         basic_accuracy = results['accuracy_stats']['sound_model']
#         seq2seq_accuracy = results['accuracy_stats']['pure_seq2seq']
#         combined_accuracy = results['accuracy_stats']['combined_model']

#         print("\n预测结果对比:")
#         if expected_sequence:
#             print(f"预期序列:     {expected_sequence}")
#         print(f"声音模型预测: {basic_prediction}" + (f" (准确率: {basic_accuracy:.2%})" if expected_sequence else ""))
#         print(f"纯Seq2Seq预测: {seq2seq_prediction}" + (
#             f" (准确率: {seq2seq_accuracy:.2%})" if expected_sequence else ""))
#         print(f"组合模型预测: {combined_prediction}" + (
#             f" (准确率: {combined_accuracy:.2%})" if expected_sequence else ""))

#         if expected_sequence:
#             improvement = results.get('improvement', 0)
#             if improvement != float('inf') and improvement != 0:
#                 print(f"相对于声音模型提升: {improvement:.2f}%")
#             elif improvement == float('inf'):
#                 print(f"相对于声音模型提升: 无限")

#         print("\n组合模型预测详情:")
#         for i, result in enumerate(results['advanced'][:10]):
#             print(f"{i + 1}. {result['text']} (综合得分: {result['combined_score']:.4f})")
#             print(f"   模板类型: {result['template_name']}")
#             print(f"   掩码: {result['mask']}")
#             if 'seq_score' in result:
#                 print(f"   序列得分: {result['seq_score']:.4f}, "
#                       f"掩码匹配度: {result['mask_match_score']:.4f}, "
#                       f"掩码质量: {result['mask_score']:.4f}")

#         try:
#             create_comparison_visualization(
#                 file_path,
#                 expected_sequence if expected_sequence else "",
#                 basic_prediction,
#                 combined_prediction,
#                 prediction_system.basic_system
#             )
#         except Exception as e:
#             print(f"可视化结果时出错: {e}")

#         return True
#     else:
#         print("未能生成预测结果")
#         return False


# def advanced_predict_directory(config_manager):
#     """对目录中的所有音频文件进行高级预测"""
#     seq2seq_model_path = "seq_best_model.pth"
#     if not os.path.exists(seq2seq_model_path):
#         print(f"警告: Seq2Seq模型文件不存在: {seq2seq_model_path}")
#         use_seq2seq = input("是否继续高级预测（将使用随机初始化的Seq2Seq模型）? (y/n): ")
#         if use_seq2seq.lower() != 'y':
#             print("高级预测已取消")
#             return False

#     prediction_system = EnhancedPredictionSystem(config_manager, seq2seq_model_path)

#     dir_path = input("请输入音频文件目录路径: ")
#     if not os.path.exists(dir_path):
#         print(f"错误: 目录不存在: {dir_path}")
#         return False

#     if not os.path.isdir(dir_path):
#         print(f"错误: {dir_path} 不是目录")
#         return False

#     top_k = int(input("每个文件返回的最佳结果数量 [默认30]: ") or "30")
#     save_viz = input("是否保存可视化结果 [y/n, 默认n]: ").lower() == 'y'
#     verbose = input("是否显示详细信息 [y/n, 默认n]: ").lower() == 'y'

#     start_time = time.time()
#     results = prediction_system.predict_directory(
#         dir_path,
#         top_k=top_k,
#         verbose=verbose,
#         save_viz=save_viz
#     )
#     elapsed_time = time.time() - start_time

#     print(f"\n整个预测过程用时: {elapsed_time:.2f}秒")
#     return bool(results)


# if __name__ == "__main__":
#     from config_manager import ConfigManager

#     config = ConfigManager()
#     advanced_predict_file(config)

import os
import torch
import numpy as np
# import librosa # 通常在 audio_processing.py 中使用
# import soundfile as sf # 通常在 audio_processing.py 中使用
# from pathlib import Path
import matplotlib.pyplot as plt # 确保已安装
import json
import time
import pickle
from collections import Counter, defaultdict # Counter 在您的版本1中已使用
from scipy.special import softmax
import itertools
import traceback
from datetime import datetime # 用于时间戳和可视化文件名

# 导入必要的模块
from config_manager import ConfigManager
from audio_processing import AudioProcessor
from feature_extraction import FeatureExtractor
from keystroke_model import KeystrokeModelTrainer # 主要用于参考scaler和idx_to_class

# --- Seq2Seq模型配置 (来自您的“稳定Seq版”) ---
class Config:
    """Seq2Seq模型配置类"""
    SOS_TOKEN = "< SOS >" # 您的版本1是 "< SOS >"
    EOS_TOKEN = "<EOS>"
    PAD_TOKEN = "<PAD>"
    MASK_TOKEN = "￥"
    vocab = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN] + [chr(i) for i in range(32, 127)] + [MASK_TOKEN]
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for i, c in enumerate(vocab)}
    VOCAB_SIZE = len(vocab)
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.5
    # BATCH_SIZE = 64 # 主要用于训练
    # TEACHER_FORCING_RATIO = 0.5 # 主要用于训练
    BEAM_WIDTH = 90 # 来自您的版本1 Config
    MAX_LEN = 30    # 来自您的版本1 Config
    # PATIENCE = 5 # 主要用于训练
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Seq2Seq模型组件 (Attention, Encoder, Decoder - 来自您的“稳定Seq版”) ---
class Attention(torch.nn.Module):
    """注意力机制模块"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = torch.nn.Linear(hidden_dim, 1) # 您版本1是Linear(hidden_dim,1)

    def forward(self, hidden, encoder_outputs): # hidden通常是decoder的hidden
        seq_len = encoder_outputs.shape[0]
        # hidden[-1] 是顶层LSTM的最后一个时间步的hidden state
        # [batch_size, dec_hid_dim] -> [1, batch_size, dec_hid_dim] -> [seq_len, batch_size, dec_hid_dim]
        repeated_hidden = hidden[-1].unsqueeze(0).repeat(seq_len, 1, 1)
        energy = torch.tanh(self.attn(torch.cat((repeated_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=0)

class Encoder(torch.nn.Module):
    """Seq2Seq编码器"""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)
        self.lstm = torch.nn.LSTM(Config.EMBED_DIM, Config.HIDDEN_DIM,
                                  Config.NUM_LAYERS, dropout=Config.DROPOUT, bidirectional=False) # 假设单向

    def forward(self, src, src_len):
        embedded = self.embedding(src)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs, hidden, cell

class Decoder(torch.nn.Module):
    """Seq2Seq解码器"""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)
        self.attention = Attention(Config.HIDDEN_DIM)
        self.lstm = torch.nn.LSTM(Config.EMBED_DIM + Config.HIDDEN_DIM, Config.HIDDEN_DIM,
                                  Config.NUM_LAYERS, dropout=Config.DROPOUT)
        self.fc = torch.nn.Linear(Config.HIDDEN_DIM * 2, Config.VOCAB_SIZE) # 您版本1的输出层

    def forward(self, input_token, hidden, cell, encoder_outputs): # input_token 是当前时间步的输入
        input_token = input_token.unsqueeze(0) # [1, batch_size]
        embedded = self.embedding(input_token) # [1, batch_size, embed_dim]
        
        # 注意力计算使用解码器的隐藏状态和编码器的所有输出
        attn_weights = self.attention(hidden, encoder_outputs) # attn_weights: [src_len, batch_size]
        
        # 创建上下文向量
        # encoder_outputs: [src_len, batch_size, enc_hid_dim]
        # attn_weights.unsqueeze(1): [src_len, 1, batch_size] -> permute -> [batch_size, 1, src_len]
        # encoder_outputs.permute(1,0,2): [batch_size, src_len, enc_hid_dim]
        context = torch.bmm(attn_weights.permute(1,0).unsqueeze(1), encoder_outputs.permute(1,0,2))
        context = context.permute(1,0,2) # [1, batch_size, enc_hid_dim]

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (new_hidden, new_cell) = self.lstm(lstm_input, (hidden, cell))
        
        # 预测下一个词 (使用当前LSTM输出和上下文向量)
        prediction = self.fc(torch.cat((output.squeeze(0), context.squeeze(0)), dim=1))
        return prediction, new_hidden, new_cell, attn_weights.permute(1,0) # 返回attn_weights [batch_size, src_len]

class Seq2Seq(torch.nn.Module):
    """Seq2Seq模型"""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder().to(Config.DEVICE)
        self.decoder = Decoder().to(Config.DEVICE)

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5): # 主要用于训练
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        outputs = torch.zeros(trg_len, batch_size, Config.VOCAB_SIZE).to(Config.DEVICE)
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        current_input = trg[0,:] # <SOS> token
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(current_input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            current_input = trg[t] if teacher_force else top1
        return outputs

    # --- beam_decode 和 beam_search_multiple 将使用您“稳定Seq版”中的实现 ---
    # --- 因为您已确认它们能正确输出数字，关键是确保 beam_search_multiple 能返回每一步的log概率列表 ---
    # --- 以下是您“稳定Seq版”中的实现，我只微调了 beam_decode 的返回值以匹配后续使用 ---
    def beam_decode(self, src, src_len, max_len=Config.MAX_LEN, beam_width=Config.BEAM_WIDTH, alpha=0.6):
        """优化的波束搜索解码，带长度惩罚 (来自您的稳定Seq版，返回值微调)"""
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        sequences = [
            [[Config.char2idx[Config.SOS_TOKEN]], 0.0, [0.0], hidden, cell] # 添加初始log_prob为0给SOS
        ]
        completed_sequences = []
        for _ in range(max_len):
            candidates = []
            for seq, score, scores_list, h, c in sequences:
                if seq[-1] == Config.char2idx[Config.EOS_TOKEN]:
                    completed_sequences.append([seq, score, scores_list, h, c])
                    continue
                input_token = torch.tensor([seq[-1]], device=Config.DEVICE)
                output, new_h, new_c, _ = self.decoder(input_token, h, c, encoder_outputs) # _ for attn_weights
                log_probs = torch.log_softmax(output, dim=1)
                topk_probs, topk_ids = log_probs.topk(min(beam_width, log_probs.size(1))) # 确保不超过词表大小
                for i in range(topk_ids.size(1)):
                    token_id = topk_ids[0][i].item()
                    prob = topk_probs[0][i].item()
                    new_seq = seq + [token_id]
                    new_score = score + prob
                    new_scores_list = scores_list + [prob]
                    lp = ((5 + len(new_seq)) ** alpha) / ((5 + 1) ** alpha) if alpha > 0 else 1.0
                    normalized_score = new_score / lp
                    candidates.append([new_seq, new_score, new_scores_list, new_h, new_c, normalized_score])
            if not candidates: break
            sequences = sorted(candidates, key=lambda x: x[5], reverse=True)[:beam_width]
            sequences = [[s, sc, sl, hs, cs] for s, sc, sl, hs, cs, _ in sequences]
        all_sequences = completed_sequences + sequences
        if not all_sequences: return None
        
        # 确保 alpha > 0 才进行长度惩罚排序，否则按原始分数
        if alpha > 0:
            all_sequences = sorted(all_sequences, key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)
        else:
            all_sequences = sorted(all_sequences, key=lambda x: x[1], reverse=True)

        # 返回: 最佳序列的token ID列表, 该序列的归一化得分, 该序列每一步的log概率列表
        best_seq_data = all_sequences[0]
        final_score = best_seq_data[1] / (len(best_seq_data[0])**alpha if alpha > 0 else 1.0)
        return best_seq_data[0], final_score, best_seq_data[2] # seq_ids, normalized_score, list_of_log_probs

    def beam_search_multiple(self, src, src_len, max_len=Config.MAX_LEN, beam_width=Config.BEAM_WIDTH, num_results=10, alpha=0.6): # 添加alpha参数
        """返回多个波束搜索结果 (来自您的稳定Seq版，返回值微调以包含原始分数和log概率列表)"""
        # 此方法基于您版本1的逻辑，但确保返回的每个候选结果中，
        # scores_list 是每一步选择的token的log概率。
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        sequences = [
            ([Config.char2idx[Config.SOS_TOKEN]], 0.0, [0.0], hidden, cell) # seq_ids, raw_score, list_of_log_probs
        ]
        completed_sequences_tuples = [] # (seq_ids, raw_score, list_of_log_probs, normalized_score)

        for _ in range(max_len):
            candidates = []
            if not sequences: break
            active_paths_in_beam = False
            for seq_ids, current_raw_score, current_log_probs_list, h, c in sequences:
                active_paths_in_beam = True
                if seq_ids[-1] == Config.char2idx[Config.EOS_TOKEN]:
                    lp_val = ((5 + len(seq_ids)) ** alpha) / ((5 + 1) ** alpha) if alpha > 0 else 1.0
                    norm_s_val = current_raw_score / lp_val
                    # 避免重复添加完全相同的已完成序列，或更新为更好的分数
                    found_idx = -1
                    for comp_idx, (comp_s_ids, _, _, comp_norm_s) in enumerate(completed_sequences_tuples):
                        if comp_s_ids == seq_ids:
                            if norm_s_val > comp_norm_s: found_idx = comp_idx # Found and better
                            else: found_idx = -2 # Found but not better
                            break
                    if found_idx == -1: completed_sequences_tuples.append((seq_ids, current_raw_score, current_log_probs_list, norm_s_val))
                    elif found_idx >=0: completed_sequences_tuples[found_idx] = (seq_ids, current_raw_score, current_log_probs_list, norm_s_val)
                    continue

                decoder_input = torch.tensor([seq_ids[-1]], device=Config.DEVICE)
                output_probs, new_h, new_c, _ = self.decoder(decoder_input, h, c, encoder_outputs)
                log_probs_step = torch.log_softmax(output_probs.squeeze(0), dim=0)
                
                current_beam_k = min(beam_width, log_probs_step.size(0))
                topk_log_probs, topk_ids = log_probs_step.topk(current_beam_k)

                for k_idx in range(topk_ids.size(0)):
                    token_id = topk_ids[k_idx].item()
                    log_prob_of_token = topk_log_probs[k_idx].item()
                    new_seq_ids_cand = seq_ids + [token_id]
                    new_raw_score_cand = current_raw_score + log_prob_of_token
                    new_log_probs_list_cand = current_log_probs_list + [log_prob_of_token]
                    lp_step_val = ((5 + len(new_seq_ids_cand)) ** alpha) / ((5 + 1) ** alpha) if alpha > 0 else 1.0
                    norm_score_step_val = new_raw_score_cand / lp_step_val
                    candidates.append((new_seq_ids_cand, new_raw_score_cand, new_log_probs_list_cand, new_h, new_c, norm_score_step_val))
            
            if not active_paths_in_beam and not candidates: break
            if not candidates and (not sequences or all(s[0][-1] == Config.char2idx[Config.EOS_TOKEN] for s in sequences)): break
                
            candidates.sort(key=lambda x: x[5], reverse=True) # Sort by normalized_score_step
            sequences = [(s, rs, slp, nh, nc) for s, rs, slp, nh, nc, _ in candidates[:beam_width]] # Update beam

            if sequences and all(s[0][-1] == Config.char2idx[Config.EOS_TOKEN] for s in sequences): # All in beam are complete
                for s_ids_f, rs_f, sl_f, _, _ in sequences:
                    lp_f_val = ((5 + len(s_ids_f)) ** alpha) / ((5 + 1) ** alpha) if alpha > 0 else 1.0
                    ns_f_val = rs_f / lp_f_val
                    is_p_idx = -1
                    for i_cs_f, (cs_ids_f, _, _, cs_norm_f) in enumerate(completed_sequences_tuples):
                        if cs_ids_f == s_ids_f:
                            if ns_f_val > cs_norm_f: is_p_idx = i_cs_f
                            else: is_p_idx = -2; break
                    if is_p_idx == -1: completed_sequences_tuples.append((s_ids_f, rs_f, sl_f, ns_f_val))
                    elif is_p_idx >=0: completed_sequences_tuples[is_p_idx] = (s_ids_f, rs_f, sl_f, ns_f_val)
                break

        for s_ids, rs, sl, _, _ in sequences: # Add any unfinished from beam
            if s_ids[-1] != Config.char2idx[Config.EOS_TOKEN]:
                lp_val = ((5 + len(s_ids)) ** alpha) / ((5 + 1) ** alpha) if alpha > 0 else 1.0
                ns_val = rs / lp_val
                is_p_idx = -1
                for i_cs_val, (cs_ids_val, _, _, cs_norm_val) in enumerate(completed_sequences_tuples):
                    if cs_ids_val == s_ids:
                        if ns_val > cs_norm_val: is_p_idx = i_cs_val
                        else: is_p_idx = -2; break
                if is_p_idx == -1: completed_sequences_tuples.append((s_ids, rs, sl, ns_val))
                elif is_p_idx >=0: completed_sequences_tuples[is_p_idx] = (s_ids, rs, sl, ns_val)

        if not completed_sequences_tuples: return []
        completed_sequences_tuples.sort(key=lambda x: x[3], reverse=True) # Sort by final normalized_score

        final_unique_results = []
        texts_seen = set()
        for seq_ids_res, raw_score_res, log_probs_list_res, norm_score_res in completed_sequences_tuples:
            text_res = ''.join([Config.idx2char[token_id] for token_id in seq_ids_res 
                                if token_id not in (Config.char2idx[Config.SOS_TOKEN], 
                                                    Config.char2idx[Config.EOS_TOKEN], 
                                                    Config.char2idx[Config.PAD_TOKEN])])
            if text_res and text_res not in texts_seen:
                texts_seen.add(text_res)
                # 返回: token ID列表, 归一化得分, 每步的log概率列表, 文本, 原始总log概率
                final_unique_results.append([seq_ids_res, norm_score_res, log_probs_list_res, text_res, raw_score_res])
                if len(final_unique_results) >= num_results:
                    break
        return final_unique_results

# --- ProbabilityAnalyzer (与您版本1或我改进版一致，确保鲁棒性) ---
class ProbabilityAnalyzer: # (代码与上一条回复中的版本一致，此处省略以节约篇幅)
    @staticmethod
    def probability_entropy(probs):
        probs_np = np.array(probs)
        if probs_np.size == 0: return np.log(Config.VOCAB_SIZE) 
        return -np.sum(probs_np * np.log(probs_np + 1e-9))
    @staticmethod
    def top_k_certainty(probs, k=3):
        probs_np = np.array(probs)
        if probs_np.size == 0: return 0.0
        k = min(k, len(probs_np));
        if k == 0: return 0.0
        return np.sum(np.sort(probs_np)[-k:])
    @staticmethod
    def probability_contrast(probs):
        probs_np = np.array(probs)
        if len(probs_np) < 2: return 0.0
        s_probs = np.sort(probs_np)
        return s_probs[-1] - s_probs[-2]
    @staticmethod
    def analyze_position(probs, idx_to_class_map):
        if not isinstance(probs, np.ndarray): probs = np.array(probs)
        if probs.size == 0:
            return {'top_chars': [], 'top_probs': [], 
                    'entropy': np.log(len(idx_to_class_map) if idx_to_class_map else Config.VOCAB_SIZE), 
                    'contrast': 0.0, 'top3_certainty': 0.0}
        entropy_val = ProbabilityAnalyzer.probability_entropy(probs)
        num_top = min(3, len(probs))
        top_indices_val = np.argsort(probs)[-num_top:][::-1] if num_top > 0 else np.array([])
        top_probs_val = probs[top_indices_val].tolist() if top_indices_val.size > 0 else []
        top_chars_val = [idx_to_class_map.get(str(idx), str(idx)) for idx in top_indices_val] if idx_to_class_map else [str(idx) for idx in top_indices_val]
        contrast_val = ProbabilityAnalyzer.probability_contrast(probs)
        top3_certainty_val = ProbabilityAnalyzer.top_k_certainty(probs, 3)
        return {'top_chars': top_chars_val, 'top_probs': top_probs_val, 
                'entropy': float(entropy_val), 'contrast': float(contrast_val), 
                'top3_certainty': float(top3_certainty_val)}


# --- MaskGenerator (新增单字符掩码策略，并调整数量控制) ---
class MaskGenerator: # (代码与上一条回复中的版本一致，此处省略以节约篇幅)
    def __init__(self, mask_token=Config.MASK_TOKEN, max_mask_ratio=0.6):
        self.mask_token = mask_token
        self.max_mask_ratio = max_mask_ratio
        self.config = None # 将由 EnhancedPredictionSystem 设置

    def _check_mask_ratio(self, mask_list_or_str):
        mask_s = "".join(mask_list_or_str) if isinstance(mask_list_or_str, list) else mask_list_or_str
        if not mask_s: return True
        mask_c = mask_s.count(self.mask_token)
        if len(mask_s) == 0: return True
        return (mask_c / len(mask_s)) <= self.max_mask_ratio
    
    def _get_char_from_pos(self, pos_data, choice_idx=0):
        if pos_data and pos_data.get('top_chars') and choice_idx < len(pos_data['top_chars']):
            return pos_data['top_chars'][choice_idx]
        return self.mask_token

    def _create_confidence_mask(self, pos_info, threshold):
        return [self._get_char_from_pos(p) if p.get('top_probs') and p['top_probs'] and p['top_probs'][0] >= threshold else self.mask_token for p in pos_info]
    
    def _create_entropy_mask(self, pos_info, threshold):
        return [self._get_char_from_pos(p) if p.get('entropy', float('inf')) <= threshold else self.mask_token for p in pos_info]

    def _create_topn_mask(self, pos_info, n_to_keep):
        """修复：确保至少有一些掩码标记"""
        if not pos_info: return []
        
        # === 修复：确保不会生成完全无掩码的模板 ===
        max_keep = max(1, len(pos_info) - 1)  # 至少保留一个掩码位置
        n_to_keep = min(n_to_keep, max_keep)
        
        sorted_indices = sorted(range(len(pos_info)), 
                                key=lambda i: pos_info[i].get('top_probs',[0.0])[0] if pos_info[i].get('top_probs') else 0.0, 
                                reverse=True)
        mask = [self.mask_token] * len(pos_info)
        for i in range(n_to_keep):
            if i < len(sorted_indices):
                idx = sorted_indices[i]
                mask[idx] = self._get_char_from_pos(pos_info[idx])
        return mask
    
    def _create_single_char_masked_variants(self, base_sequence_str, actual_length):
        """修复：处理长度不匹配的情况"""
        if not base_sequence_str: return []
        
        # === 修复：处理长度不匹配 ===
        if len(base_sequence_str) != actual_length:
            if len(base_sequence_str) > actual_length:
                # 截断到实际长度
                base_sequence_str = base_sequence_str[:actual_length]
                print(f"  警告：声音预测长度({len(base_sequence_str)})超过检测长度({actual_length})，已截断")
            else:
                # 用最后一个字符填充
                last_char = base_sequence_str[-1] if base_sequence_str else '0'
                base_sequence_str = base_sequence_str + last_char * (actual_length - len(base_sequence_str))
                print(f"  警告：声音预测长度({len(base_sequence_str)})少于检测长度({actual_length})，已填充")
        
        masked_variants = []
        sequence_list = list(base_sequence_str)
        for i in range(len(sequence_list)):
            original_char = sequence_list[i]
            sequence_list[i] = self.mask_token
            masked_variants.append("".join(sequence_list))
            sequence_list[i] = original_char 
        return masked_variants
    
    def _calculate_mask_quality(self, mask_s, pos_info, base_pred=None):
        if len(mask_s) != len(pos_info) or not pos_info: return 0.0
        conf_sum, unmasked_c, base_match_c = 0.0, 0, 0
        for i, char_m in enumerate(mask_s):
            if char_m != self.mask_token:
                unmasked_c += 1
                if base_pred and i < len(base_pred) and char_m == base_pred[i]: base_match_c += 1
                prob_c = 0.01 
                pi_entry = pos_info[i]
                if pi_entry.get('top_chars') and char_m in pi_entry['top_chars']:
                    try: 
                        idx = pi_entry['top_chars'].index(char_m)
                        if idx < len(pi_entry.get('top_probs',[])):
                             prob_c = pi_entry['top_probs'][idx]
                    except (ValueError, IndexError): pass
                conf_sum += prob_c
        avg_c = conf_sum / unmasked_c if unmasked_c > 0 else 0.0
        base_match_r = base_match_c / unmasked_c if unmasked_c > 0 else 0.0
        unmasked_r = unmasked_c / len(mask_s) if len(mask_s) > 0 else 0.0
        
        w_avg_c = self.config.get("mask_scoring.weight_avg_confidence", 0.4) if hasattr(self, 'config') and self.config else 0.4
        w_base_match = self.config.get("mask_scoring.weight_base_match", 0.4) if hasattr(self, 'config') and self.config else 0.4
        w_unmasked_r = self.config.get("mask_scoring.weight_unmasked_ratio", 0.2) if hasattr(self, 'config') and self.config else 0.2
        return w_avg_c * avg_c + w_base_match * base_match_r + w_unmasked_r * unmasked_r

    def generate_masks(self, position_info, base_prediction=None, verbose=False):
        if not position_info: return []
        all_gen_masks_set = set() 

        # 基础掩码
        for thr in [0.75, 0.5, 0.3]: 
            all_gen_masks_set.add("".join(self._create_confidence_mask(position_info, thr)))
        for thr in [1.0, 2.0]: 
            all_gen_masks_set.add("".join(self._create_entropy_mask(position_info, thr)))
        
        len_pi = len(position_info)
        if len_pi > 0:
            # === 修复：TopN掩码确保有掩码标记 ===
            key_counts_for_topn = [1]
            if len_pi > 2: key_counts_for_topn.append(max(1, len_pi // 2))
            if len_pi > 3: key_counts_for_topn.append(len_pi - 1)
            
            for kc in sorted(list(set(key_counts_for_topn))):
                if kc < len_pi:  # 确保总有掩码
                    all_gen_masks_set.add("".join(self._create_topn_mask(position_info, kc)))
        
        # === 修复：单字符掩码处理长度不匹配 ===
        if base_prediction:
            single_char_vars = self._create_single_char_masked_variants(base_prediction, len_pi)
            if verbose and single_char_vars: 
                print(f"          为基础 '{base_prediction}' 生成了 {len(single_char_vars)} 个单字符掩码变体。")
            for m_var in single_char_vars: 
                all_gen_masks_set.add(m_var)
        
        # 添加全掩码作为备用
        if len_pi > 0:
            all_gen_masks_set.add(self.mask_token * len_pi)

        final_scored_masks = []
        for mask_s_val in all_gen_masks_set:
            # === 修复：过滤完全无掩码的模板 ===
            mask_count = mask_s_val.count(self.mask_token)
            if mask_count > 0 and mask_count < len(mask_s_val):  # 必须有掩码，但不能全是掩码
                if self._check_mask_ratio(mask_s_val):
                    score_val = self._calculate_mask_quality(mask_s_val, position_info, base_prediction)
                    final_scored_masks.append((mask_s_val, score_val))
        
        # 如果没有有效掩码，至少返回一个全掩码
        if not final_scored_masks and len_pi > 0:
            full_mask = self.mask_token * len_pi
            score_val = self._calculate_mask_quality(full_mask, position_info, base_prediction)
            final_scored_masks.append((full_mask, score_val))
        
        final_scored_masks.sort(key=lambda x: (x[0].count(self.mask_token), -x[1]))
        return final_scored_masks

# --- EnhancedPredictionSystem ---
class EnhancedPredictionSystem:
    # ... (您的 __init__, load_eps_sound_models_and_components, load_seq2seq_model, 
    #      extract_keystroke_probabilities, _evaluate_accuracy 方法基本保持不变，
    #      但请确保 extract_keystroke_probabilities 的 verbose 打印逻辑正确，
    #      并使用 self.eps_scaler, self.eps_idx_to_class 等EPS自有组件)

    # 在 __init__ 中确保从配置加载参数
    def __init__(self, config_manager, seq2seq_model_path="seq_best_model.pth", sound_model_dir_override=None):
        self.config = config_manager
        self.device = Config.DEVICE
        self.sound_model_dir_override = sound_model_dir_override

        self.audio_processor = AudioProcessor(config_manager)
        self.feature_extractor = FeatureExtractor(config_manager)
        
        # === 修复1: 确保basic_system使用相同的模型目录 ===
        if sound_model_dir_override:
            # 临时修改配置，让basic_system也使用相同的模型目录
            original_model_dir = config_manager.get("paths.model_dir")
            config_manager.set("paths.model_dir", sound_model_dir_override)
            print(f"EPS: 为保持一致性，basic_system也将使用模型目录: {sound_model_dir_override}")
        
        # 确保basic_system初始化完毕后再恢复配置
        from keystroke_recognition import KeystrokeRecognitionSystem 
        self.basic_system = KeystrokeRecognitionSystem(config_manager=config_manager)
        
        # 恢复原始配置
        if sound_model_dir_override:
            config_manager.set("paths.model_dir", original_model_dir)
        
        # EPS自有模型加载
        self.eps_sound_models = {} 
        self.eps_scaler = None
        self.eps_class_indices = None
        self.eps_idx_to_class = None
        self.load_eps_sound_models_and_components()

        self.seq2seq_model = self.load_seq2seq_model(seq2seq_model_path)
        
        self.probability_analyzer = ProbabilityAnalyzer()
        max_mask_ratio_cfg = self.config.get("mask_generation.max_mask_ratio", 0.7)
        self.mask_generator = MaskGenerator(mask_token=Config.MASK_TOKEN, max_mask_ratio=max_mask_ratio_cfg)
        self.mask_generator.config = self.config

        self.current_run_stats = defaultdict(float)
        self.current_run_stats['best_model_type_counts'] = defaultdict(int)



    # load_eps_sound_models_and_components (与上次提供的版本一致)
    def load_eps_sound_models_and_components(self):
        model_dir_to_use = self.sound_model_dir_override if self.sound_model_dir_override else self.config.get_path("model_dir")
        print(f"EPS: 尝试从目录 '{model_dir_to_use}' 加载自有声音模型及组件...")
        if not os.path.exists(model_dir_to_use) or not os.path.isdir(model_dir_to_use):
            print(f"  EPS警告: 模型目录 '{model_dir_to_use}' 不存在或无效。EPS将不加载自有声音模型。")
            if self.basic_system and hasattr(self.basic_system, 'scaler') and self.basic_system.scaler:
                print("    INFO: EPS 将尝试使用 basic_system 的 scaler 作为备用。")
                self.eps_scaler = self.basic_system.scaler
            if self.basic_system and hasattr(self.basic_system, 'class_indices') and self.basic_system.class_indices:
                print("    INFO: EPS 将尝试使用 basic_system 的 class_indices 作为备用。")
                self.eps_class_indices = self.basic_system.class_indices
                self.eps_idx_to_class = {v: k for k, v in self.eps_class_indices.items()} if self.eps_class_indices else None
            return

        loaded_eps_models_count = 0
        for model_key, model_filename in [('cnn', 'cnn_model.h5'), ('lstm', 'lstm_model.h5')]:
            model_path = os.path.join(model_dir_to_use, model_filename)
            if os.path.exists(model_path):
                try:
                    import keras
                    self.eps_sound_models[model_key] = keras.models.load_model(model_path)
                    print(f"  EPS: 已使用 keras 加载 {model_key.upper()} 模型 from {model_path}")
                    loaded_eps_models_count +=1
                except ImportError: 
                    try:
                        import tensorflow as tf
                        self.eps_sound_models[model_key] = tf.keras.models.load_model(model_path)
                        print(f"  EPS: 已使用 tensorflow.keras 加载 {model_key.upper()} 模型 from {model_path}")
                        loaded_eps_models_count +=1
                    except ImportError: print(f"  EPS警告: keras 和 tensorflow.keras 均未找到，无法加载 {model_key.upper()} 模型。")
                    except Exception as e_tf: print(f"  EPS: 使用 tensorflow.keras 加载 {model_key.upper()} 模型失败 from {model_path}: {e_tf}")
                except Exception as e_k: print(f"  EPS: 使用 keras 加载 {model_key.upper()} 模型失败 from {model_path}: {e_k}")
        
        class_indices_path = os.path.join(model_dir_to_use, 'class_indices.json')
        if os.path.exists(class_indices_path):
            try:
                with open(class_indices_path, 'r') as f: self.eps_class_indices = json.load(f)
                if self.eps_class_indices: self.eps_idx_to_class = {v: k for k, v in self.eps_class_indices.items()}
                print(f"  EPS: 已加载类别索引映射 from {class_indices_path}")
            except Exception as e: print(f"  EPS: 加载类别索引映射失败 from {class_indices_path}: {e}")
        elif not self.eps_idx_to_class and self.basic_system and hasattr(self.basic_system, 'idx_to_class') and self.basic_system.idx_to_class :
            print(f"  EPS警告: 类别索引文件 '{class_indices_path}' 不存在。尝试使用basic_system的。")
            self.eps_idx_to_class = self.basic_system.idx_to_class

        scaler_path = os.path.join(model_dir_to_use, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f: self.eps_scaler = pickle.load(f)
                print(f"  EPS: 已加载特征缩放器 from {scaler_path}")
            except Exception as e: print(f"  EPS: 加载特征缩放器失败 from {scaler_path}: {e}")
        elif not self.eps_scaler and self.basic_system and hasattr(self.basic_system, 'scaler') and self.basic_system.scaler :
             print(f"  EPS警告: 特征缩放器文件 '{scaler_path}' 不存在。尝试使用basic_system的。")
             self.eps_scaler = self.basic_system.scaler
        
        if loaded_eps_models_count == 0: print(f"  EPS: 未加载任何自有声音模型。")
        if not self.eps_scaler: print(f"  EPS警告: 未能加载EPS scaler，特征提取可能依赖basic_system的scaler。")
        if not self.eps_idx_to_class: print(f"  EPS警告: 未能加载EPS idx_to_class，概率分析可能依赖basic_system的映射。")

    # load_seq2seq_model (与上次代码一致)
    def load_seq2seq_model(self, model_path):
        print(f"加载Seq2Seq模型: {model_path}")
        model = Seq2Seq().to(Config.DEVICE) 
        if not os.path.exists(model_path):
            print(f"警告: Seq2Seq模型文件 '{model_path}' 不存在. 将使用随机初始化的Seq2Seq模型.")
            return model
        try:
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            model.eval() 
            print(f"成功加载Seq2Seq模型 from {model_path}")
        except Exception as e:
            print(f"加载Seq2Seq模型失败 from {model_path}: {e}. 将使用随机初始化的Seq2Seq模型.")
        return model

    # extract_keystroke_probabilities (已包含修正后的版本)
    def extract_keystroke_probabilities(self, audio_file_path, verbose=True):
        try:
            y, sr = self.audio_processor.load_audio(audio_file_path)
            expected_length = None 
            filename = os.path.basename(audio_file_path)
            digit_part = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
            if digit_part: 
                expected_length = len(digit_part)

            segments, _ = self.audio_processor.isolate_keystrokes_ensemble(y, sr, expected_length)
            actual_num_keystrokes = len(segments) 

            if not segments: 
                if verbose: print(f"  EPS: 音频 {filename} 未检测到按键段。")
                return None, 0

            position_info = []
            current_scaler = self.eps_scaler
            current_idx_to_class = self.eps_idx_to_class

            if not current_scaler:
                if self.basic_system and hasattr(self.basic_system, 'scaler') and self.basic_system.scaler:
                    current_scaler = self.basic_system.scaler
                    if verbose: print(f"  EPS Info: extract_keystroke_probabilities 使用 basic_system 的 scaler。")
                else:
                    if verbose: print(f"  EPS错误: 无法找到有效的 scaler。无法提取概率。")
                    return None, 0
            if not current_idx_to_class:
                if self.basic_system and hasattr(self.basic_system, 'idx_to_class') and self.basic_system.idx_to_class:
                    current_idx_to_class = self.basic_system.idx_to_class
                    if verbose: print(f"  EPS Info: extract_keystroke_probabilities 使用 basic_system 的 idx_to_class。")
                else:
                    if verbose: print(f"  EPS错误: 无法找到有效的 idx_to_class。无法映射预测。")
                    return None, 0
            
            for i_seg, segment_audio_data in enumerate(segments):
                features = self.feature_extractor.extract_features(segment_audio_data, sr)
                features_scaled = current_scaler.transform(features.reshape(1, -1))
                
                eps_model_probs_list = []
                if 'cnn' in self.eps_sound_models and self.eps_sound_models['cnn']:
                    cnn_in = features_scaled.reshape(1, features_scaled.shape[1], 1)
                    eps_model_probs_list.append(self.eps_sound_models['cnn'].predict(cnn_in, verbose=0)[0])
                if 'lstm' in self.eps_sound_models and self.eps_sound_models['lstm']:
                    lstm_in = features_scaled.reshape(1, features_scaled.shape[1], 1)
                    eps_model_probs_list.append(self.eps_sound_models['lstm'].predict(lstm_in, verbose=0)[0])

                if eps_model_probs_list:
                    probs_for_pos = np.mean(eps_model_probs_list, axis=0)
                else: 
                    num_classes = len(current_idx_to_class)
                    probs_for_pos = np.ones(num_classes) / num_classes if num_classes > 0 else np.array([])
                    if verbose and actual_num_keystrokes > 0 : # 只有在实际检测到按键时才打印此警告
                        print(f"    警告: 位置 {i_seg}，EPS没有自有模型输出概率，使用均匀分布。")
                
                analyzed_pos_data = self.probability_analyzer.analyze_position(probs_for_pos, current_idx_to_class)
                position_info.append({'position': i_seg, **analyzed_pos_data})
            
            if verbose:
                if actual_num_keystrokes > 0:
                    if expected_length is not None and expected_length == actual_num_keystrokes:
                        print(f"  EPS: 为 {filename} 提取了 {actual_num_keystrokes} 个按键的概率分布 (与预期长度一致)。")
                    elif expected_length is not None:
                        print(f"  EPS警告: 为 {filename} 提取了 {actual_num_keystrokes} 个按键的概率分布, 但预期是 {expected_length}。")
                    else:
                        print(f"  EPS: 为 {filename} 提取了 {actual_num_keystrokes} 个按键的概率分布 (无预期长度信息)。")
            
            return position_info, actual_num_keystrokes
        except Exception as e_ext:
            if verbose: 
                print(f"  EPS: 获取按键概率分布时出错 for {audio_file_path}: {e_ext}")
                traceback.print_exc()
            return None, 0
            
    # _predict_with_single_mask (与我上次提供的代码一致)
    def _predict_with_single_mask(self, mask_str, num_keystrokes_fallback):
        try:
            if not mask_str and num_keystrokes_fallback > 0: mask_str = Config.MASK_TOKEN * num_keystrokes_fallback
            input_ids = [Config.char2idx.get(c, Config.char2idx[Config.MASK_TOKEN]) for c in mask_str]
            if not input_ids: return None 
            src = torch.tensor(input_ids, device=Config.DEVICE).unsqueeze(1); src_len = torch.tensor([len(input_ids)], device=Config.DEVICE)
            beam_res_tuple = self.seq2seq_model.beam_decode(src, src_len, beam_width=Config.BEAM_WIDTH, alpha=0.6)
            if beam_res_tuple:
                seq_ids, norm_s, token_log_probs = beam_res_tuple # 确保beam_decode返回这三个
                out_text = ''.join([Config.idx2char[i] for i in seq_ids if i not in {Config.char2idx[c] for c in [Config.SOS_TOKEN, Config.EOS_TOKEN, Config.PAD_TOKEN]}])
                return {'text': out_text, 'seq_score': norm_s, 'token_log_probs': token_log_probs}
            return None
        except Exception as e_single_mask:
            print(f"    处理掩码 '{mask_str}' 进行单一预测时出错: {e_single_mask}")
            return None

    # _predict_results_by_mask_count (确保解析 beam_search_multiple 返回的5元素列表)
    def _predict_results_by_mask_count(self, mask_str, num_keystrokes_fallback):
        mask_c = mask_str.count(Config.MASK_TOKEN)
        if not mask_str and num_keystrokes_fallback > 0: 
            mask_str = Config.MASK_TOKEN * num_keystrokes_fallback
            mask_c = num_keystrokes_fallback
        
        if mask_c == 0 and len(mask_str) > 0: num_s2s_res, s2s_bm = 1, 1
        elif mask_c == 1: num_s2s_res, s2s_bm = self.config.get("seq2seq_prediction.results_per_mask_1", 5), 2
        elif mask_c == 2: num_s2s_res, s2s_bm = self.config.get("seq2seq_prediction.results_per_mask_2", 15), 2
        elif mask_c == 3: num_s2s_res, s2s_bm = self.config.get("seq2seq_prediction.results_per_mask_3", 30), 2
        else: num_s2s_res, s2s_bm = max(8, mask_c * self.config.get("seq2seq_prediction.results_per_mask_gt3_multiplier", 3)), 2

        results_list = []
        input_ids = [Config.char2idx.get(c, Config.char2idx[Config.MASK_TOKEN]) for c in mask_str]
        if not input_ids: return []
        src = torch.tensor(input_ids, device=Config.DEVICE).unsqueeze(1)
        src_len = torch.tensor([len(input_ids)], device=Config.DEVICE)
        
        beam_w_search = min(num_s2s_res * s2s_bm, Config.BEAM_WIDTH * 2) 
        beam_w_search = max(1, beam_w_search)
        
        # beam_search_multiple 返回: [seq_ids, normalized_score, scores_list_logprobs, output_text_str, raw_score]
        s2s_outputs_raw = self.seq2seq_model.beam_search_multiple(
            src, src_len, 
            max_len=Config.MAX_LEN, 
            beam_width=beam_w_search, 
            num_results=num_s2s_res, 
            alpha=0.6 # 确保传递alpha
        )

        if s2s_outputs_raw:
            for s2s_cand_data in s2s_outputs_raw:
                if len(s2s_cand_data) == 5: # 严格检查5个元素
                    seq_ids_val, norm_s_val, s_list_logprobs_val, out_text_val, raw_s_val = s2s_cand_data
                    
                    match_ch, unmasked_ch_count = 0, 0
                    for i_char, orig_char_m in enumerate(mask_str):
                        if orig_char_m != Config.MASK_TOKEN:
                            unmasked_ch_count += 1
                            if i_char < len(out_text_val) and orig_char_m == out_text_val[i_char]:
                                match_ch += 1
                    adherence_score = match_ch / unmasked_ch_count if unmasked_ch_count > 0 else 1.0
                    
                    results_list.append({
                        'text': out_text_val, 
                        'mask': mask_str, 
                        'seq_score': norm_s_val, 
                        'raw_seq_score': raw_s_val, 
                        'mask_adherence_score': adherence_score,
                        'token_log_probs': s_list_logprobs_val 
                    })
                else:
                    # 仅在 verbose 模式下打印此警告，避免过多输出
                    if hasattr(self,'config') and self.config.get("system.verbose", False): # 假设有system.verbose配置
                         print(f"警告: _predict_results_by_mask_count - beam_search_multiple 返回的候选格式不符合预期的5个元素: {s2s_cand_data}")
        return results_list

    # _evaluate_accuracy (与上次代码一致)
    def _evaluate_accuracy(self, expected, predicted):
        if not expected or not predicted: return 0.0
        min_l = min(len(expected), len(predicted))
        if min_l == 0 and len(expected) > 0 : return 0.0
        if min_l == 0 and len(expected) == 0: return 1.0
        corr_chars = sum(1 for i in range(min_l) if expected[i] == predicted[i])
        return corr_chars / len(expected) if len(expected) > 0 else 0.0

    # predict_with_enhanced_masks (与上次代码一致，它已经整合了所有逻辑)
    # def predict_with_enhanced_masks(self, audio_file_path, top_k=10, verbose=True, compare_basic=True):
    #     # (代码与上一条回复中的 predict_with_enhanced_masks 完整函数一致，此处省略以节约篇幅)
    #     # --- 参数设定 ---
    #     N_SOUND_CANDIDATES_FOR_MASKING = self.config.get("advanced_prediction.n_sound_candidates_for_masking", 3)
    #     MAX_MASKS_PER_SOUND_CANDIDATE = self.config.get("advanced_prediction.max_masks_per_base", 8)

    #     expected_sequence_str_log = ''.join(c for c in os.path.splitext(os.path.basename(audio_file_path))[0] if c.isdigit())
        
    #     if verbose:
    #         print("\n" + "="*80)
    #         print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始处理文件: {os.path.basename(audio_file_path)}")
    #         if expected_sequence_str_log: print(f"预期序列: '{expected_sequence_str_log}'")
    #         else: print("预期序列: N/A (无法从文件名提取)")
    #         print("="*80 + "\n")

    #     # 1. 基础声音模型候选
    #     sound_model_candidates_list = [("", 0.0)] 
    #     basic_top_sound_prediction = "" 
    #     if compare_basic and self.basic_system:
    #         try:
    #             if verbose: print(f"1. 基础声音模型候选 (最多 Top 10):")
    #             sound_model_candidates_list = self.basic_system.predict_from_file_with_candidates(audio_file_path, num_candidates=10)
    #             if not sound_model_candidates_list: sound_model_candidates_list = [("", 0.0)]
    #             for i_c, (c_text, c_score) in enumerate(sound_model_candidates_list):
    #                 if verbose: print(f"   {i_c+1}. '{c_text}' (置信度: {c_score:.3f})")
    #             basic_top_sound_prediction = sound_model_candidates_list[0][0]
    #             if verbose: print(f"   >> 主要声音模型预测 (用于字符级准确率基准): '{basic_top_sound_prediction}'\n")
    #         except Exception as e_b: 
    #             print(f"    基础声音模型多候选预测失败: {e_b}"); basic_top_sound_prediction = ""; sound_model_candidates_list = [("",0.0)]
        
    #     position_info, num_keystrokes = self.extract_keystroke_probabilities(audio_file_path, verbose=verbose)

    #     # 2. 纯Seq2Seq模型预测
    #     pure_s2s_pred_text_for_log = "N/A"
    #     full_mask_tpl_for_log = "N/A"
    #     pure_s2s_full_result_dict = None # 用于存储包含 token_log_probs 的完整结果
    #     if num_keystrokes > 0:
    #         full_mask_tpl_for_log = Config.MASK_TOKEN * num_keystrokes
    #         pure_s2s_full_result_dict = self._predict_with_single_mask(full_mask_tpl_for_log, num_keystrokes)
    #         if pure_s2s_full_result_dict and pure_s2s_full_result_dict.get('text'):
    #             pure_s2s_pred_text_for_log = pure_s2s_full_result_dict['text']
    #     if verbose:
    #         print(f"2. 纯Seq2Seq模型预测 (全掩码模板: '{full_mask_tpl_for_log}'):")
    #         print(f"   >> 预测结果: '{pure_s2s_pred_text_for_log}'\n")

    #     if position_info is None or num_keystrokes == 0:
    #         if verbose: print(f"  无法提取按键概率或未检测到按键 for {audio_file_path}，高级预测中止。")
    #         acc_stats_def = {
    #             'sound_model_prediction': basic_top_sound_prediction, 'sound_model_char_accuracy':0.0, 'sound_model_sequence_accuracy':0.0,
    #             'pure_seq2seq_prediction': pure_s2s_pred_text_for_log, 'pure_seq2seq_char_accuracy':0.0, 'pure_seq2seq_sequence_accuracy':0.0,
    #             'advanced_model_prediction':"", 'advanced_model_char_accuracy':0.0, 'advanced_model_sequence_accuracy':0.0, 'advanced_model_source': 'N/A'
    #         }
    #         if expected_sequence_str_log:
    #             if basic_top_sound_prediction: 
    #                 acc_stats_def['sound_model_char_accuracy'] = self._evaluate_accuracy(expected_sequence_str_log, basic_top_sound_prediction)
    #             is_sm_seq_correct_early_exit = any(c_text == expected_sequence_str_log for c_text, _ in sound_model_candidates_list if c_text)
    #             acc_stats_def['sound_model_sequence_accuracy'] = 1.0 if is_sm_seq_correct_early_exit else 0.0
    #             if pure_s2s_pred_text_for_log != "N/A":
    #                 acc_stats_def['pure_seq2seq_char_accuracy'] = self._evaluate_accuracy(expected_sequence_str_log, pure_s2s_pred_text_for_log)
    #                 acc_stats_def['pure_seq2seq_sequence_accuracy'] = 1.0 if pure_s2s_pred_text_for_log == expected_sequence_str_log else 0.0
    #         return {'advanced': [], 'basic_top_candidate': basic_top_sound_prediction, 
    #                 'sound_model_all_candidates': sound_model_candidates_list, 
    #                 'improvement_char_level': 0.0, 'accuracy_stats': acc_stats_def}

    #     if verbose: print(f"3. 高级模型处理流程:")
    #     all_intermediate_s2s_results = []
    #     processed_s2s_masks_cache = set() 
    #     actual_sound_cands_to_use = sound_model_candidates_list[:N_SOUND_CANDIDATES_FOR_MASKING]
    #     if not actual_sound_cands_to_use : actual_sound_cands_to_use = [("",0.0)]

    #     for base_idx, (current_base_sound_pred, current_base_sound_score) in enumerate(actual_sound_cands_to_use):
    #         base_pred_for_mgen = current_base_sound_pred if len(current_base_sound_pred) == num_keystrokes else None
    #         if verbose: print(f"   --- 使用声音候选 {base_idx+1}/{len(actual_sound_cands_to_use)} ('{base_pred_for_mgen or '长度不匹配/空'}', score: {current_base_sound_score:.3f}) 作为掩码生成基础 ---")
            
    #         gen_scored_masks = self.mask_generator.generate_masks(position_info, base_prediction=base_pred_for_mgen, verbose=verbose) 
    #         if verbose:
    #             print(f"       为基础 '{base_pred_for_mgen or 'N/A'}' 初步生成了 {len(gen_scored_masks)} 个掩码模板 (按质量排序):")
    #             for i_m_orig, (m_s_orig, m_sc_orig) in enumerate(gen_scored_masks[:15]):
    #                 print(f"         {i_m_orig+1}. '{m_s_orig}' (质量: {m_sc_orig:.3f})")
    #             if len(gen_scored_masks) > 15: print("         ...")
            
    #         top_m_masks = gen_scored_masks[:MAX_MASKS_PER_SOUND_CANDIDATE]
    #         if verbose: print(f"       选择 Top {len(top_m_masks)} 个掩码进行Seq2Seq处理:")

    #         for mask_idx_loop, (mask_str_loop, mask_quality_s) in enumerate(top_m_masks):
    #             if mask_str_loop in processed_s2s_masks_cache:
    #                 if verbose: print(f"         跳过已由Seq2Seq处理的掩码: '{mask_str_loop}'")
    #                 continue
    #             processed_s2s_masks_cache.add(mask_str_loop)
    #             if verbose: print(f"         处理掩码 {mask_idx_loop+1}/{len(top_m_masks)}: '{mask_str_loop}' (质量: {mask_quality_s:.3f})")
                
    #             try:
    #                 s2s_outputs = self._predict_results_by_mask_count(mask_str_loop, num_keystrokes)
    #                 if s2s_outputs:
    #                     if verbose:
    #                         s2s_examples = [f"'{s_r['text']}'({s_r.get('seq_score',0.0):.2f})" for s_r in s2s_outputs[:3]]
    #                         print(f"           => Seq2Seq 为此掩码生成了 {len(s2s_outputs)} 个候选, 例: [{', '.join(s2s_examples)}...]")
                        
    #                     for s2s_res in s2s_outputs:
    #                         char_fusion_score_sum = 0.0; valid_char_fusion_count = 0
    #                         current_s2s_text = s2s_res.get('text','')
    #                         token_log_probs_from_s2s = s2s_res.get('token_log_probs', []) 
                            
    #                         idx_offset_for_log_probs = 1 # log_probs_from_s2s[0] is for SOS->token1, so text[0] corresponds to log_probs_from_s2s[1]
    #                                                    # Actually, if beam_search_multiple returns scores_list_logprobs for each *output token* excluding SOS, then offset is 0
    #                                                    # Let's assume token_log_probs_from_s2s[i] corresponds to current_s2s_text[i]
    #                         idx_offset_for_log_probs = 0 
    #                         # If your beam_search_multiple's scores_list_logprobs[0] is for the first *actual* char (after SOS), this is correct.
    #                         # If it includes a score for SOS (e.g., 0.0 as in my robust version), then it should be char_idx_in_seq + 1 for current_token_log_probs index.
    #                         # Given the log message "bMMMMFss]ssss]ssd<ssd<s<s<<<s<'(-40.40)", the -40.40 is likely total raw score.
    #                         # We need to ensure 'token_log_probs' has one log_prob per character in 'text'.
    #                         # The `scores_list` from your original beam_search_multiple would be suitable if it's per-token.

    #                         for char_idx_in_seq, predicted_char_by_s2s in enumerate(current_s2s_text):
    #                             if char_idx_in_seq >= num_keystrokes: break 
                                
    #                             s2s_char_log_prob = token_log_probs_from_s2s[char_idx_in_seq + idx_offset_for_log_probs] if (char_idx_in_seq + idx_offset_for_log_probs) < len(token_log_probs_from_s2s) else -20.0 
                                
    #                             sound_model_prob_for_this_char = 0.0
    #                             if char_idx_in_seq < len(position_info):
    #                                 pos_data_for_char = position_info[char_idx_in_seq]
    #                                 if pos_data_for_char.get('top_chars') and predicted_char_by_s2s in pos_data_for_char['top_chars']:
    #                                     try:
    #                                         idx_in_top = pos_data_for_char['top_chars'].index(predicted_char_by_s2s)
    #                                         if idx_in_top < len(pos_data_for_char.get('top_probs',[])):
    #                                             sound_model_prob_for_this_char = pos_data_for_char['top_probs'][idx_in_top]
    #                                     except (ValueError, IndexError): pass
                                
    #                             sound_model_log_prob = np.log(sound_model_prob_for_this_char + 1e-9) 
    #                             w_s2s_char_fus = self.config.get("overall_score.weight_s2s_char_log_prob_fusion", 0.7)
    #                             w_sound_char_fus = self.config.get("overall_score.weight_sound_char_log_prob_fusion", 0.3)
    #                             fused_char_score = w_s2s_char_fus * s2s_char_log_prob + w_sound_char_fus * sound_model_log_prob 
    #                             char_fusion_score_sum += fused_char_score
    #                             valid_char_fusion_count += 1
                            
    #                         avg_char_fusion_score = char_fusion_score_sum / valid_char_fusion_count if valid_char_fusion_count > 0 else -20.0
    #                         s2s_res['avg_char_fusion_score'] = avg_char_fusion_score
                            
    #                         s2s_res['mask_quality_score'] = mask_quality_s
    #                         s2s_res['sound_candidate_score'] = current_base_sound_score 
    #                         s2s_res['sound_candidate_text_source'] = current_base_sound_pred
    #                         s2s_res['template_name'] = f"SndCand{base_idx+1}-M{mask_idx_loop+1}"
                            
    #                         w_seq_s = self.config.get("overall_score.weight_seq_score", 0.30)
    #                         w_adher = self.config.get("overall_score.weight_mask_adherence", 0.15)
    #                         w_mask_q = self.config.get("overall_score.weight_mask_quality", 0.10)
    #                         w_sound_c = self.config.get("overall_score.weight_sound_candidate", 0.10)
    #                         w_char_f = self.config.get("overall_score.weight_char_fusion", 0.35) # Increased weight for fusion
                            
    #                         len_s2s_txt = len(current_s2s_text) if current_s2s_text else 1
                            
    #                         s2s_res['overall_score'] = ( 
    #                             w_seq_s * s2s_res.get('seq_score', -20.0) +       
    #                             w_adher * s2s_res.get('mask_adherence_score', 0.0) + 
    #                             w_mask_q * mask_quality_s +         
    #                             w_sound_c * current_base_sound_score +
    #                             w_char_f * (avg_char_fusion_score / len_s2s_txt ) 
    #                         )
    #                         all_intermediate_s2s_results.append(s2s_res)
    #             except Exception as e_s2s_p:
    #                  if verbose: print(f"           处理掩码 '{mask_str_loop}' 的Seq2Seq预测时出错: {e_s2s_p}")
        
    #     sorted_all_s2s_results = sorted(all_intermediate_s2s_results, key=lambda x: x['overall_score'], reverse=True)
    #     final_unique_advanced_s2s_results = []
    #     seen_txt_final_s2s = set()
    #     for res_dict_item in sorted_all_s2s_results:
    #         if res_dict_item.get('text') and res_dict_item['text'] not in seen_txt_final_s2s: # Ensure text exists
    #             seen_txt_final_s2s.add(res_dict_item['text'])
    #             final_unique_advanced_s2s_results.append(res_dict_item)
        
    #     accuracy_stats_dict = {
    #         'sound_model_prediction': basic_top_sound_prediction, 'sound_model_char_accuracy':0.0, 'sound_model_sequence_accuracy':0.0,
    #         'pure_seq2seq_prediction': pure_s2s_pred_text_for_log, 'pure_seq2seq_char_accuracy':0.0, 'pure_seq2seq_sequence_accuracy':0.0,
    #         'advanced_model_prediction': "N/A", 'advanced_model_char_accuracy':0.0, 'advanced_model_sequence_accuracy':0.0,
    #         'advanced_model_source': "N/A" }

    #     if expected_sequence_str_log:
    #         if basic_top_sound_prediction: 
    #             accuracy_stats_dict['sound_model_char_accuracy'] = self._evaluate_accuracy(expected_sequence_str_log, basic_top_sound_prediction)
    #         is_sm_multi_cand_seq_correct = any(c_text == expected_sequence_str_log for c_text, _ in sound_model_candidates_list if c_text)
    #         accuracy_stats_dict['sound_model_sequence_accuracy'] = 1.0 if is_sm_multi_cand_seq_correct else 0.0
    #         if pure_s2s_pred_text_for_log != "N/A":
    #             accuracy_stats_dict['pure_seq2seq_char_accuracy'] = self._evaluate_accuracy(expected_sequence_str_log, pure_s2s_pred_text_for_log)
    #             accuracy_stats_dict['pure_seq2seq_sequence_accuracy'] = 1.0 if pure_s2s_pred_text_for_log == expected_sequence_str_log else 0.0
            
    #         comprehensive_candidate_pool = []
    #         for sm_text, sm_score in sound_model_candidates_list:
    #             if sm_text:
    #                 char_acc = self._evaluate_accuracy(expected_sequence_str_log, sm_text)
    #                 seq_acc = 1.0 if sm_text == expected_sequence_str_log else 0.0
    #                 comprehensive_candidate_pool.append({'text': sm_text, 'char_acc': char_acc, 'seq_acc': seq_acc, 'score': sm_score, 'source': 'SoundModelDirectCandidate'})
    #         for adv_s2s_res in final_unique_advanced_s2s_results:
    #             adv_s2s_text = adv_s2s_res['text']
    #             if adv_s2s_text:
    #                 char_acc = self._evaluate_accuracy(expected_sequence_str_log, adv_s2s_text)
    #                 seq_acc = 1.0 if adv_s2s_text == expected_sequence_str_log else 0.0
    #                 comprehensive_candidate_pool.append({'text': adv_s2s_text, 'char_acc': char_acc, 'seq_acc': seq_acc,
    #                                                      'score': adv_s2s_res.get('overall_score', -float('inf')), 
    #                                                      'source': 'SoundMaskSeq2SeqFlow', 'details': adv_s2s_res })
    #         if comprehensive_candidate_pool:
    #             comprehensive_candidate_pool.sort(key=lambda x: (x['seq_acc'], x['char_acc'], x['score']), reverse=True)
    #             best_overall_advanced_choice = comprehensive_candidate_pool[0]
    #             accuracy_stats_dict['advanced_model_prediction'] = best_overall_advanced_choice['text']
    #             accuracy_stats_dict['advanced_model_char_accuracy'] = best_overall_advanced_choice['char_acc']
    #             accuracy_stats_dict['advanced_model_sequence_accuracy'] = best_overall_advanced_choice['seq_acc']
    #             accuracy_stats_dict['advanced_model_source'] = best_overall_advanced_choice['source']
    #             if verbose: print(f"   >> 综合选择的最佳高级模型源: {best_overall_advanced_choice['source']}, 文本: '{best_overall_advanced_choice['text']}'")
    #         else: accuracy_stats_dict['advanced_model_prediction'] = "N/A (无任何候选)"
        
    #     if verbose:
    #         print(f"\n4. 高级模型最终预测结果 (综合选择后, Top {min(10, top_k if ('comprehensive_candidate_pool' in locals() and comprehensive_candidate_pool) else len(final_unique_advanced_s2s_results))}):")
    #         display_candidates_list = comprehensive_candidate_pool if expected_sequence_str_log and 'comprehensive_candidate_pool' in locals() and comprehensive_candidate_pool else final_unique_advanced_s2s_results
    #         if display_candidates_list:
    #             for i_r, r_item in enumerate(display_candidates_list[:min(10, top_k)]):
    #                 print(f"   {i_r+1}. '{r_item['text']}'")
    #                 if r_item.get('source') == 'SoundModelDirectCandidate':
    #                     print(f"       (来源: {r_item['source']}, 声音模型置信度: {r_item.get('score',0.0):.4f})")
    #                 elif r_item.get('source') == 'SoundMaskSeq2SeqFlow':
    #                      print(f"       (来源: {r_item['source']}, 综合分: {r_item.get('score',0.0):.4f}, Seq2Seq归一化分: {r_item.get('details',{}).get('seq_score',0.0):.3f}, "
    #                            f"字符融合均分: {r_item.get('details',{}).get('avg_char_fusion_score', -99.9):.3f}, "
    #                            f"初始声音候选: '{r_item.get('details',{}).get('sound_candidate_text_source', 'N/A')}', "
    #                            f"原始掩码: '{r_item.get('details',{}).get('mask','N/A')}', 掩码质量: {r_item.get('details',{}).get('mask_quality_score',0.0):.3f})")
    #         else: print("     未能生成有效的高级预测结果。")

    #     if verbose:
    #         print("\n5. 准确率对比总结 (新定义):")
    #         if expected_sequence_str_log: print(f"   预期序列: '{expected_sequence_str_log}'")
    #         else: print("   预期序列: N/A")
    #         print(f"   声音模型 (最佳单候选): '{accuracy_stats_dict['sound_model_prediction']}' - 字符级: {accuracy_stats_dict['sound_model_char_accuracy']:.2%}")
    #         print(f"   声音模型 (多候选命中): 序列级准确率: {accuracy_stats_dict['sound_model_sequence_accuracy']:.0%}")
    #         print(f"   纯Seq2Seq (全掩码): '{accuracy_stats_dict['pure_seq2seq_prediction']}' - 字符级: {accuracy_stats_dict['pure_seq2seq_char_accuracy']:.2%}, 序列级: {accuracy_stats_dict['pure_seq2seq_sequence_accuracy']:.0%}")
    #         print(f"   高级模型 (综合最佳): '{accuracy_stats_dict['advanced_model_prediction']}' (来源: {accuracy_stats_dict.get('advanced_model_source','N/A')}) - 字符级: {accuracy_stats_dict['advanced_model_char_accuracy']:.2%}, 序列级: {accuracy_stats_dict['advanced_model_sequence_accuracy']:.0%}")

    #     improvement_char_final = 0.0
    #     sm_char_acc_for_calc = accuracy_stats_dict.get('sound_model_char_accuracy',0.0)
    #     am_char_acc_for_calc = accuracy_stats_dict.get('advanced_model_char_accuracy',0.0)
    #     if am_char_acc_for_calc > sm_char_acc_for_calc:
    #         if sm_char_acc_for_calc > 0: improvement_char_final = ((am_char_acc_for_calc - sm_char_acc_for_calc) / sm_char_acc_for_calc * 100)
    #         else: improvement_char_final = float('inf') 
        
    #     if verbose and expected_sequence_str_log:
    #          print(f"   高级模型(综合最佳)相较于声音模型(最佳单候选)的字符准确率提升: {improvement_char_final if improvement_char_final != float('inf') else '∞ ':.2f}%")
    #     if verbose: print("="*80)
        
    #     return_advanced_list = final_unique_advanced_s2s_results[:top_k] if final_unique_advanced_s2s_results else []
    #     return {'advanced': return_advanced_list, 'basic_top_candidate': basic_top_sound_prediction,
    #             'sound_model_all_candidates': sound_model_candidates_list,
    #             'improvement_char_level': improvement_char_final, 'accuracy_stats': accuracy_stats_dict}
    def predict_with_enhanced_masks(self, audio_file_path, top_k=10, verbose=True, compare_basic=True):
        # --- 参数设定 ---
        N_SOUND_CANDIDATES_FOR_MASKING = self.config.get("advanced_prediction.n_sound_candidates_for_masking", 3)
        MAX_MASKS_PER_SOUND_CANDIDATE = self.config.get("advanced_prediction.max_masks_per_base", 8)

        expected_sequence_str_log = ''.join(c for c in os.path.splitext(os.path.basename(audio_file_path))[0] if c.isdigit())
        
        if verbose:
            print("\n" + "="*80)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始处理文件: {os.path.basename(audio_file_path)}")
            if expected_sequence_str_log: print(f"预期序列: '{expected_sequence_str_log}'")
            else: print("预期序列: N/A (无法从文件名提取)")
            print("="*80 + "\n")

        # 1. 基础声音模型候选
        sound_model_candidates_list = [("", 0.0)] 
        basic_top_sound_prediction = "" 
        if compare_basic and self.basic_system:
            try:
                if verbose: print(f"1. 基础声音模型候选 (最多 Top 10):")
                sound_model_candidates_list = self.basic_system.predict_from_file_with_candidates(audio_file_path, num_candidates=10)
                if not sound_model_candidates_list: sound_model_candidates_list = [("", 0.0)]
                for i_c, (c_text, c_score) in enumerate(sound_model_candidates_list):
                    if verbose: print(f"   {i_c+1}. '{c_text}' (置信度: {c_score:.3f})")
                basic_top_sound_prediction = sound_model_candidates_list[0][0]
                if verbose: print(f"   >> 主要声音模型预测 (用于字符级准确率基准): '{basic_top_sound_prediction}'\n")
            except Exception as e_b: 
                print(f"    基础声音模型多候选预测失败: {e_b}"); basic_top_sound_prediction = ""; sound_model_candidates_list = [("",0.0)]
        
        position_info, num_keystrokes = self.extract_keystroke_probabilities(audio_file_path, verbose=verbose)

        # 2. 纯Seq2Seq模型预测
        pure_s2s_pred_text_for_log = "N/A"
        full_mask_tpl_for_log = "N/A"
        pure_s2s_full_result_dict = None
        if num_keystrokes > 0:
            full_mask_tpl_for_log = Config.MASK_TOKEN * num_keystrokes
            pure_s2s_full_result_dict = self._predict_with_single_mask(full_mask_tpl_for_log, num_keystrokes)
            if pure_s2s_full_result_dict and pure_s2s_full_result_dict.get('text'):
                pure_s2s_pred_text_for_log = pure_s2s_full_result_dict['text']
        if verbose:
            print(f"2. 纯Seq2Seq模型预测 (全掩码模板: '{full_mask_tpl_for_log}'):")
            print(f"   >> 预测结果: '{pure_s2s_pred_text_for_log}'\n")

        if position_info is None or num_keystrokes == 0:
            if verbose: print(f"  无法提取按键概率或未检测到按键 for {audio_file_path}，高级预测中止。")
            acc_stats_def = {
                'sound_model_prediction': basic_top_sound_prediction, 'sound_model_char_accuracy':0.0, 'sound_model_sequence_accuracy':0.0,
                'pure_seq2seq_prediction': pure_s2s_pred_text_for_log, 'pure_seq2seq_char_accuracy':0.0, 'pure_seq2seq_sequence_accuracy':0.0,
                'advanced_model_prediction':"", 'advanced_model_char_accuracy':0.0, 'advanced_model_sequence_accuracy':0.0, 'advanced_model_source': 'N/A',
                # 新增排名信息
                'sound_model_best_rank': -1,
                'mask_best_template': "N/A",
                'mask_best_rank': -1
            }
            if expected_sequence_str_log:
                if basic_top_sound_prediction: 
                    acc_stats_def['sound_model_char_accuracy'] = self._evaluate_accuracy(expected_sequence_str_log, basic_top_sound_prediction)
                # 计算声音模型排名
                sound_rank = -1
                for i, (cand_text, _) in enumerate(sound_model_candidates_list):
                    if cand_text == expected_sequence_str_log:
                        sound_rank = i + 1
                        break
                acc_stats_def['sound_model_best_rank'] = sound_rank
                
                is_sm_seq_correct_early_exit = any(c_text == expected_sequence_str_log for c_text, _ in sound_model_candidates_list if c_text)
                acc_stats_def['sound_model_sequence_accuracy'] = 1.0 if is_sm_seq_correct_early_exit else 0.0
                if pure_s2s_pred_text_for_log != "N/A":
                    acc_stats_def['pure_seq2seq_char_accuracy'] = self._evaluate_accuracy(expected_sequence_str_log, pure_s2s_pred_text_for_log)
                    acc_stats_def['pure_seq2seq_sequence_accuracy'] = 1.0 if pure_s2s_pred_text_for_log == expected_sequence_str_log else 0.0
            return {'advanced': [], 'basic_top_candidate': basic_top_sound_prediction, 
                    'sound_model_all_candidates': sound_model_candidates_list, 
                    'improvement_char_level': 0.0, 'accuracy_stats': acc_stats_def, 
                    'used_masks': []}  # 添加空的掩码列表

        if verbose: print(f"3. 高级模型处理流程:")
        all_intermediate_s2s_results = []
        processed_s2s_masks_cache = set() 
        actual_sound_cands_to_use = sound_model_candidates_list[:N_SOUND_CANDIDATES_FOR_MASKING]
        if not actual_sound_cands_to_use : actual_sound_cands_to_use = [("",0.0)]

        # 记录生成的掩码模板 - 使用更详细的数据结构
        all_generated_masks_detailed = []

        for base_idx, (current_base_sound_pred, current_base_sound_score) in enumerate(actual_sound_cands_to_use):
            base_pred_for_mgen = current_base_sound_pred if len(current_base_sound_pred) == num_keystrokes else None
            if verbose: print(f"   --- 使用声音候选 {base_idx+1}/{len(actual_sound_cands_to_use)} ('{base_pred_for_mgen or '长度不匹配/空'}', score: {current_base_sound_score:.3f}) 作为掩码生成基础 ---")
            
            gen_scored_masks = self.mask_generator.generate_masks(position_info, base_prediction=base_pred_for_mgen, verbose=verbose) 
            
            # 记录所有掩码模板，包含更多信息
            for mask_str, mask_score in gen_scored_masks:
                all_generated_masks_detailed.append({
                    'mask': mask_str,
                    'score': mask_score,
                    'base_prediction': base_pred_for_mgen,
                    'base_idx': base_idx,
                    'source_candidate': current_base_sound_pred
                })
            
            if verbose:
                print(f"       为基础 '{base_pred_for_mgen or 'N/A'}' 初步生成了 {len(gen_scored_masks)} 个掩码模板 (按质量排序):")
                for i_m_orig, (m_s_orig, m_sc_orig) in enumerate(gen_scored_masks[:15]):
                    print(f"         {i_m_orig+1}. '{m_s_orig}' (质量: {m_sc_orig:.3f})")
                if len(gen_scored_masks) > 15: print("         ...")
            
            top_m_masks = gen_scored_masks[:MAX_MASKS_PER_SOUND_CANDIDATE]
            if verbose: print(f"       选择 Top {len(top_m_masks)} 个掩码进行Seq2Seq处理:")

            for mask_idx_loop, (mask_str_loop, mask_quality_s) in enumerate(top_m_masks):
                if mask_str_loop in processed_s2s_masks_cache:
                    if verbose: print(f"         跳过已由Seq2Seq处理的掩码: '{mask_str_loop}'")
                    continue
                processed_s2s_masks_cache.add(mask_str_loop)
                if verbose: print(f"         处理掩码 {mask_idx_loop+1}/{len(top_m_masks)}: '{mask_str_loop}' (质量: {mask_quality_s:.3f})")
                
                try:
                    s2s_outputs = self._predict_results_by_mask_count(mask_str_loop, num_keystrokes)
                    if s2s_outputs:
                        if verbose:
                            s2s_examples = [f"'{s_r['text']}'({s_r.get('seq_score',0.0):.2f})" for s_r in s2s_outputs[:3]]
                            print(f"           => Seq2Seq 为此掩码生成了 {len(s2s_outputs)} 个候选, 例: [{', '.join(s2s_examples)}...]")
                        
                        for s2s_res in s2s_outputs:
                            char_fusion_score_sum = 0.0; valid_char_fusion_count = 0
                            current_s2s_text = s2s_res.get('text','')
                            token_log_probs_from_s2s = s2s_res.get('token_log_probs', []) 
                            
                            idx_offset_for_log_probs = 0 
                            
                            for char_idx_in_seq, predicted_char_by_s2s in enumerate(current_s2s_text):
                                if char_idx_in_seq >= num_keystrokes: break 
                                
                                s2s_char_log_prob = token_log_probs_from_s2s[char_idx_in_seq + idx_offset_for_log_probs] if (char_idx_in_seq + idx_offset_for_log_probs) < len(token_log_probs_from_s2s) else -20.0 
                                
                                sound_model_prob_for_this_char = 0.0
                                if char_idx_in_seq < len(position_info):
                                    pos_data_for_char = position_info[char_idx_in_seq]
                                    if pos_data_for_char.get('top_chars') and predicted_char_by_s2s in pos_data_for_char['top_chars']:
                                        try:
                                            idx_in_top = pos_data_for_char['top_chars'].index(predicted_char_by_s2s)
                                            if idx_in_top < len(pos_data_for_char.get('top_probs',[])):
                                                sound_model_prob_for_this_char = pos_data_for_char['top_probs'][idx_in_top]
                                        except (ValueError, IndexError): pass
                                
                                sound_model_log_prob = np.log(sound_model_prob_for_this_char + 1e-9) 
                                w_s2s_char_fus = self.config.get("overall_score.weight_s2s_char_log_prob_fusion", 0.7)
                                w_sound_char_fus = self.config.get("overall_score.weight_sound_char_log_prob_fusion", 0.3)
                                fused_char_score = w_s2s_char_fus * s2s_char_log_prob + w_sound_char_fus * sound_model_log_prob 
                                char_fusion_score_sum += fused_char_score
                                valid_char_fusion_count += 1
                            
                            avg_char_fusion_score = char_fusion_score_sum / valid_char_fusion_count if valid_char_fusion_count > 0 else -20.0
                            s2s_res['avg_char_fusion_score'] = avg_char_fusion_score
                            
                            s2s_res['mask_quality_score'] = mask_quality_s
                            s2s_res['sound_candidate_score'] = current_base_sound_score 
                            s2s_res['sound_candidate_text_source'] = current_base_sound_pred
                            s2s_res['template_name'] = f"SndCand{base_idx+1}-M{mask_idx_loop+1}"
                            
                            w_seq_s = self.config.get("overall_score.weight_seq_score", 0.30)
                            w_adher = self.config.get("overall_score.weight_mask_adherence", 0.15)
                            w_mask_q = self.config.get("overall_score.weight_mask_quality", 0.10)
                            w_sound_c = self.config.get("overall_score.weight_sound_candidate", 0.10)
                            w_char_f = self.config.get("overall_score.weight_char_fusion", 0.35)
                            
                            len_s2s_txt = len(current_s2s_text) if current_s2s_text else 1
                            
                            s2s_res['overall_score'] = ( 
                                w_seq_s * s2s_res.get('seq_score', -20.0) +       
                                w_adher * s2s_res.get('mask_adherence_score', 0.0) + 
                                w_mask_q * mask_quality_s +         
                                w_sound_c * current_base_sound_score +
                                w_char_f * (avg_char_fusion_score / len_s2s_txt ) 
                            )
                            all_intermediate_s2s_results.append(s2s_res)
                except Exception as e_s2s_p:
                    if verbose: print(f"           处理掩码 '{mask_str_loop}' 的Seq2Seq预测时出错: {e_s2s_p}")
        
        sorted_all_s2s_results = sorted(all_intermediate_s2s_results, key=lambda x: x['overall_score'], reverse=True)
        final_unique_advanced_s2s_results = []
        seen_txt_final_s2s = set()
        for res_dict_item in sorted_all_s2s_results:
            if res_dict_item.get('text') and res_dict_item['text'] not in seen_txt_final_s2s:
                seen_txt_final_s2s.add(res_dict_item['text'])
                final_unique_advanced_s2s_results.append(res_dict_item)
        
        accuracy_stats_dict = {
            'sound_model_prediction': basic_top_sound_prediction, 'sound_model_char_accuracy':0.0, 'sound_model_sequence_accuracy':0.0,
            'pure_seq2seq_prediction': pure_s2s_pred_text_for_log, 'pure_seq2seq_char_accuracy':0.0, 'pure_seq2seq_sequence_accuracy':0.0,
            'advanced_model_prediction': "N/A", 'advanced_model_char_accuracy':0.0, 'advanced_model_sequence_accuracy':0.0,
            'advanced_model_source': "N/A",
            # 新增排名信息
            'sound_model_best_rank': -1,
            'mask_best_template': "N/A",
            'mask_best_rank': -1
        }

        if expected_sequence_str_log:
            if basic_top_sound_prediction: 
                accuracy_stats_dict['sound_model_char_accuracy'] = self._evaluate_accuracy(expected_sequence_str_log, basic_top_sound_prediction)
            
            # 计算声音模型排名
            sound_rank = -1
            for i, (cand_text, _) in enumerate(sound_model_candidates_list):
                if cand_text == expected_sequence_str_log:
                    sound_rank = i + 1
                    break
            accuracy_stats_dict['sound_model_best_rank'] = sound_rank
            
            is_sm_multi_cand_seq_correct = any(c_text == expected_sequence_str_log for c_text, _ in sound_model_candidates_list if c_text)
            accuracy_stats_dict['sound_model_sequence_accuracy'] = 1.0 if is_sm_multi_cand_seq_correct else 0.0
            if pure_s2s_pred_text_for_log != "N/A":
                accuracy_stats_dict['pure_seq2seq_char_accuracy'] = self._evaluate_accuracy(expected_sequence_str_log, pure_s2s_pred_text_for_log)
                accuracy_stats_dict['pure_seq2seq_sequence_accuracy'] = 1.0 if pure_s2s_pred_text_for_log == expected_sequence_str_log else 0.0
            
            comprehensive_candidate_pool = []
            for sm_text, sm_score in sound_model_candidates_list:
                if sm_text:
                    char_acc = self._evaluate_accuracy(expected_sequence_str_log, sm_text)
                    seq_acc = 1.0 if sm_text == expected_sequence_str_log else 0.0
                    comprehensive_candidate_pool.append({'text': sm_text, 'char_acc': char_acc, 'seq_acc': seq_acc, 'score': sm_score, 'source': 'SoundModelDirectCandidate'})
            for adv_s2s_res in final_unique_advanced_s2s_results:
                adv_s2s_text = adv_s2s_res['text']
                if adv_s2s_text:
                    char_acc = self._evaluate_accuracy(expected_sequence_str_log, adv_s2s_text)
                    seq_acc = 1.0 if adv_s2s_text == expected_sequence_str_log else 0.0
                    comprehensive_candidate_pool.append({'text': adv_s2s_text, 'char_acc': char_acc, 'seq_acc': seq_acc,
                                                        'score': adv_s2s_res.get('overall_score', -float('inf')), 
                                                        'source': 'SoundMaskSeq2SeqFlow', 'details': adv_s2s_res })
            if comprehensive_candidate_pool:
                comprehensive_candidate_pool.sort(key=lambda x: (x['seq_acc'], x['char_acc'], x['score']), reverse=True)
                best_overall_advanced_choice = comprehensive_candidate_pool[0]
                accuracy_stats_dict['advanced_model_prediction'] = best_overall_advanced_choice['text']
                accuracy_stats_dict['advanced_model_char_accuracy'] = best_overall_advanced_choice['char_acc']
                accuracy_stats_dict['advanced_model_sequence_accuracy'] = best_overall_advanced_choice['seq_acc']
                accuracy_stats_dict['advanced_model_source'] = best_overall_advanced_choice['source']
                
                # 计算掩码模板猜对位置
                s2s_flow_rank = -1
                for i, s2s_result in enumerate(final_unique_advanced_s2s_results):
                    if s2s_result.get('text') == expected_sequence_str_log:
                        s2s_flow_rank = i + 1
                        break
                accuracy_stats_dict['mask_best_rank'] = s2s_flow_rank
                                
                # 找到最佳掩码模板
                best_mask_template = "N/A"
                if best_overall_advanced_choice['source'] == 'SoundMaskSeq2SeqFlow' and 'details' in best_overall_advanced_choice:
                    best_mask_template = best_overall_advanced_choice['details'].get('mask', 'N/A')
                elif best_overall_advanced_choice['source'] == 'SoundModelDirectCandidate':
                    best_mask_template = "声音模型直接候选"
                accuracy_stats_dict['mask_best_template'] = best_mask_template
                
                if verbose: print(f"   >> 综合选择的最佳高级模型源: {best_overall_advanced_choice['source']}, 文本: '{best_overall_advanced_choice['text']}'")
            else: 
                accuracy_stats_dict['advanced_model_prediction'] = "N/A (无任何候选)"
        
        if verbose:
            print(f"\n4. 高级模型最终预测结果 (综合选择后, Top {min(10, top_k if ('comprehensive_candidate_pool' in locals() and comprehensive_candidate_pool) else len(final_unique_advanced_s2s_results))}):")
            display_candidates_list = comprehensive_candidate_pool if expected_sequence_str_log and 'comprehensive_candidate_pool' in locals() and comprehensive_candidate_pool else final_unique_advanced_s2s_results
            if display_candidates_list:
                for i_r, r_item in enumerate(display_candidates_list[:min(10, top_k)]):
                    print(f"   {i_r+1}. '{r_item['text']}'")
                    if r_item.get('source') == 'SoundModelDirectCandidate':
                        print(f"       (来源: {r_item['source']}, 声音模型置信度: {r_item.get('score',0.0):.4f})")
                    elif r_item.get('source') == 'SoundMaskSeq2SeqFlow':
                        print(f"       (来源: {r_item['source']}, 综合分: {r_item.get('score',0.0):.4f}, Seq2Seq归一化分: {r_item.get('details',{}).get('seq_score',0.0):.3f}, "
                            f"字符融合均分: {r_item.get('details',{}).get('avg_char_fusion_score', -99.9):.3f}, "
                            f"初始声音候选: '{r_item.get('details',{}).get('sound_candidate_text_source', 'N/A')}', "
                            f"原始掩码: '{r_item.get('details',{}).get('mask','N/A')}', 掩码质量: {r_item.get('details',{}).get('mask_quality_score',0.0):.3f})")
            else: print("     未能生成有效的高级预测结果。")

        if verbose:
            print("\n5. 准确率对比总结 (新定义):")
            if expected_sequence_str_log: print(f"   预期序列: '{expected_sequence_str_log}'")
            else: print("   预期序列: N/A")
            print(f"   声音模型 (最佳单候选): '{accuracy_stats_dict['sound_model_prediction']}' - 字符级: {accuracy_stats_dict['sound_model_char_accuracy']:.2%}, 排名: {accuracy_stats_dict['sound_model_best_rank']}")
            print(f"   声音模型 (多候选命中): 序列级准确率: {accuracy_stats_dict['sound_model_sequence_accuracy']:.0%}")
            print(f"   纯Seq2Seq (全掩码): '{accuracy_stats_dict['pure_seq2seq_prediction']}' - 字符级: {accuracy_stats_dict['pure_seq2seq_char_accuracy']:.2%}, 序列级: {accuracy_stats_dict['pure_seq2seq_sequence_accuracy']:.0%}")
            print(f"   高级模型 (综合最佳): '{accuracy_stats_dict['advanced_model_prediction']}' (来源: {accuracy_stats_dict.get('advanced_model_source','N/A')}) - 字符级: {accuracy_stats_dict['advanced_model_char_accuracy']:.2%}, 序列级: {accuracy_stats_dict['advanced_model_sequence_accuracy']:.0%}, 排名: {accuracy_stats_dict['mask_best_rank']}")

        improvement_char_final = 0.0
        sm_char_acc_for_calc = accuracy_stats_dict.get('sound_model_char_accuracy',0.0)
        am_char_acc_for_calc = accuracy_stats_dict.get('advanced_model_char_accuracy',0.0)
        if am_char_acc_for_calc > sm_char_acc_for_calc:
            if sm_char_acc_for_calc > 0: improvement_char_final = ((am_char_acc_for_calc - sm_char_acc_for_calc) / sm_char_acc_for_calc * 100)
            else: improvement_char_final = float('inf') 
        
        if verbose and expected_sequence_str_log:
            if improvement_char_final == float('inf'):
                improvement_str = "∞"
            else:
                improvement_str = f"{improvement_char_final:.2f}"
            print(f"   高级模型(综合最佳)相较于声音模型(最佳单候选)的字符准确率提升: {improvement_str}%")
        if verbose: print("="*80)
        
        # ==== 整理掩码信息 ====
        used_masks_info = []
        seen_masks_set = set()
        
        # 从实际使用的掩码中提取信息
        for mask_info in all_generated_masks_detailed:
            mask_str = mask_info['mask']
            
            if mask_str not in seen_masks_set:
                seen_masks_set.add(mask_str)
                
                # 查找模板名称（通过分析掩码模式）
                template_name = "未知类型"
                mask_token = Config.MASK_TOKEN if hasattr(Config, 'MASK_TOKEN') else '￥'
                
                # 分析掩码模式来确定类型
                if mask_str == mask_token * num_keystrokes:
                    template_name = "全掩码"
                elif mask_str.count(mask_token) == 0:
                    template_name = "无掩码"
                elif mask_str.startswith(mask_token):
                    template_name = f"前缀掩码({mask_str.count(mask_token)}位)"
                elif mask_str.endswith(mask_token):
                    template_name = f"后缀掩码({mask_str.count(mask_token)}位)"
                else:
                    # 分析掩码分布
                    mask_positions = [i for i, c in enumerate(mask_str) if c == mask_token]
                    if len(mask_positions) == 1:
                        template_name = f"单位掩码(位置{mask_positions[0]+1})"
                    else:
                        template_name = f"混合掩码({len(mask_positions)}位)"
                
                mask_item = {
                    'template': mask_str,
                    'type': template_name,
                    'maskCount': mask_str.count(mask_token),
                    'score': mask_info.get('score', 0.0)
                }
                
                # 添加源候选信息
                if mask_info.get('source_candidate'):
                    mask_item['sourceCandidate'] = mask_info['source_candidate']
                    mask_item['candidateRank'] = mask_info.get('base_idx', 0) + 1
                
                used_masks_info.append(mask_item)
        
        # 按分数排序并限制数量
        used_masks_info.sort(key=lambda x: x['score'], reverse=True)
        used_masks_info = used_masks_info[:20]  # 只返回前20个掩码
        
        return_advanced_list = final_unique_advanced_s2s_results[:top_k] if final_unique_advanced_s2s_results else []
        return {
            'advanced': return_advanced_list, 
            'basic_top_candidate': basic_top_sound_prediction,
            'sound_model_all_candidates': sound_model_candidates_list,
            'improvement_char_level': improvement_char_final, 
            'accuracy_stats': accuracy_stats_dict,
            'used_masks': used_masks_info  # 添加掩码信息
        }
    # predict_directory (与上次代码一致，它现在依赖更新后的 predict_with_enhanced_masks)
    # def predict_directory(self, dir_path, top_k=10, verbose=False, save_viz=False):
    #     # (这个函数的代码与我上一条回复中的版本基本一致，它依赖 predict_with_enhanced_masks 更新后的 accuracy_stats)
    #     # 只需要确保CSV列名和打印的统计信息与新的准确率定义对齐
    #     if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
    #         print(f"错误: 目录 {dir_path} 不存在或不是有效目录。"); return {}
    #     wav_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.wav')]
    #     if not wav_files: print(f"错误: 目录 {dir_path} 中没有WAV文件"); return {}
        
    #     print(f"在目录 {dir_path} 中找到 {len(wav_files)} 个WAV文件。")
    #     results_file_path = os.path.join(dir_path, f"adv_pred_results_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        
    #     # 更新CSV表头
    #     csv_headers = [
    #         "文件名", "预期序列",
    #         "声音模型预测(最佳单候选)", "声音模型字符准确率(最佳单候选)", "声音模型序列准确率(多候选命中)",
    #         "纯Seq2Seq预测", "纯Seq2Seq字符准确率", "纯Seq2Seq序列准确率",
    #         "高级模型预测(综合最佳)", "高级模型字符准确率(综合最佳)", "高级模型序列准确率(综合最佳)", "高级模型结果来源",
    #         "提升率(字符级, 高级vs声音最佳单)", "相关S2S流程模板名称", "相关S2S流程掩码"
    #     ]
    #     with open(results_file_path, 'w', encoding='utf-8', newline='') as f:
    #         import csv; writer = csv.writer(f); writer.writerow(csv_headers)

    #     self.current_run_stats = defaultdict(float)
    #     self.current_run_stats['best_model_type_counts'] = defaultdict(int) 
    #     self.current_run_stats['start_time'] = time.time() 
    #     all_individual_file_results_map = {}

    #     print("\n开始批量高级预测...")
    #     for i, filename in enumerate(wav_files):
    #         file_path = os.path.join(dir_path, filename)
    #         print(f"\n[{i+1}/{len(wav_files)}] 处理文件: {filename}") 
    #         expected_s = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
    #         if expected_s: print(f"  预期序列: '{expected_s}'")
    #         else: print(f"  警告: 文件 '{filename}' 未能提取到预期序列。")
            
    #         try:
    #             file_pred_out_dict = self.predict_with_enhanced_masks(file_path, top_k=top_k, verbose=verbose, compare_basic=True)
    #             acc_s = file_pred_out_dict.get('accuracy_stats', {})
                
    #             sound_p_txt = str(acc_s.get('sound_model_prediction', "N/A"))
    #             sound_c_acc = float(acc_s.get('sound_model_char_accuracy', 0.0))
    #             sound_s_acc = float(acc_s.get('sound_model_sequence_accuracy', 0.0))
    #             pure_s2s_p_txt = str(acc_s.get('pure_seq2seq_prediction', "N/A"))
    #             pure_s2s_c_acc = float(acc_s.get('pure_seq2seq_char_accuracy', 0.0))
    #             pure_s2s_s_acc = float(acc_s.get('pure_seq2seq_sequence_accuracy', 0.0))
    #             adv_p_txt = str(acc_s.get('advanced_model_prediction', "N/A"))
    #             adv_c_acc = float(acc_s.get('advanced_model_char_accuracy', 0.0))
    #             adv_s_acc = float(acc_s.get('advanced_model_sequence_accuracy', 0.0))
    #             adv_source = str(acc_s.get('advanced_model_source', "N/A"))

    #             improve_char_val = file_pred_out_dict.get('improvement_char_level', 0.0)
    #             improve_csv = "0.00"
    #             if isinstance(improve_char_val, (int, float)):
    #                 if improve_char_val == float('inf'): improve_csv = "inf"
    #                 elif improve_char_val != 0.0: improve_csv = f"{improve_char_val:.2f}"
                
    #             s2s_flow_template_name, s2s_flow_mask_str = "N/A", "N/A"
    #             if adv_source == 'SoundMaskSeq2SeqFlow':
    #                 best_s2s_detail = None
    #                 if file_pred_out_dict.get('advanced'): # 'advanced' contains list of S2S flow results
    #                      potential_matches = [item for item in file_pred_out_dict['advanced'] if item['text'] == adv_p_txt]
    #                      if potential_matches: best_s2s_detail = potential_matches[0]
    #                 if best_s2s_detail:
    #                     s2s_flow_template_name = best_s2s_detail.get('template_name', 'N/A')
    #                     s2s_flow_mask_str = best_s2s_detail.get('mask', 'N/A')
    #             elif adv_source == 'SoundModelDirectCandidate':
    #                  s2s_flow_template_name = "N/A (源自声音模型)"
    #                  s2s_flow_mask_str = "N/A (源自声音模型)"

    #             if not verbose:
    #                 improve_disp_str = improve_csv + "%" if improve_csv != "inf" else "inf%"
    #                 print(f"  高级模型预测(综合最佳): '{adv_p_txt}' (来源: {adv_source}, 字符级: {adv_c_acc:.2%}, 序列级: {adv_s_acc:.0%}, 提升: {improve_disp_str})")

    #             with open(results_file_path, 'a', encoding='utf-8', newline='') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow([
    #                     filename, expected_s,
    #                     sound_p_txt, f"{sound_c_acc:.4f}", f"{sound_s_acc:.0f}",
    #                     pure_s2s_p_txt, f"{pure_s2s_c_acc:.4f}", f"{pure_s2s_s_acc:.0f}",
    #                     adv_p_txt, f"{adv_c_acc:.4f}", f"{adv_s_acc:.0f}", adv_source,
    #                     improve_csv, s2s_flow_template_name, s2s_flow_mask_str
    #                 ])
    #             all_individual_file_results_map[filename] = file_pred_out_dict

    #             if expected_s:
    #                 self.current_run_stats['total_files_with_expected_sequence'] += 1
    #                 seq_l = len(expected_s)
    #                 if seq_l > 0:
    #                     self.current_run_stats['total_expected_chars'] += seq_l
    #                     self.current_run_stats['sound_model_total_correct_chars'] += sound_c_acc * seq_l
    #                     self.current_run_stats['sound_model_total_correct_sequences'] += sound_s_acc
    #                     self.current_run_stats['pure_seq2seq_total_correct_chars'] += pure_s2s_c_acc * seq_l
    #                     self.current_run_stats['pure_seq2seq_total_correct_sequences'] += pure_s2s_s_acc
    #                     self.current_run_stats['advanced_model_total_correct_chars'] += adv_c_acc * seq_l
    #                     self.current_run_stats['advanced_model_total_correct_sequences'] += adv_s_acc
    #                     self.current_run_stats['best_model_type_counts'][adv_source] += 1
    #             if save_viz and self.basic_system: 
    #                  try: create_comparison_visualization(file_path, expected_s or "N/A", sound_p_txt, adv_p_txt, self.basic_system)
    #                  except Exception as e_viz_b: print(f"  可视化生成失败 for {filename}: {e_viz_b}")
    #         except Exception as e_file_m:
    #             print(f"处理文件 {filename} 时发生严重错误: {e_file_m}"); traceback.print_exc()
    #             with open(results_file_path, 'a', encoding='utf-8', newline='') as f:
    #                 writer = csv.writer(f); error_row = [filename, expected_s] + ["ERROR"]*3 + ["0.0000"]*7 + ["0.00", "N/A", "N/A"] 
    #                 writer.writerow(error_row)
        
    #     elapsed_run_t = time.time() - self.current_run_stats['start_time']
    #     print("\n====== 目录预测统计信息 (本次运行) ======")
    #     total_valid_f = self.current_run_stats['total_files_with_expected_sequence']
    #     total_exp_c = self.current_run_stats['total_expected_chars']
    #     print(f"已处理文件总数 (有预期序列的): {int(total_valid_f)}")
    #     if total_valid_f == 0: print("没有文件带有可用于统计的预期序列。"); return all_individual_file_results_map
    #     print(f"总预期字符数: {int(total_exp_c)}")
    #     print(f"处理时间: {elapsed_run_t:.2f}秒 (平均: {elapsed_run_t/max(1, len(wav_files)):.2f}秒/文件)")

    #     sm_c_acc_all = (self.current_run_stats['sound_model_total_correct_chars'] / total_exp_c) if total_exp_c > 0 else 0
    #     sm_s_acc_all = (self.current_run_stats['sound_model_total_correct_sequences'] / total_valid_f) if total_valid_f > 0 else 0
    #     print(f"\n声音模型总体 (字符准确率基于最佳单候选，序列准确率基于多候选命中):")
    #     print(f"  字符准确率={sm_c_acc_all:.2%}, 序列准确率={sm_s_acc_all:.2%}")
        
    #     ps_c_acc_all = (self.current_run_stats['pure_seq2seq_total_correct_chars'] / total_exp_c) if total_exp_c > 0 else 0
    #     ps_s_acc_all = (self.current_run_stats['pure_seq2seq_total_correct_sequences'] / total_valid_f) if total_valid_f > 0 else 0
    #     print(f"纯Seq2Seq总体: 字符准确率={ps_c_acc_all:.2%}, 序列准确率={ps_s_acc_all:.2%}")
        
    #     am_c_acc_all = (self.current_run_stats['advanced_model_total_correct_chars'] / total_exp_c) if total_exp_c > 0 else 0
    #     am_s_acc_all = (self.current_run_stats['advanced_model_total_correct_sequences'] / total_valid_f) if total_valid_f > 0 else 0
    #     print(f"高级模型总体 (综合最佳选择):")
    #     print(f"  字符准确率={am_c_acc_all:.2%}, 序列准确率={am_s_acc_all:.2%}")

    #     if sm_c_acc_all > 0 and am_c_acc_all > sm_c_acc_all:
    #         char_improve_all = (am_c_acc_all - sm_c_acc_all) / sm_c_acc_all * 100
    #         print(f"高级模型(综合)相较于声音模型(最佳单候选)的字符准确率提升: {char_improve_all:.2f}%")
    #     elif am_c_acc_all > 0 and sm_c_acc_all == 0: print(f"高级模型(综合)相较于声音模型(最佳单候选)的字符准确率提升: ∞")
        
    #     print(f"\n高级模型最佳结果来源统计:")
    #     for source_type_key, count_val_stat in self.current_run_stats['best_model_type_counts'].items():
    #          print(f"  来自 {source_type_key}: {int(count_val_stat)} 次 ({count_val_stat / total_valid_f:.2% if total_valid_f > 0 else 0.0})")
    #     print(f"\n详细结果已保存至: {results_file_path}")
    #     return all_individual_file_results_map

    def predict_directory(self, dir_path, top_k=10, verbose=False, save_viz=False):
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            print(f"错误: 目录 {dir_path} 不存在或不是有效目录。"); return {}
        wav_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.wav')]
        if not wav_files: print(f"错误: 目录 {dir_path} 中没有WAV文件"); return {}
        
        print(f"在目录 {dir_path} 中找到 {len(wav_files)} 个WAV文件。")
        results_file_path = os.path.join(dir_path, f"adv_pred_results_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        
        # 更新CSV表头
        csv_headers = [
            "文件名", "预期序列",
            "声音模型最佳结果", "声音模型最佳位置", "声音模型字符准确率",
            "纯Seq2Seq最佳结果", "纯Seq2Seq字符准确率",
            "最佳掩码模板", "最终融合最佳结果", "掩码模板猜对位置", "最终字符准确率"
        ]
        
        with open(results_file_path, 'w', encoding='utf-8', newline='') as f:
            import csv; writer = csv.writer(f); writer.writerow(csv_headers)

        self.current_run_stats = defaultdict(float)
        self.current_run_stats['best_model_type_counts'] = defaultdict(int) 
        self.current_run_stats['start_time'] = time.time() 
        all_individual_file_results_map = {}

        print("\n开始批量高级预测...")
        for i, filename in enumerate(wav_files):
            file_path = os.path.join(dir_path, filename)
            print(f"\n[{i+1}/{len(wav_files)}] 处理文件: {filename}") 
            expected_s = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
            if expected_s: print(f"  预期序列: '{expected_s}'")
            else: print(f"  警告: 文件 '{filename}' 未能提取到预期序列。")
            
            try:
                file_pred_out_dict = self.predict_with_enhanced_masks(file_path, top_k=top_k, verbose=verbose, compare_basic=True)
                acc_s = file_pred_out_dict.get('accuracy_stats', {})
                
                sound_p_txt = str(acc_s.get('sound_model_prediction', "N/A"))
                sound_rank = int(acc_s.get('sound_model_best_rank', -1))
                sound_c_acc = float(acc_s.get('sound_model_char_accuracy', 0.0))
                
                pure_s2s_p_txt = str(acc_s.get('pure_seq2seq_prediction', "N/A"))
                pure_s2s_c_acc = float(acc_s.get('pure_seq2seq_char_accuracy', 0.0))
                
                mask_template = str(acc_s.get('mask_best_template', "N/A"))
                adv_p_txt = str(acc_s.get('advanced_model_prediction', "N/A"))
                mask_rank = int(acc_s.get('mask_best_rank', -1))
                adv_c_acc = float(acc_s.get('advanced_model_char_accuracy', 0.0))

                if not verbose:
                    print(f"  声音模型: '{sound_p_txt}' (位置:{sound_rank}, 准确率:{sound_c_acc:.2%})")
                    print(f"  纯Seq2Seq: '{pure_s2s_p_txt}' (准确率:{pure_s2s_c_acc:.2%})")
                    print(f"  高级模型: '{adv_p_txt}' (位置:{mask_rank}, 准确率:{adv_c_acc:.2%})")

                with open(results_file_path, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        filename, expected_s,
                        sound_p_txt, sound_rank, f"{sound_c_acc:.4f}",
                        pure_s2s_p_txt, f"{pure_s2s_c_acc:.4f}",
                        mask_template, adv_p_txt, mask_rank, f"{adv_c_acc:.4f}"
                    ])
                all_individual_file_results_map[filename] = file_pred_out_dict

                if expected_s:
                    self.current_run_stats['total_files_with_expected_sequence'] += 1
                    seq_l = len(expected_s)
                    if seq_l > 0:
                        self.current_run_stats['total_expected_chars'] += seq_l
                        self.current_run_stats['sound_model_total_correct_chars'] += sound_c_acc * seq_l
                        self.current_run_stats['pure_seq2seq_total_correct_chars'] += pure_s2s_c_acc * seq_l
                        self.current_run_stats['advanced_model_total_correct_chars'] += adv_c_acc * seq_l
                        
                        # 统计排名信息
                        if sound_rank > 0:
                            self.current_run_stats['sound_model_hit_count'] += 1
                            self.current_run_stats['sound_model_total_rank'] += sound_rank
                        if mask_rank > 0:
                            self.current_run_stats['mask_model_hit_count'] += 1
                            self.current_run_stats['mask_model_total_rank'] += mask_rank

                if save_viz and self.basic_system: 
                    try: create_comparison_visualization(file_path, expected_s or "N/A", sound_p_txt, adv_p_txt, self.basic_system)
                    except Exception as e_viz_b: print(f"  可视化生成失败 for {filename}: {e_viz_b}")
            except Exception as e_file_m:
                print(f"处理文件 {filename} 时发生严重错误: {e_file_m}"); traceback.print_exc()
                with open(results_file_path, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f); error_row = [filename, expected_s] + ["ERROR"]*3 + ["0.0000"]*4 + ["-1", "0.0000"] 
                    writer.writerow(error_row)
        
        elapsed_run_t = time.time() - self.current_run_stats['start_time']
        print("\n====== 目录预测统计信息 (本次运行) ======")
        total_valid_f = self.current_run_stats['total_files_with_expected_sequence']
        total_exp_c = self.current_run_stats['total_expected_chars']
        print(f"已处理文件总数 (有预期序列的): {int(total_valid_f)}")
        if total_valid_f == 0: print("没有文件带有可用于统计的预期序列。"); return all_individual_file_results_map
        print(f"总预期字符数: {int(total_exp_c)}")
        print(f"处理时间: {elapsed_run_t:.2f}秒 (平均: {elapsed_run_t/max(1, len(wav_files)):.2f}秒/文件)")

        sm_c_acc_all = (self.current_run_stats['sound_model_total_correct_chars'] / total_exp_c) if total_exp_c > 0 else 0
        ps_c_acc_all = (self.current_run_stats['pure_seq2seq_total_correct_chars'] / total_exp_c) if total_exp_c > 0 else 0
        am_c_acc_all = (self.current_run_stats['advanced_model_total_correct_chars'] / total_exp_c) if total_exp_c > 0 else 0
        
        print(f"\n准确率统计:")
        print(f"声音模型总体字符准确率: {sm_c_acc_all:.2%}")
        print(f"纯Seq2Seq总体字符准确率: {ps_c_acc_all:.2%}")
        print(f"高级模型总体字符准确率: {am_c_acc_all:.2%}")

        # 排名统计
        if self.current_run_stats['sound_model_hit_count'] > 0:
            avg_sound_rank = self.current_run_stats['sound_model_total_rank'] / self.current_run_stats['sound_model_hit_count']
            print(f"声音模型命中率: {self.current_run_stats['sound_model_hit_count']}/{total_valid_f} ({self.current_run_stats['sound_model_hit_count']/total_valid_f:.2%}), 平均排名: {avg_sound_rank:.1f}")
        
        if self.current_run_stats['mask_model_hit_count'] > 0:
            avg_mask_rank = self.current_run_stats['mask_model_total_rank'] / self.current_run_stats['mask_model_hit_count']
            print(f"掩码模型命中率: {self.current_run_stats['mask_model_hit_count']}/{total_valid_f} ({self.current_run_stats['mask_model_hit_count']/total_valid_f:.2%}), 平均排名: {avg_mask_rank:.1f}")

        if sm_c_acc_all > 0 and am_c_acc_all > sm_c_acc_all:
            char_improve_all = (am_c_acc_all - sm_c_acc_all) / sm_c_acc_all * 100
            print(f"高级模型相较于声音模型的字符准确率提升: {char_improve_all:.2f}%")
        elif am_c_acc_all > 0 and sm_c_acc_all == 0: 
            print(f"高级模型相较于声音模型的字符准确率提升: ∞")
        
        print(f"\n详细结果已保存至: {results_file_path}")
        return all_individual_file_results_map

# --- 全局 create_comparison_visualization 函数 ---
def create_comparison_visualization(file_path, expected, basic_pred_str, advanced_pred_str, basic_rec_system_instance):
    # (与我之前提供的版本一致)
    if not basic_rec_system_instance or not hasattr(basic_rec_system_instance, 'audio_processor'):
        print(f"  可视化错误 for {file_path}: 未提供有效的 KeystrokeRecognitionSystem 实例或其 audio_processor 不可用。")
        return
    try:
        y, sr = basic_rec_system_instance.audio_processor.load_audio(file_path)
        segments, segment_times, _ = basic_rec_system_instance.audio_processor.detect_keystrokes(y, sr) 
    except Exception as e_audio_viz:
        print(f"  可视化错误: 加载或处理音频 {file_path} 失败: {e_audio_viz}")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    times_audio = np.linspace(0, len(y) / sr, num=len(y))

    axes[0].plot(times_audio, y, label="音频波形", color='darkblue', linewidth=0.7)
    axes[0].set_title("原始音频波形与按键分割", fontsize=13)
    axes[0].set_ylabel("振幅", fontsize=11)
    min_y_ax0, max_y_ax0 = axes[0].get_ylim()
    text_y_pos_ax0 = max_y_ax0 * 0.80 if max_y_ax0 > min_y_ax0 else 0 

    for i_seg, (start, end) in enumerate(segment_times):
        axes[0].axvspan(start, end, alpha=0.2, color='lightcoral', label='按键段' if i_seg == 0 else "")
        axes[0].text((start + end) / 2, text_y_pos_ax0, f"S{i_seg+1}", ha='center', va='center', fontsize=8, color='maroon', bbox=dict(facecolor='white', alpha=0.5, pad=1))
    if segment_times : axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    def plot_preds_on_ax(ax, title_prefix, pred_text, y_audio_data, seg_times_list, exp_seq_text):
        ax.plot(times_audio, y_audio_data, color='silver', linewidth=0.7, alpha=0.6)
        ax.set_title(f"{title_prefix}: '{pred_text}'", fontsize=13)
        ax.set_ylabel("振幅", fontsize=11)
        _, ax_max_y = ax.get_ylim()
        pred_text_y_pos = ax_max_y * 0.80 if ax_max_y > ax.get_ylim()[0] else 0
        for i_s, (s_time, e_time) in enumerate(seg_times_list):
            if i_s < len(pred_text):
                char_predicted = pred_text[i_s]
                is_correct_char = exp_seq_text and exp_seq_text != "N/A" and i_s < len(exp_seq_text) and char_predicted == exp_seq_text[i_s]
                box_clr = 'lightgreen' if is_correct_char else 'lightcoral'
                text_clr = 'darkgreen' if is_correct_char else 'darkred'
                ax.axvspan(s_time, e_time, alpha=0.3, color=box_clr)
                ax.text((s_time + e_time) / 2, pred_text_y_pos, char_predicted, 
                         ha='center', va='center', fontsize=10, color=text_clr, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.6, pad=1, boxstyle="round,pad=0.3"))
        ax.grid(True, linestyle=':', alpha=0.6)

    plot_preds_on_ax(axes[1], "声音模型(最佳单候选)", basic_pred_str, y, segment_times, expected)
    plot_preds_on_ax(axes[2], "高级模型(综合最佳)", advanced_pred_str, y, segment_times, expected)
    
    axes[2].set_xlabel("时间 (秒)", fontsize=12)
    fig.suptitle(f"预测对比: {os.path.basename(file_path)}\n预期序列: {expected or 'N/A'}", fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    results_dir_path = basic_rec_system_instance.config_manager.get_path("results_dir", "results") 
    viz_subdir = os.path.join(results_dir_path, "visualizations_advanced_comparison") 
    os.makedirs(viz_subdir, exist_ok=True)
    time_stamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_audio_filename = "".join(c if c.isalnum() else "_" for c in os.path.splitext(os.path.basename(file_path))[0])
    save_figure_path = os.path.join(viz_subdir, f'adv_compare_{safe_audio_filename}_{time_stamp_str}.png')
    try:
        plt.savefig(save_figure_path, dpi=150)
    except Exception as e_save_fig: print(f"  保存可视化图像失败 for {file_path}: {e_save_fig}")
    plt.close(fig)

# --- 脚本入口函数 (由 main_enhanced.py 调用) ---
# 注意：这些函数定义在 advanced_prediction.py 的全局作用域
# main_enhanced.py 中会 from advanced_prediction import advanced_predict_file, advanced_predict_directory

def advanced_predict_file(config_manager_obj):
    # (与我上次回复中的版本一致，它会调用更新后的 EnhancedPredictionSystem)
    seq2seq_model_file = "seq_best_model.pth" 
    if not os.path.exists(seq2seq_model_file):
        print(f"警告: Seq2Seq模型文件不存在: {seq2seq_model_file}")
        if input("是否继续高级预测（将使用随机初始化的Seq2Seq模型）? (y/n): ").lower() != 'y':
            print("高级预测已取消。"); return False
    custom_eps_sound_model_dir = input("请输入用于高级预测的声音模型目录 (EPS自用CNN/LSTM等) [可选, 回车使用默认配置]: ").strip() or None
    if custom_eps_sound_model_dir and not os.path.isdir(custom_eps_sound_model_dir):
        print(f"错误: 指定的声音模型目录 '{custom_eps_sound_model_dir}' 无效或不是目录。"); return False
    
    prediction_system_inst = EnhancedPredictionSystem(config_manager_obj, seq2seq_model_file, sound_model_dir_override=custom_eps_sound_model_dir)
    audio_f_path = input("请输入音频文件路径: ")
    if not os.path.exists(audio_f_path) or not os.path.isfile(audio_f_path):
        print(f"错误: 文件 '{audio_f_path}' 不存在或不是有效文件。"); return False
    top_k_results_to_show = int(input("返回的最佳高级结果数量 (显示用) [默认5, 最多10]: ") or "5") # 调整了默认和上限
    top_k_results_to_show = min(max(1, top_k_results_to_show), 10) 
    print("\n开始高级预测...")
    start_time_single_pred = time.time()
    prediction_output = prediction_system_inst.predict_with_enhanced_masks(audio_f_path, top_k=top_k_results_to_show, verbose=True, compare_basic=True)
    elapsed_time_single_pred = time.time() - start_time_single_pred
    print(f"\n文件 '{os.path.basename(audio_f_path)}' 预测完成! (用时: {elapsed_time_single_pred:.2f}秒)")
    
    acc_stats_res = prediction_output.get('accuracy_stats', {})
    expected_seq_from_filename = ''.join(c for c in os.path.splitext(os.path.basename(audio_f_path))[0] if c.isdigit())
    print("\n--- 预测结果与准确率对比 (新定义) ---")
    if expected_seq_from_filename: print(f"预期序列:     '{expected_seq_from_filename}'")
    else: print("预期序列:     N/A (无法从文件名提取)")
    
    def print_accuracy_line_detail(model_name_str, pred_key_str, char_acc_key_str, seq_acc_key_str, stats_data_dict, expected_seq_exists_bool, extra_info=""):
        pred_val = str(stats_data_dict.get(pred_key_str, 'N/A')); char_acc_val = float(stats_data_dict.get(char_acc_key_str, 0.0))
        seq_acc_val = float(stats_data_dict.get(seq_acc_key_str, 0.0))
        acc_display_str = f" (字符级: {char_acc_val:.2%}, 序列级: {seq_acc_val:.0%})" if expected_seq_exists_bool else ""
        print(f"{model_name_str:<28}: '{pred_val}'{acc_display_str} {extra_info}")

    print_accuracy_line_detail("声音模型(最佳单候选)", 'sound_model_prediction', 'sound_model_char_accuracy', 
                               'sound_model_sequence_accuracy', acc_stats_res, bool(expected_seq_from_filename), 
                               f"(序列准确率基于多候选)")
    print_accuracy_line_detail("纯Seq2Seq(全掩码)", 'pure_seq2seq_prediction', 'pure_seq2seq_char_accuracy', 
                               'pure_seq2seq_sequence_accuracy', acc_stats_res, bool(expected_seq_from_filename))
    adv_model_source_info = f"(来源: {acc_stats_res.get('advanced_model_source','N/A')})"
    print_accuracy_line_detail("高级模型(综合最佳)", 'advanced_model_prediction', 'advanced_model_char_accuracy', 
                               'advanced_model_sequence_accuracy', acc_stats_res, bool(expected_seq_from_filename), adv_model_source_info)

    if expected_seq_from_filename:
        improvement_val = prediction_output.get('improvement_char_level', 0.0)
        if improvement_val == float('inf'): print(f"高级模型(综合)相较于声音模型(最佳单候选)的字符准确率提升: ∞")
        elif improvement_val != 0.0 : print(f"高级模型(综合)相较于声音模型(最佳单候选)的字符准确率提升: {improvement_val:.2f}%")

    print("\n--- 高级模型最终选择池的Top结果 (部分) ---")
    # prediction_output['advanced'] 在新逻辑下可能指的是S2S流程的结果，而真正的综合最佳在 accuracy_stats 中
    # 或者，让 predict_with_enhanced_masks 返回的 'advanced' 字段就是 comprehensive_candidate_pool 的前 top_k
    # 这里假设 'advanced' 列表包含了带有 'source' 和 'score' (综合池排序依据) 的字典
    adv_results_to_display = prediction_output.get('advanced_comprehensive_pool_top', []) # 假设返回了这个新键
    if not adv_results_to_display and prediction_output.get('advanced'): # Fallback to S2S flow results if comprehensive not directly returned
        adv_results_to_display = prediction_output['advanced']
        print("   (仅显示S2S流程产出的高分候选，综合选择的最佳预测已在上方准确率部分显示)")


    if adv_results_to_display:
        for i_res, res_item_dict in enumerate(adv_results_to_display[:min(5, top_k_results_to_show)]): 
            print(f"  {i_res+1}. '{res_item_dict.get('text','N/A')}'")
            if res_item_dict.get('source') == 'SoundModelDirectCandidate':
                 print(f"       (来源: {res_item_dict['source']}, 声音模型置信度: {res_item_dict.get('score',0.0):.4f})")
            elif res_item_dict.get('source') == 'SoundMaskSeq2SeqFlow' and 'details' in res_item_dict:
                 details = res_item_dict['details']
                 print(f"       (来源: {res_item_dict['source']}, 综合分: {res_item_dict.get('score',0.0):.4f}, Seq2Seq归一化分: {details.get('seq_score',0.0):.3f}, "
                       f"字符融合均分: {details.get('avg_char_fusion_score', -99.9):.3f}, "
                       f"初始声音候选: '{details.get('sound_candidate_text_source', 'N/A')}', "
                       f"原始掩码: '{details.get('mask','N/A')}', 掩码质量: {details.get('mask_quality_score',0.0):.3f})")
            else: # Fallback for S2S flow if 'details' not present
                print(f"       (综合分: {res_item_dict.get('overall_score',res_item_dict.get('score',0.0)):.4f}, 模板: {res_item_dict.get('template_name','N/A')}, 掩码: '{res_item_dict.get('mask','N/A')}')")


    if prediction_system_inst.basic_system:
        try:
            create_comparison_visualization(audio_f_path, expected_seq_from_filename or "N/A", 
                                            str(acc_stats_res.get('sound_model_prediction','')), 
                                            str(acc_stats_res.get('advanced_model_prediction','')), 
                                            prediction_system_inst.basic_system )
            print(f"对比可视化图已生成。")
        except Exception as e_viz_adv_file: print(f"可视化结果时出错: {e_viz_adv_file}")
    else: print("无法生成对比可视化：basic_system 未有效初始化。")
    return True

def advanced_predict_directory(config_manager_obj): # (与我之前提供的版本一致)
    # ... (代码同上一个回复)
    seq2seq_model_file = "seq_best_model.pth" 
    if not os.path.exists(seq2seq_model_file):
        print(f"警告: Seq2Seq模型文件不存在: {seq2seq_model_file}")
        if input("是否继续高级预测（将使用随机初始化的Seq2Seq模型）? (y/n): ").lower() != 'y':
            print("高级预测已取消。"); return False
    custom_eps_sound_model_dir = input("请输入用于高级预测的声音模型目录 (EPS自用CNN/LSTM等) [可选, 回车使用默认配置]: ").strip() or None
    if custom_eps_sound_model_dir and not os.path.isdir(custom_eps_sound_model_dir):
        print(f"错误: 指定的声音模型目录 '{custom_eps_sound_model_dir}' 无效或不是目录。"); return False
    
    prediction_system_inst = EnhancedPredictionSystem(config_manager_obj, seq2seq_model_file, sound_model_dir_override=custom_eps_sound_model_dir)
    audio_dir_path = input("请输入音频文件目录路径: ")
    if not os.path.isdir(audio_dir_path): print(f"错误: 目录 '{audio_dir_path}' 不是有效目录。"); return False
    
    # top_k 参数现在主要用于 predict_with_enhanced_masks 返回的 'advanced' 列表长度，
    # 它影响的是S2S流程结果的返回数量，间接影响CSV中“相关S2S流程模板/掩码”的获取。
    top_k_s2s_results_for_csv = int(input("每个文件高级预测S2S流程产出的候选数量 (影响CSV细节) [默认5]: ") or "5")
    top_k_s2s_results_for_csv = min(max(1, top_k_s2s_results_for_csv), 10) 
    
    save_all_visualizations = input("是否为每个文件保存可视化对比图? [y/n, 默认n]: ").lower() == 'y'
    verbose_each_file = input("是否显示每个文件的详细处理信息? [y/n, 默认n]: ").lower() == 'y'
    print("\n开始批量高级预测...")
    start_time_batch_pred = time.time()
    dir_results_map = prediction_system_inst.predict_directory(
        audio_dir_path, 
        top_k=top_k_s2s_results_for_csv, # 传递给predict_directory，它再传递给predict_with_enhanced_masks
        verbose=verbose_each_file, 
        save_viz=save_all_visualizations
    )
    elapsed_time_batch_pred = time.time() - start_time_batch_pred
    print(f"\n整个高级预测过程用时: {elapsed_time_batch_pred:.2f}秒。")
    return bool(dir_results_map)

if __name__ == "__main__": # (与我之前提供的版本一致)
    print("独立运行 advanced_prediction.py (测试模式)")
    class DummyConfigManagerForTest:
        def __init__(self):
            self.config_data = {
                "paths": {"model_dir": "models", "results_dir": "results_test_adv", "data_dir": "data_test_adv" },
                "audio_processing": {"sample_rate": 44100, "silence_thresh_db": -40.0, "min_silence_len_ms": 50 },
                "feature_extraction": {"n_mfcc": 20, "max_len_mfcc": 100 },
                "advanced_prediction": {"n_sound_candidates_for_masking": 3, "max_masks_per_base": 8 }, # 高级预测参数
                "seq2seq_prediction": { # Seq2Seq 预测相关参数
                    "results_per_mask_1": 5, "results_per_mask_2": 15, 
                    "results_per_mask_3": 30, "results_per_mask_gt3_multiplier": 3
                },
                "mask_generation": {"max_mask_ratio": 0.7}, # 掩码生成器最大掩码率
                "scoring_weights": { # 新增的评分权重配置
                    "weight_seq_score": 0.30,
                    "weight_mask_adherence": 0.15,
                    "weight_mask_quality": 0.10,
                    "weight_sound_candidate": 0.10,
                    "weight_char_fusion": 0.35,
                    "weight_s2s_char_log_prob_fusion": 0.7, # 字符级融合中s2s的权重
                    "weight_sound_char_log_prob_fusion": 0.3 # 字符级融合中声音模型的权重
                }
            }
            for p_key in self.config_data["paths"]: os.makedirs(self.config_data["paths"][p_key], exist_ok=True)
        def get(self, key_path, default_val=None):
            keys = key_path.split('.'); val = self.config_data
            try:
                for k in keys: val = val[k]
                return val
            except (KeyError, TypeError): return default_val
        def get_path(self, key_name, default_sub_val=None):
            path = self.config_data["paths"].get(key_name)
            if path: return path
            if default_sub_val: return default_sub_val
            return key_name 
        @property
        def config_path(self): return None 
    test_config_mgr = DummyConfigManagerForTest()
    # print("\n--- 测试单个文件高级预测 ---"); # advanced_predict_file(test_config_mgr)
    # print("\n--- 测试目录高级预测 ---"); # advanced_predict_directory(test_config_mgr)
    print("\n独立测试结束。请通过 main_enhanced.py 运行完整系统。")