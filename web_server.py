# web_server.py - Flask后端服务（优化版）
import os
import json
import traceback
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import time
import tempfile
from werkzeug.utils import secure_filename

# 导入你的现有模块
from config_manager import ConfigManager
from advanced_prediction import EnhancedPredictionSystem

app = Flask(__name__)
CORS(app)  # 允许跨域请求

class AdvancedTestProgress:
   """高级测试进度管理类"""
   def __init__(self):
       self.progress = 0
       self.message = "准备中..."
       self.logs = []
       self.results = None
       self.error = None
       self.completed = False
       self.current_file_results = None  # 当前文件的详细结果
       self.log_buffer = []  # 日志缓冲区
       self.last_log_time = time.time()

   def update_progress(self, progress, message):
       self.progress = progress
       self.message = message

   def add_log(self, message, force=False):
       """添加日志到缓冲区，减少频繁更新"""
       timestamp = datetime.now().strftime("%H:%M:%S")
       log_entry = f"[{timestamp}] {message}"
       self.log_buffer.append(log_entry)
       
       # 每秒或强制刷新缓冲区
       current_time = time.time()
       if force or (current_time - self.last_log_time) > 1.0:
           self.logs.extend(self.log_buffer)
           self.log_buffer = []
           self.last_log_time = current_time

   def flush_logs(self):
       """刷新所有缓冲的日志"""
       if self.log_buffer:
           self.logs.extend(self.log_buffer)
           self.log_buffer = []

   def set_current_file_results(self, results):
       """设置当前文件的详细预测结果"""
       self.current_file_results = results

   def set_results(self, results):
       self.flush_logs()  # 确保所有日志都被刷新
       self.results = results
       self.completed = True

   def set_error(self, error):
       self.flush_logs()  # 确保所有日志都被刷新
       self.error = error
       self.completed = True

# 全局进度跟踪器
current_test_progress = None

@app.route('/')
def index():
   """提供HTML界面"""
   try:
       with open('model_test_interface.html', 'r', encoding='utf-8') as f:
           return f.read()
   except FileNotFoundError:
       return """
       <html>
       <head><title>键盘声音识别系统</title></head>
       <body>
       <h1>键盘声音识别系统 - 高级预测模式</h1>
       <p>请将 model_test_interface.html 文件放在与此服务相同的目录中。</p>
       <p>或者访问 <a href="/api/status">/api/status</a> 查看API状态。</p>
       </body>
       </html>
       """

@app.route('/api/status')
def api_status():
   """API状态检查"""
   return jsonify({
       "status": "running",
       "message": "键盘声音识别系统后端服务运行正常（高级预测模式）",
       "timestamp": datetime.now().isoformat(),
       "version": "2.0.0",
       "mode": "advanced_prediction"
   })

@app.route('/api/advanced_predict', methods=['POST'])
def advanced_predict():
   """执行高级预测的API"""
   global current_test_progress
   
   try:
       config = request.json
       if not config:
           return jsonify({"error": "缺少配置参数"}), 400

       # 创建新的进度跟踪器
       current_test_progress = AdvancedTestProgress()
       
       # 在后台线程中执行测试
       thread = threading.Thread(target=run_advanced_prediction, args=(config,))
       thread.daemon = True
       thread.start()

       # 返回流式响应
       return Response(
           generate_advanced_stream(),
           mimetype='application/json',
           headers={
               'Cache-Control': 'no-cache',
               'Connection': 'keep-alive',
           }
       )

   except Exception as e:
       return jsonify({"error": f"启动高级预测失败: {str(e)}"}), 500

def generate_advanced_stream():
   """生成高级预测流式响应"""
   global current_test_progress
   
   if current_test_progress is None:
       yield json.dumps({"type": "error", "message": "预测未初始化"}) + '\n'
       return

   last_progress = -1
   last_log_count = 0

   while not current_test_progress.completed or current_test_progress.current_file_results is not None:
       # 刷新日志缓冲区
       current_test_progress.flush_logs()
       
       # 发送进度更新
       if current_test_progress.progress != last_progress:
           yield json.dumps({
               "type": "progress",
               "progress": current_test_progress.progress,
               "message": current_test_progress.message
           }) + '\n'
           last_progress = current_test_progress.progress

       # 发送新的日志（批量发送）
       if len(current_test_progress.logs) > last_log_count:
           new_logs = current_test_progress.logs[last_log_count:]
           for log in new_logs:
               yield json.dumps({
                   "type": "log",
                   "message": log
               }) + '\n'
           last_log_count = len(current_test_progress.logs)

       # 发送当前文件的实时结果
       if current_test_progress.current_file_results:
           yield json.dumps({
               "type": "file_result",
               "data": current_test_progress.current_file_results
           }) + '\n'
           current_test_progress.current_file_results = None

       # 检查是否完成
       if current_test_progress.completed and current_test_progress.current_file_results is None:
            break

       time.sleep(0.5)  # 减少检查频率，提升性能

   # 发送最终结果
   if current_test_progress.error:
       yield json.dumps({
           "type": "error",
           "message": current_test_progress.error
       }) + '\n'
   elif current_test_progress.results:
       yield json.dumps({
           "type": "final_result",
           "data": current_test_progress.results
       }) + '\n'

def run_advanced_prediction(config):
   """在后台线程中运行高级预测"""
   global current_test_progress
   
   try:
       current_test_progress.add_log("开始初始化高级预测环境...")
       current_test_progress.update_progress(5, "初始化配置管理器...")

       # 初始化配置管理器
       config_manager = ConfigManager()
       
       # 设置模型目录（如果提供）
       original_model_dir = config_manager.get("paths.model_dir")
       if config.get('modelDir'):
           config_manager.set("paths.model_dir", config['modelDir'])
           current_test_progress.add_log(f"使用自定义模型目录: {config['modelDir']}")
       else:
           current_test_progress.add_log(f"使用默认模型目录: {original_model_dir}")

       current_test_progress.update_progress(10, "初始化高级预测系统...")
       
       # 检查Seq2Seq模型（使用与功能12相同的模型）
       seq2seq_model_path = "best_model_PIN_Dodonew_v2.pth"
       if not os.path.exists(seq2seq_model_path):
           current_test_progress.add_log(f"警告: Seq2Seq模型文件不存在: {seq2seq_model_path}")
           current_test_progress.add_log("将使用随机初始化的Seq2Seq模型")
       
       # 创建高级预测系统
       prediction_system = EnhancedPredictionSystem(
           config_manager, 
           seq2seq_model_path,
           sound_model_dir_override=config.get('modelDir')
       )
       current_test_progress.add_log("高级预测系统初始化完成")

       # 检查测试目录
       test_dir = config.get('testDir', 'test')
       if not os.path.exists(test_dir):
           raise Exception(f"测试目录不存在: {test_dir}")

       # 获取测试文件
       wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
       if not wav_files:
           raise Exception(f"测试目录中没有WAV文件: {test_dir}")

       current_test_progress.add_log(f"找到 {len(wav_files)} 个测试文件")
       current_test_progress.update_progress(15, f"准备处理 {len(wav_files)} 个文件...")

       # 准备结果收集
       all_file_results = []
       
       # 初始化统计变量
       total_files_with_expected_sequence = 0
       total_expected_chars = 0
       sound_model_total_correct_chars = 0
       pure_seq2seq_total_correct_chars = 0
       advanced_model_total_correct_chars = 0
       sound_model_hit_count = 0
       sound_model_total_rank = 0
       mask_model_hit_count = 0
       mask_model_total_rank = 0
       
       # 处理每个文件
       for i, filename in enumerate(wav_files):
           try:
               file_path = os.path.join(test_dir, filename)
               progress = 15 + (i / len(wav_files)) * 70
               current_test_progress.update_progress(progress, f"处理文件: {filename}")
               current_test_progress.add_log(f"\n[{i+1}/{len(wav_files)}] 处理文件: {filename}")

               # 从文件名提取预期序列
               expected_s = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
               if expected_s:
                   current_test_progress.add_log(f"  从文件名提取的预期序列: '{expected_s}'")
               else:
                   current_test_progress.add_log(f"  警告: 文件 '{filename}' 未能提取到预期序列。")

               # 调用 predict_with_enhanced_masks
               file_pred_out_dict = prediction_system.predict_with_enhanced_masks(
                   file_path,
                   top_k=10,  # 使用功能12的默认值
                   verbose=False,  # 关闭内部verbose以提升性能
                   compare_basic=True
               )

               # 提取结果
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

               # 打印简要结果
               current_test_progress.add_log(f"  声音模型: '{sound_p_txt}' (位置:{sound_rank}, 准确率:{sound_c_acc:.2%})")
               current_test_progress.add_log(f"  纯Seq2Seq: '{pure_s2s_p_txt}' (准确率:{pure_s2s_c_acc:.2%})")
               current_test_progress.add_log(f"  高级模型: '{adv_p_txt}' (位置:{mask_rank}, 准确率:{adv_c_acc:.2%})")

               # 构建前端需要的文件结果数据
               file_result = {
                   "filename": filename,
                   "expected": expected_s,
                   "soundModelResults": [],
                   "maskTemplates": [],
                   "finalResults": [],
                   "accuracyStats": {
                       "soundModel": sound_c_acc,
                       "pureSeq2seq": pure_s2s_c_acc,
                       "advanced": adv_c_acc
                   }
               }

               # 1. 声音模型结果（Top 10）
               sound_candidates = file_pred_out_dict.get('sound_model_all_candidates', [])
               for idx, (seq, conf) in enumerate(sound_candidates[:10]):
                   file_result["soundModelResults"].append({
                       "rank": idx + 1,
                       "sequence": seq,
                       "confidence": float(conf),
                       "isCorrect": seq == expected_s,
                       "isUsedForMask": idx < 3
                   })

               # 2. 收集实际使用的掩码模板
               if file_pred_out_dict.get('used_masks'):
                   for mask_info in file_pred_out_dict['used_masks']:
                       file_result["maskTemplates"].append({
                           "template": mask_info.get('template', ''),
                           "type": mask_info.get('type', '未知类型'),
                           "maskCount": mask_info.get('maskCount', 0)
                       })

               # 3. 最终预测结果
               advanced_results = file_pred_out_dict.get('advanced', [])
               for idx, result in enumerate(advanced_results):
                   file_result["finalResults"].append({
                       "rank": idx + 1,
                       "sequence": result['text'],
                       "isCorrect": result['text'] == expected_s,
                       "maskTemplate": result.get('mask', ''),
                       "templateName": result.get('template_name', ''),
                       "overallScore": result.get('overall_score', 0.0)
                   })

               # 发送当前文件结果到前端
               current_test_progress.set_current_file_results(file_result)
               all_file_results.append(file_result)

               # 更新统计
               if expected_s:
                   total_files_with_expected_sequence += 1
                   seq_l = len(expected_s)
                   if seq_l > 0:
                       total_expected_chars += seq_l
                       sound_model_total_correct_chars += sound_c_acc * seq_l
                       pure_seq2seq_total_correct_chars += pure_s2s_c_acc * seq_l
                       advanced_model_total_correct_chars += adv_c_acc * seq_l
                       
                       if sound_rank > 0:
                           sound_model_hit_count += 1
                           sound_model_total_rank += sound_rank
                       if mask_rank > 0:
                           mask_model_hit_count += 1
                           mask_model_total_rank += mask_rank

           except Exception as e:
               current_test_progress.add_log(f"处理文件 {filename} 时发生错误: {str(e)}", force=True)
               # 添加错误文件的占位结果
               error_result = {
                   "filename": filename,
                   "expected": ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit()),
                   "soundModelResults": [],
                   "maskTemplates": [],
                   "finalResults": [],
                   "accuracyStats": {
                       "soundModel": 0,
                       "pureSeq2seq": 0,
                       "advanced": 0
                   }
               }
               all_file_results.append(error_result)
               continue

       # 计算最终统计
       current_test_progress.update_progress(90, "计算最终统计...")
       
       current_test_progress.add_log("\n====== 目录预测统计信息 ======", force=True)
       current_test_progress.add_log(f"已处理文件总数 (有预期序列的): {int(total_files_with_expected_sequence)}")
       
       if total_files_with_expected_sequence == 0:
           current_test_progress.add_log("没有文件带有可用于统计的预期序列。")
           summary = {
               "totalFiles": len(wav_files),
               "processedFiles": 0,
               "soundModelCharAccuracy": 0,
               "pureSeq2seqCharAccuracy": 0,
               "advancedCharAccuracy": 0,
               "soundModelHitRate": 0,
               "soundModelAvgRank": 0,
               "maskModelHitRate": 0,
               "maskModelAvgRank": 0,
               "soundSequenceAccuracy": 0,
               "advancedSequenceAccuracy": 0
           }
       else:
           current_test_progress.add_log(f"总预期字符数: {int(total_expected_chars)}")
           
           # 计算字符准确率
           sm_c_acc_all = (sound_model_total_correct_chars / total_expected_chars) if total_expected_chars > 0 else 0
           ps_c_acc_all = (pure_seq2seq_total_correct_chars / total_expected_chars) if total_expected_chars > 0 else 0
           am_c_acc_all = (advanced_model_total_correct_chars / total_expected_chars) if total_expected_chars > 0 else 0
           
           current_test_progress.add_log(f"\n准确率统计:")
           current_test_progress.add_log(f"声音模型总体字符准确率: {sm_c_acc_all:.2%}")
           current_test_progress.add_log(f"纯Seq2Seq总体字符准确率: {ps_c_acc_all:.2%}")
           current_test_progress.add_log(f"高级模型总体字符准确率: {am_c_acc_all:.2%}")
           
           # 计算命中率和平均排名
           if sound_model_hit_count > 0:
               avg_sound_rank = sound_model_total_rank / sound_model_hit_count
               current_test_progress.add_log(f"声音模型命中率: {sound_model_hit_count}/{total_files_with_expected_sequence} ({sound_model_hit_count/total_files_with_expected_sequence:.2%}), 平均排名: {avg_sound_rank:.1f}")
           else:
               avg_sound_rank = 0
               current_test_progress.add_log(f"声音模型命中率: 0/{total_files_with_expected_sequence} (0.00%)")
           
           if mask_model_hit_count > 0:
               avg_mask_rank = mask_model_total_rank / mask_model_hit_count
               current_test_progress.add_log(f"掩码模型命中率: {mask_model_hit_count}/{total_files_with_expected_sequence} ({mask_model_hit_count/total_files_with_expected_sequence:.2%}), 平均排名: {avg_mask_rank:.1f}")
           else:
               avg_mask_rank = 0
               current_test_progress.add_log(f"掩码模型命中率: 0/{total_files_with_expected_sequence} (0.00%)")
           
           # 计算提升
           if sm_c_acc_all > 0 and am_c_acc_all > sm_c_acc_all:
               char_improvement_overall = (am_c_acc_all - sm_c_acc_all) / sm_c_acc_all * 100
               current_test_progress.add_log(f"高级模型相较于声音模型的字符准确率提升: {char_improvement_overall:.2f}%")
           elif am_c_acc_all > 0 and sm_c_acc_all == 0:
               current_test_progress.add_log(f"高级模型相较于声音模型的字符准确率提升: ∞ (声音模型准确率为0)")
           
           # 构建summary对象
           summary = {
               "totalFiles": len(wav_files),
               "processedFiles": total_files_with_expected_sequence,
               "soundModelCharAccuracy": round(sm_c_acc_all * 100, 2),
               "pureSeq2seqCharAccuracy": round(ps_c_acc_all * 100, 2),
               "advancedCharAccuracy": round(am_c_acc_all * 100, 2),
               "soundModelHitRate": round((sound_model_hit_count / total_files_with_expected_sequence * 100) if total_files_with_expected_sequence > 0 else 0, 2),
               "soundModelAvgRank": round(avg_sound_rank, 1),
               "maskModelHitRate": round((mask_model_hit_count / total_files_with_expected_sequence * 100) if total_files_with_expected_sequence > 0 else 0, 2),
               "maskModelAvgRank": round(avg_mask_rank, 1),
               "soundSequenceAccuracy": round((sound_model_hit_count / total_files_with_expected_sequence * 100) if total_files_with_expected_sequence > 0 else 0, 2),
               "advancedSequenceAccuracy": round((mask_model_hit_count / total_files_with_expected_sequence * 100) if total_files_with_expected_sequence > 0 else 0, 2)
           }

       # 构建最终结果
       results = {
           "summary": summary,
           "fileResults": all_file_results
       }

       current_test_progress.update_progress(100, "高级预测完成!")
       current_test_progress.add_log("所有文件处理完成", force=True)
       current_test_progress.set_results(results)

       # 恢复原始配置
       if config.get('modelDir'):
           config_manager.set("paths.model_dir", original_model_dir)

   except Exception as e:
       error_msg = f"高级预测执行失败: {str(e)}"
       current_test_progress.add_log(f"错误: {error_msg}", force=True)
       current_test_progress.add_log(f"详细错误: {traceback.format_exc()}", force=True)
       current_test_progress.set_error(error_msg)

@app.route('/api/export_results', methods=['POST'])
def export_results():
    """保存测试结果到temp_log目录（匹配前端调用的端点名称）"""
    try:
        data = request.json
        
        # 创建temp_log目录
        temp_log_dir = 'temp_log'
        os.makedirs(temp_log_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        brand = data.get('brand', 'unknown')
        filename = f'test_results_{brand}_{timestamp}.json'
        filepath = os.path.join(temp_log_dir, filename)
        
        # 保存JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': f'结果已保存到: {filename}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history')
def get_history():
    """获取历史记录列表"""
    try:
        temp_log_dir = 'temp_log'
        if not os.path.exists(temp_log_dir):
            return jsonify({'results': []})
        
        files = []
        for filename in os.listdir(temp_log_dir):
            if filename.endswith('.json') and filename.startswith('test_results_'):
                filepath = os.path.join(temp_log_dir, filename)
                try:
                    # 读取文件获取基本信息
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # 兼容不同的数据结构
                        if 'metadata' in data:
                            # 新格式（带metadata）
                            export_time = data['metadata'].get('exportTime', '')
                            brand = data['metadata'].get('brand', '未知')
                            total_files = data['summary'].get('totalFiles', 0)
                        else:
                            # 旧格式（直接存储）
                            export_time = data.get('exportTime', '')
                            brand = data.get('brand', '未知')
                            total_files = data.get('totalFiles', 0)
                        
                        files.append({
                            'filename': filename,
                            'exportTime': export_time,
                            'brand': brand,
                            'totalFiles': total_files
                        })
                except Exception as e:
                    print(f"读取文件 {filename} 失败: {str(e)}")
                    continue
        
        # 按时间排序（最新的在前）
        files.sort(key=lambda x: x['exportTime'], reverse=True)
        
        return jsonify({'results': files})
        
    except Exception as e:
        print(f"获取历史记录失败: {str(e)}")
        return jsonify({'error': str(e), 'results': []}), 500

@app.route('/api/history/<filename>')
def get_history_detail(filename):
    """获取特定历史记录的详细数据"""
    try:
        # 防止路径遍历攻击
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'error': '非法文件名'}), 400
            
        filepath = os.path.join('temp_log', filename)
        if not os.path.exists(filepath):
            return jsonify({'error': f'文件不存在: {filename}'}), 404
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return jsonify(data)
        
    except Exception as e:
        print(f"读取历史详情失败: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/history')
def history_page():
    """提供历史记录查看器页面"""
    try:
        with open('history_viewer.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <head><title>历史记录查看器</title></head>
        <body>
        <h1>错误</h1>
        <p>history_viewer.html 文件未找到</p>
        <p><a href="/">返回主页</a></p>
        </body>
        </html>
        """, 404

@app.route('/api/predict_single', methods=['POST'])
def predict_single():
    """功能11 - 单个音频文件的实时预测接口"""
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 创建临时目录（如果不存在）
        temp_wav_dir = 'temp_wav'
        os.makedirs(temp_wav_dir, exist_ok=True)
        
        # 保存文件到临时目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"realtime_{timestamp}.wav"
        filepath = os.path.join(temp_wav_dir, filename)
        file.save(filepath)
        
        try:
            # 初始化配置和预测系统（与功能11相同）
            config_manager = ConfigManager()
            
            # 检查Seq2Seq模型
            seq2seq_model_file = "best_model_PIN_Dodonew_v2.pth"

            if not os.path.exists(seq2seq_model_file):
                os.remove(filepath)
                return jsonify({'error': 'Seq2Seq模型文件不存在'}), 500
            
            # 创建预测系统实例
            prediction_system = EnhancedPredictionSystem(
                config_manager, 
                seq2seq_model_file
            )
            
            # 执行功能11的预测
            result = prediction_system.predict_with_enhanced_masks(
                filepath,
                top_k=10,
                verbose=False,
                compare_basic=True
            )
            
            # 提取结果
            if result and 'accuracy_stats' in result:
                acc_stats = result['accuracy_stats']
                
                # 构建返回数据
                response = {
                    'success': True,
                    'predicted_sequence': acc_stats.get('advanced_model_prediction', ''),
                    'top_candidates': []
                }
                
                # 添加高级模型的候选结果
                if 'advanced_s2s_results' in result:
                    for i, candidate in enumerate(result['advanced_s2s_results'][:10]):
                        response['top_candidates'].append({
                            'rank': i + 1,
                            'sequence': candidate.get('text', ''),
                            'score': float(candidate.get('score', 0.0)),
                            'source': candidate.get('source', 'advanced')
                        })
                
                # 如果没有高级结果，使用声音模型结果
                elif 'sound_model_all_candidates' in result:
                    for i, (seq, conf) in enumerate(result['sound_model_all_candidates'][:10]):
                        response['top_candidates'].append({
                            'rank': i + 1,
                            'sequence': seq,
                            'score': float(conf),
                            'source': 'sound_model'
                        })
                
                return jsonify(response)
            else:
                return jsonify({'error': '预测失败，未返回有效结果'}), 500
                
        finally:
            # 清理临时文件
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            
    except Exception as e:
        # 确保清理临时文件
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/api/get_model_dirs', methods=['GET'])
def get_model_dirs():
    """获取可用的模型目录"""
    try:
        dirs = []
        # 检查常见的模型目录
        for dir_name in os.listdir('.'):
            if os.path.isdir(dir_name) and dir_name.endswith('_models'):
                dirs.append(dir_name)
        
        return jsonify({'dirs': dirs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_single_func11', methods=['POST'])
def predict_single_func11():
    """功能11 - 使用高级分析页面设置的模型目录进行预测"""
    start_time = time.time()
    
    try:
        # 获取参数
        file = request.files['file']
        model_dir = request.form.get('model_dir', 'models')  # 从前端传入
        expected_sequence = request.form.get('expected_sequence', '').strip()
        
        # 创建临时目录
        temp_wav_dir = 'temp_wav'
        os.makedirs(temp_wav_dir, exist_ok=True)
        
        # 保存文件（使用简单命名）
        filename = secure_filename(file.filename)  # 前端已经命名为 预期序列.wav
        filepath = os.path.join(temp_wav_dir, filename)
        file.save(filepath)
        
        try:
            # 初始化配置
            config_manager = ConfigManager()
            
            # 创建预测系统（使用指定的模型目录）
            seq2seq_model_file = "best_model_PIN_Dodonew_v2.pth"
            if not os.path.exists(seq2seq_model_file):
                return jsonify({'error': 'Seq2Seq模型文件不存在'}), 500
            
            # 创建预测系统时指定模型目录
            config_manager.set("paths.model_dir", model_dir)
            prediction_system = EnhancedPredictionSystem(config_manager, seq2seq_model_file)
            

            # 执行功能11预测（不需要传递expected_sequence，函数会从文件名自动提取）
            result = prediction_system.predict_with_enhanced_masks(
                filepath,
                top_k=10,
                verbose=False,
                compare_basic=True
            )
                        
            # 构建返回数据
            response = {
                'success': True,
                'filename': filename,
                'expected_sequence': expected_sequence,
                'processing_time': time.time() - start_time
            }
            
            # 提取声音模型候选
            if 'sound_model_all_candidates' in result:
                response['sound_model_candidates'] = []
                for seq, conf in result['sound_model_all_candidates'][:10]:
                    response['sound_model_candidates'].append({
                        'sequence': seq,
                        'confidence': float(conf)
                    })
            
            # 提取纯Seq2Seq预测
            if 'seq2seq_full_mask' in result:
                response['full_mask'] = result['seq2seq_full_mask'].get('mask', '')
                response['seq2seq_prediction'] = result['seq2seq_full_mask'].get('prediction', '')
            else:
                # 如果没有seq2seq_full_mask，尝试从accuracy_stats获取
                if 'accuracy_stats' in result:
                    response['seq2seq_prediction'] = result['accuracy_stats'].get('pure_seq2seq_prediction', '')
                    response['full_mask'] = '￥' * len(expected_sequence) if expected_sequence else '￥￥￥￥￥￥'

            # 提取高级模型预测
            if 'advanced_s2s_results' in result:
                response['advanced_predictions'] = []
                for item in result['advanced_s2s_results'][:10]:
                    response['advanced_predictions'].append({
                        'text': item.get('text', ''),
                        'score': float(item.get('score', 0)),
                        'source': item.get('source', ''),
                        'mask': item.get('mask', '')
                    })
            elif 'advanced' in result:
                # 兼容旧格式
                response['advanced_predictions'] = []
                for idx, item in enumerate(result['advanced'][:10]):
                    response['advanced_predictions'].append({
                        'text': item.get('text', ''),
                        'score': float(item.get('overall_score', item.get('score', 0))),
                        'source': item.get('template_name', item.get('source', '')),
                        'mask': item.get('mask', '')
                    })

            # 在提取高级模型预测后，添加掩码模板信息
            # 提取掩码模板（与func12保持一致）
            if 'used_masks' in result:
                response['maskTemplates'] = []
                for mask_info in result['used_masks']:
                    response['maskTemplates'].append({
                        'template': mask_info.get('template', ''),
                        'type': mask_info.get('type', '未知类型'),
                        'maskCount': mask_info.get('maskCount', 0),
                        'score': mask_info.get('score', 0)
                    })
                print(f"返回 {len(response['maskTemplates'])} 个掩码模板到前端")
            
            # 提取准确率对比
            if 'accuracy_stats' in result:
                stats = result['accuracy_stats']
                
                # 计算实际的准确率
                def calculate_char_accuracy(expected, predicted):
                    if not expected or not predicted:
                        return 0
                    correct = sum(1 for i in range(min(len(expected), len(predicted))) 
                                if expected[i] == predicted[i])
                    return round(correct / len(expected) * 100, 1)
                
                sound_prediction = stats.get('sound_model_prediction', '')
                seq2seq_prediction = stats.get('pure_seq2seq_prediction', '')
                advanced_prediction = stats.get('advanced_model_prediction', '')
                
                response['accuracy_comparison'] = {
                    'sound_model': {
                        'prediction': sound_prediction,
                        'char_accuracy': calculate_char_accuracy(expected_sequence, sound_prediction),
                        'rank': stats.get('sound_model_best_rank', -1)
                    },
                    'seq2seq': {
                        'prediction': seq2seq_prediction,
                        'char_accuracy': calculate_char_accuracy(expected_sequence, seq2seq_prediction),
                        'sequence_accuracy': 100 if seq2seq_prediction == expected_sequence else 0
                    },
                    'advanced': {
                        'prediction': advanced_prediction,
                        'char_accuracy': calculate_char_accuracy(expected_sequence, advanced_prediction),
                        'sequence_accuracy': 100 if advanced_prediction == expected_sequence else 0
                    }
                }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'预测失败: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500
    
@app.route('/api/upload_temp_wav', methods=['POST'])
def upload_temp_wav():
    """上传WAV文件到temp_wav目录"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 创建temp_wav目录
        temp_wav_dir = 'temp_wav'
        os.makedirs(temp_wav_dir, exist_ok=True)
        
        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_wav_dir, filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': filepath
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_temp_file', methods=['POST'])
def delete_temp_file():
    """删除指定的临时文件"""
    try:
        data = request.json
        filename = data.get('filename', '')
        
        if not filename:
            return jsonify({'error': '未指定文件名'}), 400
        
        filepath = os.path.join('temp_wav', filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': f'已删除文件: {filename}'})
        else:
            return jsonify({'success': False, 'message': '文件不存在'})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
   print("启动键盘声音识别系统Web服务（高级预测模式）...")
   print("访问 http://localhost:5000 来使用界面")
   print("API状态: http://localhost:5000/api/status")
   
   # 生产环境配置
   app.run(
       host='0.0.0.0',
       port=5000,
       debug=False,  # 关闭调试模式以提升性能
       threaded=True
   )