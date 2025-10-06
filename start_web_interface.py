# start_web_interface.py - 启动Web界面的脚本
import os
import sys
import webbrowser
import time
import subprocess
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = ['flask', 'flask-cors']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_project_structure():
    """检查项目结构"""
    required_files = [
        'config_manager.py',
        'keystroke_recognition.py',
        'audio_processing.py',
        'feature_extraction.py',
        'keystroke_model.py',
        'data_manager.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ 缺少以下核心文件:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n请确保所有项目文件都在当前目录中")
        return False
    
    return True

def create_html_file():
    """创建HTML文件（如果不存在）"""
    html_file = "model_test_interface.html"
    if not os.path.exists(html_file):
        print(f"⚠️  HTML界面文件 {html_file} 不存在")
        print("请将HTML文件保存到当前目录，或者使用以下URL直接访问API:")
        print("http://localhost:5000/api/status")
        return False
    return True

def start_server():
    """启动Flask服务器"""
    print("🚀 启动键盘声音识别Web界面...")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        return False
        
    # 检查项目结构
    if not check_project_structure():
        return False
        
    # 检查HTML文件
    create_html_file()
    
    print("✅ 环境检查完成")
    print("🌐 启动Web服务器...")
    
    try:
        # 导入并启动Flask应用
        from web_server import app
        
        print("\n" + "=" * 60)
        print("🎉 键盘声音识别系统已启动!")
        print("=" * 60)
        print("📱 Web界面: http://localhost:5000")
        print("🔧 API状态: http://localhost:5000/api/status")
        print("📚 API文档: http://localhost:5000/api")
        print("=" * 60)
        print("💡 使用说明:")
        print("1. 在浏览器中打开 http://localhost:5000")
        print("2. 配置模型目录和测试数据目录")
        print("3. 选择是否启用多候选预测")
        print("4. 点击'开始测试'按钮")
        print("5. 查看实时进度和详细结果")
        print("=" * 60)
        print("⏹️  按 Ctrl+C 停止服务")
        print()
        
        # 尝试自动打开浏览器
        try:
            time.sleep(1)  # 等待服务器启动
            webbrowser.open('http://localhost:5000')
            print("🌐 已自动打开浏览器")
        except:
            print("ℹ️  请手动在浏览器中打开 http://localhost:5000")
        
        # 启动Flask应用
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # 生产环境关闭debug
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ 导入web_server模块失败: {e}")
        print("请确保 web_server.py 文件存在并且所有依赖都已正确安装")
        return False
    except KeyboardInterrupt:
        print("\n👋 用户停止了服务")
        return True
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")
        return False

def show_help():
    """显示帮助信息"""
    print("🎹 键盘声音识别系统 - Web界面")
    print("=" * 50)
    print("\n📋 功能特性:")
    print("• 🎯 模型性能测试与评估")
    print("• 📊 实时进度显示")
    print("• 🎪 多候选预测支持")
    print("• 📈 详细的测试报告")
    print("• 🌐 友好的Web界面")
    
    print("\n🛠️ 使用步骤:")
    print("1. 确保已安装所有依赖")
    print("   pip install flask flask-cors")
    print("2. 准备好训练的模型文件")
    print("3. 准备测试数据（WAV格式）")
    print("4. 运行此脚本启动服务")
    print("5. 在浏览器中访问界面")
    
    print("\n📁 项目结构:")
    print("project/")
    print("├── start_web_interface.py    # 启动脚本")
    print("├── web_server.py            # Flask后端")
    print("├── model_test_interface.html # Web界面")
    print("├── keystroke_recognition.py  # 核心识别模块")
    print("├── config_manager.py        # 配置管理")
    print("├── models/                  # 训练好的模型")
    print("└── test/                    # 测试数据")
    
    print("\n🔧 故障排除:")
    print("• 如果模块导入失败，检查Python路径")
    print("• 如果端口被占用，修改web_server.py中的端口号")
    print("• 如果测试失败，检查模型文件和测试数据路径")
    print("• 查看控制台日志获取详细错误信息")
    
    print("\n📞 技术支持:")
    print("• 检查所有核心Python文件是否存在")
    print("• 确保模型已经训练完成")
    print("• 验证测试数据格式正确")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
    else:
        success = start_server()
        if not success:
            print("\n" + "=" * 50)
            print("❌ 启动失败！请检查上述错误信息")
            print("💡 运行 'python start_web_interface.py --help' 查看详细帮助")
            sys.exit(1)