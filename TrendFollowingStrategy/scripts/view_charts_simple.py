"""
MA20趋势跟踪策略 - 图表查看器
简单查看生成的可视化图表
说明：
- 作用：列出 results 目录下所有 PNG 图表，并生成一个轻量版 HTML 查看器（charts_viewer.html）。
- 输入/依赖：results 下的 PNG 图片；如无图片会提示先运行 simple_visualization。
- 输出：终端列出图表信息，生成 charts_viewer.html。
- 适用场景：快速浏览已有图表文件，轻量展示，无报告解读。
- 参考代码：图表列举与类型识别见 view_charts_simple.py:L27-L48 ，HTML 生成见 view_charts_simple.py:L110-L125 、 view_charts_simple.py:L246-L253
"""

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import glob
from datetime import datetime
from config import get_paths

def show_charts():
    """显示所有图表文件"""
    print("=" * 80)
    print("                    📊 MA20趋势跟踪策略 - 可视化图表")
    print("=" * 80)
    
    # 获取结果目录
    results_dir = get_paths()['results_dir']
    
    if not os.path.exists(results_dir):
        print("❌ 结果目录不存在!")
        return
    
    # 查找所有PNG图片文件
    png_files = glob.glob(os.path.join(results_dir, '*.png'))
    
    if not png_files:
        print("❌ 未找到PNG图片文件!")
        print("💡 请先运行: python simple_visualization.py")
        return
    
    print(f"📁 找到 {len(png_files)} 个可视化图表:")
    print()
    
    # 按时间排序
    png_files.sort(key=os.path.getctime, reverse=True)
    
    # 显示文件信息
    chart_info = {
        'strategy_analysis': '📈 策略综合分析图',
        'pnl_distribution': '📊 盈亏分布对比图', 
        'cumulative_pnl': '📈 累计盈亏趋势图',
        'monthly_analysis': '📅 月度表现分析图',
        'sample_charts': '🎨 示例图表'
    }
    
    for i, png_file in enumerate(png_files, 1):
        filename = os.path.basename(png_file)
        file_size = os.path.getsize(png_file)
        create_time = datetime.fromtimestamp(os.path.getctime(png_file))
        
        # 确定图表类型
        chart_type = "其他图表"
        for key, description in chart_info.items():
            if key in filename:
                chart_type = description
                break
        
        print(f"{i:2d}. {chart_type}")
        print(f"    📄 文件名: {filename}")
        print(f"    📅 创建时间: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    📊 文件大小: {file_size/1024:.1f} KB")
        print()
    
    # 显示最新回测结果
    txt_files = glob.glob(os.path.join(results_dir, 'backtest_report_*.txt'))
    if txt_files:
        latest_txt = max(txt_files, key=os.path.getctime)
        print("📋 最新回测报告:")
        print(f"   📄 {os.path.basename(latest_txt)}")
        
        # 读取并显示关键信息
        try:
            with open(latest_txt, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print("   📊 关键指标:")
            lines = content.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['总收益率', '胜率', '盈亏比', '最终资金']):
                    print(f"      {line.strip()}")
        except Exception as e:
            print(f"   ❌ 读取报告失败: {e}")
    
    print()
    print("=" * 80)
    print("💡 如何查看这些图表:")
    print()
    print("方法1: 文件管理器查看")
    print("   • 打开文件管理器")
    print("   • 导航到: results 目录")
    print("   • 双击 PNG 图片文件")
    print()
    print("方法2: 命令行查看 (Windows)")
    print("   • 在文件资源管理器中输入: cmd")
    print("   • 执行: start results\\图片文件名.png")
    print()
    print("方法3: Python查看 (需要额外库)")
    print("   • 安装: pip install pillow")
    print("   • 使用Python脚本打开图片")
    print()
    print("📁 完整路径:")
    print(f"   {os.path.abspath(results_dir)}")
    print()
    print("=" * 80)

def create_simple_html_viewer():
    """创建简单的HTML查看器"""
    results_dir = get_paths()['results_dir']
    png_files = glob.glob(os.path.join(results_dir, '*.png'))
    
    if not png_files:
        print("❌ 未找到图片文件")
        return
    
    # 筛选策略相关的图片
    strategy_images = [f for f in png_files if 'sample' not in os.path.basename(f)]
    
    if not strategy_images:
        print("❌ 未找到策略分析图片")
        return
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MA20趋势跟踪策略 - 可视化图表</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .chart-section {{
            margin-bottom: 40px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .chart-container img:hover {{
            transform: scale(1.02);
        }}
        .info {{
            background-color: #e8f4f8;
            border: 1px solid #3498db;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }}
        .timestamp {{
            text-align: right;
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 MA20趋势跟踪策略 - 可视化图表</h1>
        
        <div class="info">
            <strong>📋 图表说明:</strong>
            <p>以下图表展示了MA20趋势跟踪策略的回测结果分析，包括盈亏分布、累计表现、月度分析等关键可视化内容。</p>
        </div>
    """
    
    # 图表标题映射
    chart_titles = {
        'strategy_analysis': '📈 策略综合分析',
        'pnl_distribution': '📊 盈亏分布分析',
        'cumulative_pnl': '📈 累计盈亏趋势',
        'monthly_analysis': '📅 月度表现分析'
    }
    
    # 为每个图片添加部分
    for img_file in strategy_images:
        filename = os.path.basename(img_file)
        
        # 确定图表类型
        chart_type = None
        for key, title in chart_titles.items():
            if key in filename:
                chart_type = key
                break
        
        if chart_type:
            html_content += f"""
        <div class="chart-section">
            <div class="chart-title">{chart_titles[chart_type]}</div>
            <div class="chart-container">
                <img src="{filename}" alt="{chart_titles[chart_type]}">
            </div>
        </div>
            """
    
    # 添加结尾
    html_content += f"""
        <div class="timestamp">
            报告生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
        </div>
    </div>
</body>
</html>
    """
    
    # 保存HTML文件
    html_file = os.path.join(results_dir, 'charts_viewer.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML图表查看器已创建: {html_file}")
    print(f"📁 文件路径: {os.path.abspath(html_file)}")
    return html_file

if __name__ == "__main__":
    print("🎨 开始查看可视化图表...")
    print()
    
    # 显示图表列表
    show_charts()
    print()
    
    # 创建HTML查看器
    print("🌐 创建HTML图表查看器...")
    try:
        html_file = create_simple_html_viewer()
        print(f"\n🎉 完成!")
        print(f"   请用浏览器打开: {html_file}")
        print(f"   或直接双击打开 charts_viewer.html")
    except Exception as e:
        print(f"\n❌ HTML查看器创建失败: {e}")
        print("   但图片文件已经生成，可以直接查看!")
    
    print("\n" + "=" * 80)
    print("🎯 建议操作:")
    print("1. 打开 results 目录")
    print("2. 双击 charts_viewer.html 用浏览器查看")
    print("3. 或直接用图片查看器打开 PNG 文件")
    print("=" * 80)
