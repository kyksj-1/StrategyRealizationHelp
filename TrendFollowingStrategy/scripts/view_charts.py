"""
MA20趋势跟踪策略 - 可视化报告查看器
查看和分析生成的图表
说明：
- 作用：可视化报告查看与汇总，总览所有 PNG 图表并解析最新回测报告中的关键指标；可生成更丰富样式的 HTML 报告（visualization_report.html）。
- 输入/依赖：results 下的 PNG 图片与 backtest_report_*.txt。
- 输出：终端展示图表清单与回测关键指标；生成 visualization_report.html。
- 适用场景：需要更正式的图文报告输出，便于分享或归档。
- 参考代码：图表枚举与报告摘要见 view_charts.py:L27-L33 、 view_charts.py:L67-L83 ，HTML 生成见 view_charts.py:L243-L250
"""

import os
import glob
from datetime import datetime
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import get_paths

def show_visualization_report():
    """显示可视化报告"""
    print("=" * 80)
    print("                    📊 MA20趋势跟踪策略 - 可视化报告")
    print("=" * 80)
    
    # 获取结果目录
    results_dir = get_paths()['results_dir']
    
    if not os.path.exists(results_dir):
        print("❌ 结果目录不存在!")
        return
    
    # 查找所有图片文件
    image_files = glob.glob(os.path.join(results_dir, '*.png'))
    
    if not image_files:
        print("❌ 未找到可视化图片!")
        return
    
    print(f"📁 找到 {len(image_files)} 个可视化文件:")
    print()
    
    # 按时间排序
    image_files.sort(key=os.path.getctime, reverse=True)
    
    # 显示文件信息
    for i, img_file in enumerate(image_files, 1):
        filename = os.path.basename(img_file)
        file_size = os.path.getsize(img_file)
        create_time = datetime.fromtimestamp(os.path.getctime(img_file))
        
        print(f"{i:2d}. 📈 {filename}")
        print(f"    📅 创建时间: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    📊 文件大小: {file_size/1024:.1f} KB")
        
        # 文件类型说明
        if 'strategy_analysis' in filename:
            print("    📝 类型: 策略综合分析图")
        elif 'pnl_distribution' in filename:
            print("    📊 类型: 盈亏分布对比图")
        elif 'cumulative_pnl' in filename:
            print("    📈 类型: 累计盈亏趋势图")
        elif 'monthly_analysis' in filename:
            print("    📅 类型: 月度表现分析图")
        elif 'sample_charts' in filename:
            print("    🎨 类型: 示例图表")
        else:
            print("    📋 类型: 其他图表")
        
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
                lines = f.readlines()
            
            print("   📊 关键指标:")
            for line in lines:
                if any(keyword in line for keyword in ['总收益率', '胜率', '盈亏比', '最终资金']):
                    print(f"      {line.strip()}")
        except Exception as e:
            print(f"   ❌ 读取报告失败: {e}")
    
    print()
    print("=" * 80)
    print("💡 如何查看这些图表:")
    print("1. 在文件管理器中打开 results 目录")
    print("2. 双击 PNG 图片文件即可查看")
    print("3. 或使用任何图片查看器/编辑器打开")
    print()
    print("📁 完整路径:", os.path.abspath(results_dir))
    print("=" * 80)

def create_html_report():
    """创建HTML可视化报告"""
    results_dir = get_paths()['results_dir']
    image_files = glob.glob(os.path.join(results_dir, '*.png'))
    
    if not image_files:
        print("❌ 未找到图片文件，无法创建HTML报告")
        return
    
    # 筛选策略相关的图片
    strategy_images = [f for f in image_files if 'sample' not in os.path.basename(f)]
    
    if not strategy_images:
        print("❌ 未找到策略分析图片")
        return
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MA20趋势跟踪策略 - 可视化报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
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
        }}
        .info-box {{
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
        .summary {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 MA20趋势跟踪策略 - 可视化分析报告</h1>
        
        <div class="info-box">
            <strong>📋 报告说明:</strong>
            <p>本报告基于MA20趋势跟踪策略的回测结果生成，包含策略的盈亏分布、累计表现、月度分析等关键可视化图表。</p>
        </div>
    """
    
    # 添加图表部分
    chart_titles = {
        'strategy_analysis': '📈 策略综合分析',
        'pnl_distribution': '📊 盈亏分布分析',
        'cumulative_pnl': '📈 累计盈亏趋势',
        'monthly_analysis': '📅 月度表现分析'
    }
    
    for img_file in strategy_images:
            filename = os.path.basename(img_file)
            
            # 确定图表类型
            chart_type = None
            for key in chart_titles:
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
    
    # 添加总结和时间戳
    html_content += f"""
        <div class="summary">
            <strong>💡 图表解读要点:</strong>
            <ul>
                <li><strong>盈亏分布图:</strong> 显示策略的盈利和亏损交易分布情况</li>
                <li><strong>累计盈亏曲线:</strong> 展示策略的整体资金变化趋势</li>
                <li><strong>月度分析图:</strong> 分析策略在不同月份的表现</li>
                <li><strong>交易统计:</strong> 提供胜率、平均盈亏等关键指标</li>
            </ul>
        </div>
        
        <div class="timestamp">
            报告生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
        </div>
    </div>
</body>
</html>
    """
    
    # 保存HTML文件
    html_file = os.path.join(results_dir, 'visualization_report.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML可视化报告已创建: {html_file}")
    print(f"📁 文件路径: {os.path.abspath(html_file)}")
    return html_file

if __name__ == "__main__":
    print("🎨 开始生成可视化报告...")
    
    # 显示可视化文件列表
    show_visualization_report()
    
    print()
    print("🌐 创建HTML可视化报告...")
    
    # 创建HTML报告
    try:
        html_file = create_html_report()
        print(f"\n🎉 可视化报告生成完成!")
        print(f"   您可以用浏览器打开: {html_file}")
    except Exception as e:
        print(f"\n❌ HTML报告生成失败: {e}")
        print("   但图片文件已经生成，可以直接查看!")
    
    print("\n" + "=" * 80)
    print("🎯 建议下一步操作:")
    print("1. 打开 results 目录查看所有图表")
    print("2. 用浏览器打开 visualization_report.html 查看完整报告")
    print("3. 分析图表中的策略表现特征")
    print("=" * 80)
