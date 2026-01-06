"""
MA20趋势跟踪策略 - 简化可视化工具
生成基础图表展示策略表现
说明：
- 作用：基于最新的交易记录文件（results/trades_*.csv）生成核心PNG图表，含盈亏分布、累计盈亏、策略综合分析与月度表现。
- 输入/依赖：results/trades_*.csv（由回测脚本生成）；未找到文件会报错并退出。
- 输出：在 results 目录生成多张 PNG 图片；会额外生成一张示例图。
- 适用场景：已完成一次回测后，快速生成可视化图表以分析表现。
- 参考代码：读取 trades 文件见 simple_visualization.py:L31-L41 ，保存图像见 simple_visualization.py:L126-L133 、 simple_visualization.py:L161-L167 、 simple_visualization.py:L199-L205
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import logging
from config import get_paths

# 设置中文字体和日志
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_visualization():
    """创建简化可视化"""
    logger.info("创建简化可视化报告...")
    
    # 创建保存目录（统一使用配置路径）
    paths = get_paths()
    save_dir = paths['results_dir']
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 读取最新的交易数据
    import glob
    trade_files = glob.glob(os.path.join(save_dir, 'trades_*.csv'))
    
    if not trade_files:
        logger.error("未找到交易数据文件")
        return
    
    latest_trade_file = max(trade_files, key=os.path.getctime)
    logger.info(f"使用交易文件: {latest_trade_file}")
    
    trades_df = pd.read_csv(latest_trade_file)
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    
    # 筛选有盈亏的交易
    trades_with_pnl = trades_df[trades_df['pnl'].notna()].copy()
    
    if len(trades_with_pnl) == 0:
        logger.error("没有有效的盈亏数据")
        return
    
    # 1. 盈亏分布直方图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    pnls = trades_with_pnl['pnl']
    plt.hist(pnls, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='盈亏平衡点')
    plt.axvline(x=pnls.mean(), color='orange', linestyle='--', 
                label=f'平均值: {pnls.mean():.0f}')
    plt.title('盈亏分布直方图', fontsize=14, fontweight='bold')
    plt.xlabel('盈亏 (CNY)')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 盈亏时间序列
    plt.subplot(2, 2, 2)
    colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
    plt.scatter(range(len(pnls)), pnls, c=colors, alpha=0.7, s=50)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.title('盈亏时间序列', fontsize=14, fontweight='bold')
    plt.xlabel('交易序号')
    plt.ylabel('盈亏 (CNY)')
    plt.grid(True, alpha=0.3)
    
    # 3. 累计盈亏
    plt.subplot(2, 2, 3)
    cumulative_pnl = pnls.cumsum()
    plt.plot(range(len(cumulative_pnl)), cumulative_pnl, 
             color='darkblue', linewidth=2)
    plt.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 
                     alpha=0.3, color='lightblue')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.title('累计盈亏曲线', fontsize=14, fontweight='bold')
    plt.xlabel('交易序号')
    plt.ylabel('累计盈亏 (CNY)')
    plt.grid(True, alpha=0.3)
    
    # 4. 盈亏统计
    plt.subplot(2, 2, 4)
    
    # 计算统计指标
    win_trades = pnls[pnls > 0]
    loss_trades = pnls[pnls < 0]
    
    stats = {
        '总交易': len(pnls),
        '盈利': len(win_trades),
        '亏损': len(loss_trades),
        '胜率': f"{len(win_trades)/len(pnls)*100:.1f}%"
    }
    
    # 创建文本显示
    plt.text(0.1, 0.8, '交易统计', fontsize=16, fontweight='bold', 
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"总交易次数: {stats['总交易']}", fontsize=12, 
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"盈利交易: {stats['盈利']}", fontsize=12, 
             color='green', transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"亏损交易: {stats['亏损']}", fontsize=12, 
             color='red', transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f"胜率: {stats['胜率']}", fontsize=12, 
             transform=plt.gca().transAxes)
    
    if len(win_trades) > 0:
        plt.text(0.1, 0.2, f"平均盈利: {win_trades.mean():.0f}", fontsize=12, 
                 color='green', transform=plt.gca().transAxes)
    if len(loss_trades) > 0:
        plt.text(0.1, 0.1, f"平均亏损: {loss_trades.mean():.0f}", fontsize=12, 
                 color='red', transform=plt.gca().transAxes)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'strategy_analysis_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"策略分析图已保存: {save_path}")
    
    # 2. 创建单独的盈亏分布图
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(pnls, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='盈亏平衡')
    plt.axvline(x=pnls.mean(), color='orange', linestyle='--', linewidth=2, 
                label=f'均值: {pnls.mean():.0f}')
    plt.title('盈亏分布', fontsize=14, fontweight='bold')
    plt.xlabel('盈亏 (CNY)')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 盈利vs亏损对比
    if len(win_trades) > 0 and len(loss_trades) > 0:
        plt.boxplot([win_trades, abs(loss_trades)], 
                   labels=['盈利', '亏损(绝对值)'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', alpha=0.7))
        plt.title('盈利vs亏损分布', fontsize=14, fontweight='bold')
        plt.ylabel('金额 (CNY)')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, '数据不足', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=14)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, f'pnl_distribution_{timestamp}.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"盈亏分布图已保存: {save_path2}")
    
    # 3. 创建累计盈亏图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 1, 1)
    cumulative_pnl = pnls.cumsum()
    
    # 创建颜色渐变效果
    colors = ['green' if x >= 0 else 'red' for x in cumulative_pnl]
    
    plt.plot(range(len(cumulative_pnl)), cumulative_pnl, 
             color='darkblue', linewidth=2, label='累计盈亏')
    
    # 填充颜色
    for i in range(len(cumulative_pnl)-1):
        plt.fill_between([i, i+1], [cumulative_pnl.iloc[i], cumulative_pnl.iloc[i+1]], 
                        alpha=0.3, color=colors[i])
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 标记最终值
    final_value = cumulative_pnl.iloc[-1]
    plt.scatter(len(cumulative_pnl)-1, final_value, 
               color='red' if final_value < 0 else 'green', s=100, 
               marker='o', label=f'最终值: {final_value:,.0f}')
    
    plt.title('累计盈亏趋势', fontsize=16, fontweight='bold')
    plt.xlabel('交易序号')
    plt.ylabel('累计盈亏 (CNY)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path3 = os.path.join(save_dir, f'cumulative_pnl_{timestamp}.png')
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"累计盈亏图已保存: {save_path3}")
    
    # 4. 创建月度表现分析（如果有足够数据）
    if len(trades_with_pnl) > 10:
        plt.figure(figsize=(12, 8))
        
        # 按月份分组
        trades_with_pnl['month'] = trades_with_pnl['date'].dt.to_period('M')
        monthly_stats = trades_with_pnl.groupby('month').agg({
            'pnl': ['sum', 'count', 'mean']
        })
        monthly_stats.columns = ['total_pnl', 'trade_count', 'avg_pnl']
        
        plt.subplot(2, 1, 1)
        monthly_pnls = monthly_stats['total_pnl']
        colors = ['green' if x >= 0 else 'red' for x in monthly_pnls]
        
        plt.bar(range(len(monthly_pnls)), monthly_pnls, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.title('月度盈亏', fontsize=14, fontweight='bold')
        plt.ylabel('月度盈亏 (CNY)')
        plt.xticks(range(len(monthly_pnls)), 
                  [str(month) for month in monthly_pnls.index], rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        monthly_counts = monthly_stats['trade_count']
        plt.plot(range(len(monthly_counts)), monthly_counts, 
                marker='o', linewidth=2, markersize=6, color='steelblue')
        plt.title('月度交易次数', fontsize=14, fontweight='bold')
        plt.ylabel('交易次数')
        plt.xlabel('月份')
        plt.xticks(range(len(monthly_counts)), 
                  [str(month) for month in monthly_counts.index], rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path4 = os.path.join(save_dir, f'monthly_analysis_{timestamp}.png')
        plt.savefig(save_path4, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"月度分析图已保存: {save_path4}")
    
    logger.info("🎉 可视化报告生成完成!")
    logger.info(f"生成的图表文件:")
    logger.info(f"1. 策略综合分析图: {save_path}")
    logger.info(f"2. 盈亏分布对比图: {save_path2}")
    logger.info(f"3. 累计盈亏趋势图: {save_path3}")
    if len(trades_with_pnl) > 10:
        logger.info(f"4. 月度表现分析图: {save_path4}")


def show_sample_charts():
    """显示示例图表"""
    logger.info("创建示例图表...")
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 1. 示例盈亏分布
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    # 模拟盈亏数据
    sample_pnls = np.concatenate([
        np.random.normal(2000, 1000, 60),  # 盈利
        np.random.normal(-1000, 500, 40)   # 亏损
    ])
    
    plt.hist(sample_pnls, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='盈亏平衡')
    plt.axvline(x=sample_pnls.mean(), color='orange', linestyle='--', 
                label=f'均值: {sample_pnls.mean():.0f}')
    plt.title('示例: 盈亏分布', fontsize=12, fontweight='bold')
    plt.xlabel('盈亏')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 示例累计盈亏
    plt.subplot(1, 3, 2)
    cumulative = np.cumsum(sample_pnls)
    plt.plot(cumulative, color='darkblue', linewidth=2)
    plt.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='lightblue')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.title('示例: 累计盈亏', fontsize=12, fontweight='bold')
    plt.xlabel('交易序号')
    plt.ylabel('累计盈亏')
    plt.grid(True, alpha=0.3)
    
    # 3. 示例交易信号
    plt.subplot(1, 3, 3)
    # 模拟价格数据
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    prices = 4000 + np.cumsum(np.random.normal(0, 20, 50))
    
    plt.plot(dates, prices, color='black', linewidth=1.5, label='价格')
    
    # 添加模拟信号
    buy_dates = dates[::10]
    buy_prices = prices[::10]
    sell_dates = dates[5::10]
    sell_prices = prices[5::10]
    
    plt.scatter(buy_dates, buy_prices, color='green', s=100, marker='^', 
                label='买入信号', zorder=5)
    plt.scatter(sell_dates, sell_prices, color='red', s=100, marker='v', 
                label='卖出信号', zorder=5)
    
    plt.title('示例: 交易信号', fontsize=12, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_path = 'results/sample_charts.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"示例图表已保存: {save_path}")


if __name__ == "__main__":
    # 首先创建可视化
    create_simple_visualization()
    
    # 然后显示示例
    show_sample_charts()
    
    logger.info("✅ 所有可视化任务完成!")
    logger.info("📊 请查看 results 目录下的 PNG 图片文件!")
