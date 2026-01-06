"""
MA20趋势跟踪策略 - 可视化工具
完善PNG图输出并生成HTML仪表盘
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob
import logging

from config import get_paths

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyVisualizer:
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'neutral': '#7f7f7f',
            'background': '#f8f9fa'
        }

    def create_comprehensive_report(self, data: pd.DataFrame, trades: pd.DataFrame,
                                   backtest_results: dict, save_dir: str = None, generate_html: bool = True):
        paths = get_paths()
        save_dir = save_dir or paths['results_dir']
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.plot_equity_curve(data, trades, backtest_results, save_dir, timestamp)
        self.plot_price_chart_with_signals(data, trades, save_dir, timestamp)
        self.plot_trade_distribution(trades, save_dir, timestamp)
        self.plot_monthly_performance(trades, save_dir, timestamp)
        self.plot_drawdown_analysis(data, trades, save_dir, timestamp)
        self.plot_trade_timing_analysis(trades, save_dir, timestamp)

        if generate_html:
            self.generate_html_dashboard(save_dir, timestamp)
        logger.info(f"可视化报告已保存到: {save_dir}")

    def plot_equity_curve(self, data: pd.DataFrame, trades: pd.DataFrame, backtest_results: dict, save_dir: str, timestamp: str):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, gridspec_kw={'height_ratios': [3, 1]})
        equity_curve = self._calculate_equity_curve(trades, backtest_results['initial_capital'])
        ax1.plot(equity_curve.index, equity_curve.values, color=self.colors['primary'], linewidth=2, label='权益曲线')
        ax1.axhline(y=backtest_results['initial_capital'], color=self.colors['neutral'], linestyle='--', alpha=0.7, label='初始资金线')
        max_equity = equity_curve.max(); min_equity = equity_curve.min()
        ax1.set_title('权益曲线与回撤分析', fontsize=16, fontweight='bold')
        ax1.set_ylabel('资金 (CNY)', fontsize=12); ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
        drawdown = self._calculate_drawdown(equity_curve)
        ax2.fill_between(drawdown.index, drawdown.values, 0, color=self.colors['danger'], alpha=0.3, label='回撤')
        ax2.set_ylabel('回撤 (%)', fontsize=12); ax2.set_xlabel('日期', fontsize=12); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); save_path = os.path.join(save_dir, f'equity_curve_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    def plot_price_chart_with_signals(self, data: pd.DataFrame, trades: pd.DataFrame, save_dir: str, timestamp: str):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(data.index, data['close'], color=self.colors['primary'], linewidth=1.5, label='收盘价')
        if 'ma20' in data.columns:
            ax1.plot(data.index, data['ma20'], color=self.colors['secondary'], linewidth=1.5, label='MA20')
        buy_signals = trades[trades['type'] == 'BUY']; sell_signals = trades[trades['type'] == 'SELL']
        for _, trade in buy_signals.iterrows():
            if trade['date'] in data.index:
                ax1.scatter(trade['date'], trade['price'], color=self.colors['success'], s=80, marker='^')
        for _, trade in sell_signals.iterrows():
            if trade['date'] in data.index:
                ax1.scatter(trade['date'], trade['price'], color=self.colors['danger'], s=80, marker='v')
        ax1.set_title('价格走势与交易信号', fontsize=16, fontweight='bold'); ax1.set_ylabel('价格 (CNY)', fontsize=12); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.bar(data.index, data['volume'], color=self.colors['neutral'], alpha=0.7)
        ax2.set_ylabel('成交量', fontsize=12); ax2.set_xlabel('日期', fontsize=12); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); save_path = os.path.join(save_dir, f'price_signals_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    def plot_trade_distribution(self, trades: pd.DataFrame, save_dir: str, timestamp: str):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        pnls = trades[trades['pnl'].notna()]['pnl']
        ax1 = axes[0, 0]; ax1.hist(pnls, bins=20, alpha=0.7, color=self.colors['primary']); ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('盈亏分布', fontweight='bold'); ax1.set_xlabel('盈亏 (CNY)'); ax1.set_ylabel('频次'); ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]; trades_with_pnl = trades[trades['pnl'].notna()]
        colors = [self.colors['success'] if pnl > 0 else self.colors['danger'] for pnl in trades_with_pnl['pnl']]
        ax2.scatter(trades_with_pnl.index, trades_with_pnl['pnl'], c=colors, alpha=0.7, s=50); ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('盈亏时间序列', fontweight='bold'); ax2.set_xlabel('交易序号'); ax2.set_ylabel('盈亏 (CNY)'); ax2.grid(True, alpha=0.3)
        ax3 = axes[1, 0]; win_pnls = pnls[pnls > 0]; loss_pnls = pnls[pnls < 0]
        if len(win_pnls) > 0 and len(loss_pnls) > 0:
            ax3.boxplot([win_pnls, loss_pnls], labels=['盈利', '亏损'], patch_artist=True, boxprops=dict(facecolor=self.colors['primary'], alpha=0.7))
        ax3.set_title('盈亏幅度箱线图', fontweight='bold'); ax3.set_ylabel('盈亏 (CNY)'); ax3.grid(True, alpha=0.3)
        ax4 = axes[1, 1]; trades_with_date = trades.copy(); trades_with_date['date_only'] = trades_with_date['date'].dt.date
        daily_trades = trades_with_date.groupby('date_only').size()
        if len(daily_trades) > 1:
            ax4.plot(daily_trades.index, daily_trades.values, color=self.colors['primary'], linewidth=1.5)
            ax4.fill_between(daily_trades.index, daily_trades.values, alpha=0.3, color=self.colors['primary'])
        ax4.set_title('日交易频率', fontweight='bold'); ax4.set_xlabel('日期'); ax4.set_ylabel('交易次数'); ax4.grid(True, alpha=0.3)
        plt.tight_layout(); save_path = os.path.join(save_dir, f'trade_distribution_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    def plot_monthly_performance(self, trades: pd.DataFrame, save_dir: str, timestamp: str):
        trades_with_pnl = trades[trades['pnl'].notna()].copy()
        trades_with_pnl['month'] = trades_with_pnl['date'].dt.to_period('M')
        monthly_stats = trades_with_pnl.groupby('month').agg({'pnl': ['sum', 'count', 'mean']}).round(2)
        monthly_stats.columns = ['total_pnl', 'trade_count', 'avg_pnl']
        monthly_wr = trades_with_pnl.groupby('month').apply(lambda x: (x['pnl'] > 0).mean() * 100).round(1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.imshow(monthly_stats['total_pnl'].values.reshape(-1, 1), cmap='RdYlGn', aspect='auto'); ax1.set_title('月度盈亏热力图', fontsize=14, fontweight='bold')
        ax1.set_yticks(range(len(monthly_stats))); ax1.set_yticklabels([str(p) for p in monthly_stats.index]); plt.colorbar(ax=ax1.imshow(monthly_stats['total_pnl'].values.reshape(-1, 1), cmap='RdYlGn', aspect='auto'))
        im2 = ax2.imshow(monthly_wr.values.reshape(-1, 1), cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax2.set_title('月度胜率热力图', fontsize=14, fontweight='bold'); ax2.set_yticks(range(len(monthly_wr))); ax2.set_yticklabels([str(p) for p in monthly_wr.index]); plt.colorbar(im2, ax=ax2, label='胜率 (%)')
        plt.tight_layout(); save_path = os.path.join(save_dir, f'monthly_heatmap_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    def plot_drawdown_analysis(self, data: pd.DataFrame, trades: pd.DataFrame, save_dir: str, timestamp: str):
        equity_curve = self._calculate_equity_curve(trades, 100000)
        drawdown = self._calculate_drawdown(equity_curve)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax1 = axes[0, 0]; ax1.fill_between(drawdown.index, drawdown.values, 0, color=self.colors['danger'], alpha=0.3)
        ax1.plot(drawdown.index, drawdown.values, color=self.colors['danger'], linewidth=1); ax1.set_title('回撤时间序列', fontweight='bold'); ax1.set_ylabel('回撤 (%)'); ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]; ax2.hist(drawdown[drawdown < 0], bins=30, alpha=0.7, color=self.colors['danger']); ax2.set_title('回撤分布', fontweight='bold'); ax2.set_xlabel('回撤 (%)'); ax2.set_ylabel('频次'); ax2.grid(True, alpha=0.3)
        ax3 = axes[1, 0]; recovery_times = self._calculate_recovery_times(equity_curve, drawdown)
        if recovery_times: ax3.hist(recovery_times, bins=15, alpha=0.7, color=self.colors['primary'])
        ax3.set_title('回撤恢复时间分布', fontweight='bold'); ax3.set_xlabel('恢复时间 (天)'); ax3.set_ylabel('频次'); ax3.grid(True, alpha=0.3)
        ax4 = axes[1, 1]; monthly_returns = equity_curve.resample('M').last().pct_change() * 100; monthly_drawdowns = drawdown.resample('M').min()
        if len(monthly_returns) > 1 and len(monthly_drawdowns) > 1: ax4.scatter(monthly_drawdowns, monthly_returns, alpha=0.7, color=self.colors['primary'])
        ax4.set_title('月度回撤与收益关系', fontweight='bold'); ax4.set_xlabel('月度最大回撤 (%)'); ax4.set_ylabel('月度收益 (%)'); ax4.grid(True, alpha=0.3)
        plt.tight_layout(); save_path = os.path.join(save_dir, f'drawdown_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    def plot_trade_timing_analysis(self, trades: pd.DataFrame, save_dir: str, timestamp: str):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax1 = axes[0, 0]; trades_with_time = trades.copy(); trades_with_time['hour'] = trades_with_time['date'].dt.hour
        hour_counts = trades_with_time['hour'].value_counts().sort_index()
        ax1.bar(hour_counts.index, hour_counts.values, color=self.colors['primary'], alpha=0.7); ax1.set_title('交易时间分布（小时）', fontweight='bold'); ax1.set_xlabel('小时'); ax1.set_ylabel('交易次数'); ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]; trades_with_time['weekday'] = trades_with_time['date'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = trades_with_time['weekday'].value_counts().reindex(weekday_order, fill_value=0)
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        ax2.bar(range(len(weekday_counts)), weekday_counts.values, color=self.colors['secondary'], alpha=0.7)
        ax2.set_title('交易星期分布', fontweight='bold'); ax2.set_xlabel('星期'); ax2.set_ylabel('交易次数'); ax2.set_xticks(range(len(weekday_names))); ax2.set_xticklabels(weekday_names); ax2.grid(True, alpha=0.3)
        ax3 = axes[1, 0]
        if 'holding_days' in trades.columns:
            holding_days = trades[trades['holding_days'].notna()]['holding_days']
            ax3.hist(holding_days, bins=20, alpha=0.7, color=self.colors['success'])
        ax3.set_title('持仓时间分布', fontweight='bold'); ax3.set_xlabel('持仓天数'); ax3.set_ylabel('频次'); ax3.grid(True, alpha=0.3)
        ax4 = axes[1, 1]; trades_with_date = trades.copy(); trades_with_date['date_only'] = trades_with_date['date'].dt.date
        daily_trades = trades_with_date.groupby('date_only').size()
        if len(daily_trades) > 1:
            ax4.plot(daily_trades.index, daily_trades.values, color=self.colors['primary'], linewidth=1.5)
            ax4.fill_between(daily_trades.index, daily_trades.values, alpha=0.3, color=self.colors['primary'])
        ax4.set_title('日交易频率', fontweight='bold'); ax4.set_xlabel('日期'); ax4.set_ylabel('交易次数'); ax4.grid(True, alpha=0.3)
        plt.tight_layout(); save_path = os.path.join(save_dir, f'trade_timing_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    def _calculate_equity_curve(self, trades: pd.DataFrame, initial_capital: float) -> pd.Series:
        equity_curve = []; current_capital = initial_capital
        for _, trade in trades.iterrows():
            if 'pnl' in trade and pd.notna(trade['pnl']):
                current_capital += trade['pnl']
            equity_curve.append(current_capital)
        dates = trades[trades['pnl'].notna()]['date'].values
        return pd.Series(equity_curve, index=pd.to_datetime(dates))

    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        return drawdown

    def _calculate_recovery_times(self, equity_curve: pd.Series, drawdown: pd.Series) -> list:
        recovery_times = []; in_drawdown = False; start_date = None
        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True; start_date = date
            elif dd == 0 and in_drawdown and start_date:
                in_drawdown = False; recovery_times.append((date - start_date).days); start_date = None
        return recovery_times

    def generate_html_dashboard(self, save_dir: str, timestamp: str):
        png_files = sorted(glob.glob(os.path.join(save_dir, f"*_{timestamp}.png")))
        html_path = os.path.join(save_dir, "charts_viewer.html")
        parts = ["<html><head><meta charset='utf-8'><title>策略可视化报告</title>",
                 "<style>body{font-family:Arial;margin:20px;} img{max-width:100%;border:1px solid #ddd;margin:10px 0;} h2{margin-top:30px}</style>",
                 "</head><body><h1>策略可视化报告</h1>"]
        for f in png_files:
            title = os.path.basename(f).replace(f"_{timestamp}.png", "").replace("_", " ").title()
            parts.append(f"<h2>{title}</h2><img src='{os.path.basename(f)}' />")
        parts.append("</body></html>")
        with open(html_path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(parts))
        logger.info(f"HTML仪表盘已生成: {html_path}")


def create_visualization_from_backtest_results():
    paths = get_paths()
    results_dir = paths['results_dir']
    trade_files = glob.glob(os.path.join(results_dir, 'trades_*.csv'))
    if not trade_files:
        logger.error("未找到回测交易结果文件"); return
    latest_trade_file = max(trade_files, key=os.path.getctime)
    trades_df = pd.read_csv(latest_trade_file)
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    data_cache_dir = paths['data_cache_dir']
    data_files = glob.glob(os.path.join(data_cache_dir, '*.csv'))
    if data_files:
        latest_data_file = max(data_files, key=os.path.getctime)
        data_df = pd.read_csv(latest_data_file)
        data_df['date'] = pd.to_datetime(data_df['date']); data_df = data_df.set_index('date')
    else:
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        np.random.seed(42)
        prices = 4000 + np.cumsum(np.random.normal(0, 50, len(dates)))
        data_df = pd.DataFrame({'close': prices, 'ma20': pd.Series(prices).rolling(20).mean(), 'volume': np.random.randint(10000, 100000, len(dates))}, index=dates)
    visualizer = StrategyVisualizer()
    backtest_results = {'initial_capital': 100000, 'final_capital': 100000, 'total_return': 0, 'total_trades': len(trades_df[trades_df['pnl'].notna()])}
    visualizer.create_comprehensive_report(data_df, trades_df, backtest_results, save_dir=results_dir, generate_html=True)
    logger.info("可视化报告生成完成")


if __name__ == "__main__":
    create_visualization_from_backtest_results()

