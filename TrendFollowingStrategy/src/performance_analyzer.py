"""
MA20趋势跟踪策略 - 绩效分析和可视化模块
实现回测结果的详细分析和图表生成
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    # 在初始化Visualizer或绘图前添加
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib或seaborn未安装，图表功能将受限")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly未安装，交互式图表功能将受限")


class PerformanceAnalyzer:
    """绩效分析器"""
    
    def __init__(self):
        """初始化绩效分析器"""
        self.metrics = {}
        self.analysis_results = {}
    
    def calculate_metrics(self, trades_df: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
        """计算各项绩效指标
        
        Args:
            trades_df: 交易记录DataFrame
            equity_curve: 权益曲线Series
            
        Returns:
            绩效指标字典
        """
        logger.info("开始计算绩效指标...")
        
        if trades_df.empty:
            logger.warning("交易记录为空")
            return {}
        
        # 基础统计
        total_trades = len(trades_df)
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] < 0).sum()
        breakeven_trades = (trades_df['pnl'] == 0).sum()
        
        # 盈亏统计
        total_pnl = trades_df['pnl'].sum()
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = trades_df[trades_df['pnl'] < 0]['pnl'].sum()
        
        # 计算指标
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # 最大单笔盈亏
        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        
        # 连续盈亏
        consecutive_wins = self._calculate_consecutive_trades(trades_df, 'win')
        consecutive_losses = self._calculate_consecutive_trades(trades_df, 'loss')
        
        # 持仓时间统计
        avg_holding_days = trades_df['holding_days'].mean()
        max_holding_days = trades_df['holding_days'].max()
        min_holding_days = trades_df['holding_days'].min()
        
        # 回撤分析
        drawdown_analysis = self._calculate_drawdown(equity_curve)
        
        # 收益风险指标
        if equity_curve is not None and len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            annual_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # 计算Sortino比率
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
            
            # 计算Calmar比率
            max_drawdown = drawdown_analysis.get('max_drawdown', 0)
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            annual_return = volatility = sharpe_ratio = sortino_ratio = calmar_ratio = 0
        
        metrics = {
            'trade_summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'breakeven_trades': breakeven_trades,
                'win_rate_pct': win_rate * 100,
                'avg_holding_days': avg_holding_days,
                'max_holding_days': max_holding_days,
                'min_holding_days': min_holding_days,
            },
            'pnl_summary': {
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'net_profit': gross_profit + gross_loss,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_win': max_win,
                'max_loss': max_loss,
            },
            'consecutive_trades': {
                'max_consecutive_wins': consecutive_wins['max_consecutive'],
                'max_consecutive_losses': consecutive_losses['max_consecutive'],
                'avg_consecutive_wins': consecutive_wins['avg_consecutive'],
                'avg_consecutive_losses': consecutive_losses['avg_consecutive'],
            },
            'risk_metrics': {
                'max_drawdown_pct': drawdown_analysis.get('max_drawdown', 0) * 100,
                'max_drawdown_period': drawdown_analysis.get('max_drawdown_period', 0),
                'avg_drawdown_pct': drawdown_analysis.get('avg_drawdown', 0) * 100,
                'recovery_factor': abs(total_pnl / drawdown_analysis.get('max_drawdown', 1)),
            },
            'return_metrics': {
                'annual_return_pct': annual_return * 100,
                'volatility_pct': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
            }
        }
        
        self.metrics = metrics
        logger.info("绩效指标计算完成")
        return metrics
    
    def _calculate_consecutive_trades(self, trades_df: pd.DataFrame, trade_type: str) -> Dict[str, Any]:
        """计算连续交易统计
        
        Args:
            trades_df: 交易记录
            trade_type: 'win' 或 'loss'
            
        Returns:
            连续交易统计
        """
        if trade_type == 'win':
            mask = trades_df['pnl'] > 0
        else:
            mask = trades_df['pnl'] < 0
        
        # 找到连续序列
        consecutive_groups = []
        current_group = 0
        current_length = 0
        
        for is_target in mask:
            if is_target:
                current_length += 1
            else:
                if current_length > 0:
                    consecutive_groups.append(current_length)
                current_length = 0
        
        if current_length > 0:
            consecutive_groups.append(current_length)
        
        if consecutive_groups:
            max_consecutive = max(consecutive_groups)
            avg_consecutive = np.mean(consecutive_groups)
        else:
            max_consecutive = 0
            avg_consecutive = 0
        
        return {
            'max_consecutive': max_consecutive,
            'avg_consecutive': avg_consecutive,
            'groups': consecutive_groups
        }
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """计算回撤指标
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            回撤分析结果
        """
        if equity_curve is None or len(equity_curve) < 2:
            return {'max_drawdown': 0, 'max_drawdown_period': 0, 'avg_drawdown': 0}
        
        # 计算累计最大值
        rolling_max = equity_curve.expanding().max()
        
        # 计算回撤
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        # 最大回撤
        max_drawdown = drawdown.min()
        
        # 最大回撤期
        max_dd_end = drawdown.idxmin()
        max_dd_start = rolling_max.loc[:max_dd_end].idxmax()
        max_drawdown_period = (pd.to_datetime(max_dd_end) - pd.to_datetime(max_dd_start)).days
        
        # 平均回撤
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_period': max_drawdown_period,
            'avg_drawdown': avg_drawdown,
            'drawdown_series': drawdown
        }
    
    def analyze_monthly_returns(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """分析月度收益
        
        Args:
            trades_df: 交易记录
            
        Returns:
            月度收益分析DataFrame
        """
        if trades_df.empty:
            return pd.DataFrame()
        
        # 确保日期格式正确
        trades_df = trades_df.copy()
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        # 按月份分组
        trades_df['year'] = trades_df['exit_date'].dt.year
        trades_df['month'] = trades_df['exit_date'].dt.month
        
        monthly_stats = trades_df.groupby(['year', 'month']).agg({
            'pnl': ['sum', 'count', 'mean'],
            'holding_days': 'mean'
        }).round(2)
        
        # 重命名列
        monthly_stats.columns = ['total_pnl', 'trade_count', 'avg_pnl', 'avg_holding_days']
        monthly_stats = monthly_stats.reset_index()
        
        # 添加胜率
        monthly_win_rate = trades_df.groupby(['year', 'month']).apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).reset_index()
        monthly_win_rate.columns = ['year', 'month', 'win_rate_pct']
        
        monthly_stats = monthly_stats.merge(monthly_win_rate, on=['year', 'month'])
        
        return monthly_stats
    
    def generate_performance_report(self, trades_df: pd.DataFrame, 
                                 equity_curve: Optional[pd.Series] = None) -> str:
        """生成绩效报告
        
        Args:
            trades_df: 交易记录
            equity_curve: 权益曲线
            
        Returns:
            格式化报告字符串
        """
        # 计算指标
        metrics = self.calculate_metrics(trades_df, equity_curve)
        
        if not metrics:
            return "无交易数据，无法生成报告"
        
        # 月度分析
        monthly_analysis = self.analyze_monthly_returns(trades_df)
        
        # 生成报告
        report = []
        report.append("=" * 60)
        report.append("                 MA20趋势跟踪策略绩效报告")
        report.append("=" * 60)
        
        # 交易统计
        trade_summary = metrics['trade_summary']
        report.append(f"\n【交易统计】")
        report.append(f"总交易次数: {trade_summary['total_trades']}")
        report.append(f"盈利交易: {trade_summary['winning_trades']} ({trade_summary['win_rate_pct']:.1f}%)")
        report.append(f"亏损交易: {trade_summary['losing_trades']}")
        report.append(f"平均持仓天数: {trade_summary['avg_holding_days']:.1f}")
        
        # 盈亏统计
        pnl_summary = metrics['pnl_summary']
        report.append(f"\n【盈亏统计】")
        report.append(f"总盈亏: {pnl_summary['total_pnl']:,.2f} CNY")
        report.append(f"毛利润: {pnl_summary['gross_profit']:,.2f} CNY")
        report.append(f"毛亏损: {pnl_summary['gross_loss']:,.2f} CNY")
        report.append(f"净利润: {pnl_summary['net_profit']:,.2f} CNY")
        report.append(f"盈亏比: {pnl_summary['profit_factor']:.2f}")
        report.append(f"平均盈利: {pnl_summary['avg_win']:,.2f} CNY")
        report.append(f"平均亏损: {pnl_summary['avg_loss']:,.2f} CNY")
        
        # 风险指标
        risk_metrics = metrics['risk_metrics']
        report.append(f"\n【风险指标】")
        report.append(f"最大回撤: {risk_metrics['max_drawdown_pct']:.2f}%")
        report.append(f"最大回撤期: {risk_metrics['max_drawdown_period']} 天")
        report.append(f"回撤恢复因子: {risk_metrics['recovery_factor']:.2f}")
        
        # 收益指标
        return_metrics = metrics['return_metrics']
        report.append(f"\n【收益指标】")
        report.append(f"年化收益率: {return_metrics['annual_return_pct']:+.2f}%")
        report.append(f"年化波动率: {return_metrics['volatility_pct']:.2f}%")
        report.append(f"夏普比率: {return_metrics['sharpe_ratio']:.2f}")
        report.append(f"索提诺比率: {return_metrics['sortino_ratio']:.2f}")
        report.append(f"卡玛比率: {return_metrics['calmar_ratio']:.2f}")
        
        # 月度表现
        if not monthly_analysis.empty:
            report.append(f"\n【月度表现（前6个月）】")
            recent_months = monthly_analysis.tail(6)
            for _, month_data in recent_months.iterrows():
                report.append(f"{month_data['year']}-{month_data['month']:02d}: "
                           f"盈亏={month_data['total_pnl']:,.0f}, "
                           f"交易={month_data['trade_count']}, "
                           f"胜率={month_data['win_rate_pct']:.1f}%")
        
        report.append("\n" + "=" * 60)
        report.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        return "\n".join(report)


class PerformanceVisualizer:
    """绩效可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib不可用，部分图表功能将受限")
    
    def plot_equity_curve(self, equity_curve: pd.Series, benchmark: Optional[pd.Series] = None,
                         title: str = "权益曲线", save_path: Optional[str] = None) -> None:
        """绘制权益曲线
        
        Args:
            equity_curve: 策略权益曲线
            benchmark: 基准权益曲线（可选）
            title: 图表标题
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法绘制权益曲线")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制策略权益曲线
            ax.plot(equity_curve.index, equity_curve.values, 
                   label='策略', color='blue', linewidth=2)
            
            # 绘制基准曲线（如果有）
            if benchmark is not None:
                ax.plot(benchmark.index, benchmark.values, 
                       label='基准', color='gray', linewidth=1, alpha=0.7)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('日期')
            ax.set_ylabel('权益 (CNY)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 格式化日期轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"权益曲线图已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制权益曲线失败: {e}")
    
    def drawdown_chart(self, equity_curve: pd.Series, save_path: Optional[str] = None) -> None:
        """绘制回撤图表
        
        Args:
            equity_curve: 权益曲线
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法绘制回撤图表")
            return
        
        try:
            # 计算回撤
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max * 100
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制回撤
            ax.fill_between(drawdown.index, drawdown.values, 0, 
                           color='red', alpha=0.3, label='回撤')
            ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            
            # 标记最大回撤
            max_dd_idx = drawdown.idxmin()
            max_dd_value = drawdown.min()
            ax.scatter(max_dd_idx, max_dd_value, color='darkred', s=100, zorder=5)
            ax.annotate(f'最大回撤: {max_dd_value:.1f}%', 
                       xy=(max_dd_idx, max_dd_value),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax.set_title('回撤分析', fontsize=16, fontweight='bold')
            ax.set_xlabel('日期')
            ax.set_ylabel('回撤 (%)')
            ax.grid(True, alpha=0.3)
            
            # 格式化日期轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"回撤图已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制回撤图失败: {e}")
    
    def trade_distribution(self, trades_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """绘制交易分布图
        
        Args:
            trades_df: 交易记录
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法绘制交易分布图")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 盈亏分布直方图
            ax1 = axes[0, 0]
            trades_df['pnl'].hist(bins=30, ax=ax1, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='盈亏平衡点')
            ax1.set_title('盈亏分布')
            ax1.set_xlabel('盈亏 (CNY)')
            ax1.set_ylabel('频次')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 盈亏散点图（按时间）
            ax2 = axes[0, 1]
            colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
            ax2.scatter(range(len(trades_df)), trades_df['pnl'], c=colors, alpha=0.6)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_title('盈亏序列')
            ax2.set_xlabel('交易序号')
            ax2.set_ylabel('盈亏 (CNY)')
            ax2.grid(True, alpha=0.3)
            
            # 3. 持仓时间分布
            ax3 = axes[1, 0]
            trades_df['holding_days'].hist(bins=20, ax=ax3, alpha=0.7, color='orange', edgecolor='black')
            ax3.set_title('持仓时间分布')
            ax3.set_xlabel('持仓天数')
            ax3.set_ylabel('频次')
            ax3.grid(True, alpha=0.3)
            
            # 4. 盈亏vs持仓时间散点图
            ax4 = axes[1, 1]
            ax4.scatter(trades_df['holding_days'], trades_df['pnl'], alpha=0.6, color='purple')
            ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax4.set_title('盈亏vs持仓时间')
            ax4.set_xlabel('持仓天数')
            ax4.set_ylabel('盈亏 (CNY)')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('交易分布分析', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"交易分布图已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制交易分布图失败: {e}")
    
    def monthly_performance_heatmap(self, trades_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """绘制月度表现热力图
        
        Args:
            trades_df: 交易记录
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法绘制月度热力图")
            return
        
        try:
            # 准备月度数据
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_df['year'] = trades_df['exit_date'].dt.year
            trades_df['month'] = trades_df['exit_date'].dt.month
            
            monthly_pnl = trades_df.groupby(['year', 'month'])['pnl'].sum().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 创建热力图
            sns.heatmap(monthly_pnl, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                       ax=ax, cbar_kws={'label': '月度盈亏 (CNY)'})
            
            ax.set_title('月度盈亏热力图', fontsize=16, fontweight='bold')
            ax.set_xlabel('月份')
            ax.set_ylabel('年份')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"月度热力图已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制月度热力图失败: {e}")


def test_performance_analyzer():
    """测试绩效分析器"""
    print("测试绩效分析器...")
    
    # 创建测试交易数据
    np.random.seed(42)
    n_trades = 100
    
    # 生成测试交易记录
    dates = pd.date_range(start='2023-01-01', periods=n_trades, freq='3D')
    
    # 生成盈亏数据（正态分布，稍微偏向盈利）
    pnls = np.random.normal(100, 500, n_trades)
    pnls = pnls + np.abs(pnls.min()) + 100  # 确保大部分是盈利的
    
    # 随机生成一些亏损交易
    loss_indices = np.random.choice(n_trades, size=int(n_trades * 0.3), replace=False)
    pnls[loss_indices] = -np.random.uniform(100, 800, len(loss_indices))
    
    test_trades = pd.DataFrame({
        'entry_date': dates - pd.Timedelta(days=2),
        'exit_date': dates,
        'entry_price': np.random.uniform(4000, 4500, n_trades),
        'exit_price': np.random.uniform(4000, 4500, n_trades),
        'pnl': pnls,
        'holding_days': np.random.randint(1, 10, n_trades),
        'position_side': np.random.choice(['LONG', 'SHORT'], n_trades),
        'reason': np.random.choice(['止损', '反转', '止盈'], n_trades)
    })
    
    # 生成权益曲线
    cumulative_pnl = test_trades['pnl'].cumsum()
    initial_capital = 100000
    equity_curve = pd.Series(initial_capital + cumulative_pnl, index=test_trades['exit_date'])
    
    # 测试绩效分析
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(test_trades, equity_curve)
    
    print("\n1. 绩效指标:")
    print(f"总交易次数: {metrics['trade_summary']['total_trades']}")
    print(f"胜率: {metrics['trade_summary']['win_rate_pct']:.2f}%")
    print(f"总盈亏: {metrics['pnl_summary']['total_pnl']:,.2f}")
    print(f"盈亏比: {metrics['pnl_summary']['profit_factor']:.2f}")
    print(f"夏普比率: {metrics['return_metrics']['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['risk_metrics']['max_drawdown_pct']:.2f}%")
    
    # 测试报告生成
    print("\n2. 生成绩效报告:")
    report = analyzer.generate_performance_report(test_trades, equity_curve)
    print(report[:500] + "...")  # 打印报告前500字符
    
    # 测试可视化
    if MATPLOTLIB_AVAILABLE:
        print("\n3. 测试可视化功能:")
        visualizer = PerformanceVisualizer()
        
        print("绘制权益曲线...")
        visualizer.plot_equity_curve(equity_curve)
        
        print("绘制回撤图...")
        visualizer.drawdown_chart(equity_curve)
        
        print("绘制交易分布图...")
        visualizer.trade_distribution(test_trades)
        
        print("绘制月度热力图...")
        visualizer.monthly_performance_heatmap(test_trades)
    
    print("\n绩效分析器测试完成!")


if __name__ == "__main__":
    test_performance_analyzer()