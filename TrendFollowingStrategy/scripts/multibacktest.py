"""
MA20趋势跟踪策略 - 多品种回测验证
对螺纹钢、铜、沪深300等多个品种进行回测对比
"""

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from main_simple import MA20TrendFollowingStrategySimple
from config import get_paths
from src.performance_analyzer import PerformanceAnalyzer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiInstrumentBacktest:
    """多品种回测验证器"""
    
    def __init__(self):
        """初始化多品种回测器"""
        self.results = {}
        self.comparison_df = None
        self.analyzer = PerformanceAnalyzer()
    
    def test_single_instrument(self, symbol: str, start_date: str = '2020-01-01',
                             end_date: str = '2024-12-31', 
                             initial_capital: float = 100000) -> Dict[str, Any]:
        """测试单个品种
        
        Args:
            symbol: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            
        Returns:
            测试结果字典
        """
        logger.info(f"开始测试品种: {symbol}")
        
        try:
            # 创建策略实例
            strategy = MA20TrendFollowingStrategySimple(symbol=symbol, data_source='akshare')
            
            # 运行完整策略
            results = strategy.run_complete_strategy(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                save_results=False  # 不单独保存，统一保存
            )
            
            logger.info(f"品种 {symbol} 测试完成")
            return results
            
        except Exception as e:
            logger.error(f"品种 {symbol} 测试失败: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def test_multiple_instruments(self, symbols: List[str], 
                                start_date: str = '2020-01-01',
                                end_date: str = '2024-12-31',
                                initial_capital: float = 100000) -> Dict[str, Any]:
        """测试多个品种
        
        Args:
            symbols: 品种代码列表
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            
        Returns:
            所有测试结果
        """
        logger.info(f"开始多品种测试: {symbols}")
        
        all_results = {}
        
        for symbol in symbols:
            try:
                result = self.test_single_instrument(symbol, start_date, end_date, initial_capital)
                all_results[symbol] = result
                
                # 简要输出结果
                if 'error' not in result:
                    basic_info = result.get('backtest_results', {}).get('basic_info', {})
                    total_return = basic_info.get('total_return', 0) * 100
                    total_trades = basic_info.get('total_trades', 0)
                    logger.info(f"{symbol}: 收益率 {total_return:+.2f}%, 交易次数 {total_trades}")
                else:
                    logger.warning(f"{symbol}: 测试失败 - {result['error']}")
                    
            except Exception as e:
                logger.error(f"测试 {symbol} 时发生异常: {e}")
                all_results[symbol] = {'error': str(e), 'symbol': symbol}
        
        self.results = all_results
        logger.info("多品种测试完成")
        return all_results
    
    def compare_results(self) -> pd.DataFrame:
        """对比各品种结果
        
        Returns:
            对比结果DataFrame
        """
        if not self.results:
            logger.warning("没有测试结果可供对比")
            return pd.DataFrame()
        
        comparison_data = []
        
        for symbol, result in self.results.items():
            if 'error' in result:
                continue
                
            try:
                # 提取基本信息
                basic_info = result.get('backtest_results', {}).get('basic_info', {})
                return_metrics = result.get('backtest_results', {}).get('return_metrics', {})
                risk_metrics = result.get('backtest_results', {}).get('risk_metrics', {})
                trade_metrics = result.get('backtest_results', {}).get('trade_metrics', {})
                
                # 提取数据
                row = {
                    '品种': symbol,
                    '初始资金': basic_info.get('initial_capital', 0),
                    '最终资产': basic_info.get('final_value', 0),
                    '总收益率(%)': basic_info.get('total_return', 0) * 100,
                    '年化收益率(%)': return_metrics.get('annual_return_pct', 0),
                    '夏普比率': risk_metrics.get('sharpe_ratio', 0),
                    '最大回撤(%)': risk_metrics.get('max_drawdown_pct', 0),
                    '胜率(%)': trade_metrics.get('win_rate_pct', 0),
                    '盈亏比': trade_metrics.get('profit_factor', 0),
                    '总交易次数': basic_info.get('total_trades', 0),
                    '盈利交易': trade_metrics.get('won_trades', 0),
                    '亏损交易': trade_metrics.get('lost_trades', 0),
                    '平均盈利': trade_metrics.get('avg_win', 0),
                    '平均亏损': trade_metrics.get('avg_loss', 0),
                }
                
                comparison_data.append(row)
                
            except Exception as e:
                logger.error(f"处理 {symbol} 结果时出错: {e}")
                continue
        
        if not comparison_data:
            logger.warning("没有有效的结果数据")
            return pd.DataFrame()
        
        # 创建对比DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # 按收益率排序
        comparison_df = comparison_df.sort_values('总收益率(%)', ascending=False)
        
        self.comparison_df = comparison_df
        return comparison_df
    
    def generate_comparison_report(self) -> str:
        """生成对比报告
        
        Returns:
            格式化报告字符串
        """
        if self.comparison_df is None or self.comparison_df.empty:
            return "没有对比数据可供生成报告"
        
        df = self.comparison_df
        
        report = []
        report.append("=" * 80)
        report.append("                    MA20趋势跟踪策略 - 多品种对比报告")
        report.append("=" * 80)
        
        # 总体统计
        total_symbols = len(df)
        successful_symbols = len(df[df['总收益率(%)'] > 0])
        
        report.append(f"\n【总体统计】")
        report.append(f"测试品种数量: {total_symbols}")
        report.append(f"盈利品种数量: {successful_symbols}")
        report.append(f"整体胜率: {successful_symbols/total_symbols*100:.1f}%")
        
        # 最佳和最差表现
        best_performer = df.iloc[0]
        worst_performer = df.iloc[-1]
        
        report.append(f"\n【最佳表现】")
        report.append(f"品种: {best_performer['品种']}")
        report.append(f"总收益率: {best_performer['总收益率(%)']:+.2f}%")
        report.append(f"夏普比率: {best_performer['夏普比率']:.2f}")
        report.append(f"最大回撤: {best_performer['最大回撤(%)']:.2f}%")
        report.append(f"胜率: {best_performer['胜率(%)']:.1f}%")
        
        report.append(f"\n【最差表现】")
        report.append(f"品种: {worst_performer['品种']}")
        report.append(f"总收益率: {worst_performer['总收益率(%)']:+.2f}%")
        report.append(f"夏普比率: {worst_performer['夏普比率']:.2f}")
        report.append(f"最大回撤: {worst_performer['最大回撤(%)']:.2f}%")
        report.append(f"胜率: {worst_performer['胜率(%)']:.1f}%")
        
        # 平均表现
        avg_return = df['总收益率(%)'].mean()
        avg_sharpe = df['夏普比率'].mean()
        avg_drawdown = df['最大回撤(%)'].mean()
        avg_win_rate = df['胜率(%)'].mean()
        
        report.append(f"\n【平均表现】")
        report.append(f"平均收益率: {avg_return:+.2f}%")
        report.append(f"平均夏普比率: {avg_sharpe:.2f}")
        report.append(f"平均最大回撤: {avg_drawdown:.2f}%")
        report.append(f"平均胜率: {avg_win_rate:.1f}%")
        
        # 详细对比表
        report.append(f"\n【详细对比】")
        report.append(df.to_string(index=False))
        
        report.append(f"\n【报告生成时间】")
        report.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_comparison_results(self, save_dir: str = None):
        """保存对比结果
        
        Args:
            save_dir: 保存目录
        """
        try:
            paths = get_paths()
            base_results_dir = paths['results_dir']
            target_dir = save_dir or os.path.join(base_results_dir, 'multibacktest')
            os.makedirs(target_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存对比表格
            if self.comparison_df is not None:
                csv_path = os.path.join(target_dir, f'multibacktest_comparison_{timestamp}.csv')
                self.comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                logger.info(f"对比表格已保存到: {csv_path}")
            
            # 保存对比报告
            report = self.generate_comparison_report()
            report_path = os.path.join(target_dir, f'multibacktest_report_{timestamp}.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"对比报告已保存到: {report_path}")
            
            # 保存详细结果
            import json
            results_path = os.path.join(target_dir, f'multibacktest_results_{timestamp}.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"详细结果已保存到: {results_path}")
            
        except Exception as e:
            logger.error(f"保存对比结果失败: {e}")
    
    def sensitivity_analysis(self, symbol: str = 'RB0', 
                           ma_periods: List[int] = [15, 20, 25, 30],
                           stop_loss_pcts: List[float] = [0.04, 0.06, 0.08],
                           start_date: str = '2020-01-01',
                           end_date: str = '2024-12-31') -> pd.DataFrame:
        """敏感性分析
        
        Args:
            symbol: 测试品种
            ma_periods: MA周期列表
            stop_loss_pcts: 止损比例列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            敏感性分析结果DataFrame
        """
        logger.info(f"开始敏感性分析: {symbol}")
        
        from config import get_config
        from src.risk_manager import RiskManager, RiskParameters
        
        sensitivity_results = []
        
        for ma_period in ma_periods:
            for stop_loss_pct in stop_loss_pcts:
                try:
                    logger.info(f"测试参数组合: MA{ma_period}, 止损{stop_loss_pct*100:.0f}%")
                    
                    # 修改配置
                    config = get_config()
                    config['ma_period'] = ma_period
                    config['max_loss_pct'] = stop_loss_pct
                    
                    # 创建策略
                    strategy = MA20TrendFollowingStrategySimple(symbol=symbol, data_source='akshare')
                    
                    # 运行测试
                    results = strategy.run_complete_strategy(
                        start_date=start_date,
                        end_date=end_date,
                        save_results=False
                    )
                    
                    # 提取结果
                    if 'error' not in results:
                        basic_info = results.get('backtest_results', {}).get('basic_info', {})
                        risk_metrics = results.get('backtest_results', {}).get('risk_metrics', {})
                        trade_metrics = results.get('backtest_results', {}).get('trade_metrics', {})
                        
                        row = {
                            'MA周期': ma_period,
                            '止损比例(%)': stop_loss_pct * 100,
                            '总收益率(%)': basic_info.get('total_return', 0) * 100,
                            '年化收益率(%)': results.get('backtest_results', {}).get('return_metrics', {}).get('annual_return_pct', 0),
                            '夏普比率': risk_metrics.get('sharpe_ratio', 0),
                            '最大回撤(%)': risk_metrics.get('max_drawdown_pct', 0),
                            '胜率(%)': trade_metrics.get('win_rate_pct', 0),
                            '盈亏比': trade_metrics.get('profit_factor', 0),
                            '总交易次数': basic_info.get('total_trades', 0),
                        }
                        
                        sensitivity_results.append(row)
                        
                except Exception as e:
                    logger.error(f"参数组合测试失败: MA{ma_period}, 止损{stop_loss_pct*100:.0f}% - {e}")
                    continue
        
        if not sensitivity_results:
            logger.warning("没有敏感性分析结果")
            return pd.DataFrame()
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # 找出最佳参数组合
        best_return = sensitivity_df.loc[sensitivity_df['总收益率(%)'].idxmax()]
        best_sharpe = sensitivity_df.loc[sensitivity_df['夏普比率'].idxmax()]
        best_drawdown = sensitivity_df.loc[sensitivity_df['最大回撤(%)'].idxmin()]
        
        logger.info(f"敏感性分析完成")
        logger.info(f"最佳收益率: MA{best_return['MA周期']}, 止损{best_return['止损比例(%)']:.0f}%")
        logger.info(f"最佳夏普: MA{best_sharpe['MA周期']}, 止损{best_sharpe['止损比例(%)']:.0f}%")
        logger.info(f"最小回撤: MA{best_drawdown['MA周期']}, 止损{best_drawdown['止损比例(%)']:.0f}%")
        
        return sensitivity_df


def run_comprehensive_multibacktest():
    """运行综合多品种回测"""
    print("开始运行MA20趋势跟踪策略 - 多品种回测验证")
    print("=" * 80)
    
    # 测试品种列表
    test_symbols = ['RB0', 'CU0', 'IF0']  # 螺纹钢、铜、沪深300
    
    # 创建多品种回测器
    multibacktest = MultiInstrumentBacktest()
    
    # 运行多品种测试
    results = multibacktest.test_multiple_instruments(
        symbols=test_symbols,
        start_date='2020-01-01',
        end_date='2024-12-31',
        initial_capital=100000
    )
    
    # 生成对比结果
    comparison_df = multibacktest.compare_results()
    
    if not comparison_df.empty:
        print("\n多品种对比结果:")
        print(comparison_df.to_string(index=False))
        
        # 生成对比报告
        comparison_report = multibacktest.generate_comparison_report()
        print(f"\n{comparison_report}")
        
        # 保存结果
        multibacktest.save_comparison_results()
        
        # 运行敏感性分析（以螺纹钢为例）
        print("\n运行敏感性分析（螺纹钢RB0）...")
        sensitivity_df = multibacktest.sensitivity_analysis(
            symbol='RB0',
            ma_periods=[15, 20, 25, 30],
            stop_loss_pcts=[0.04, 0.06, 0.08]
        )
        
        if not sensitivity_df.empty:
            print("\n敏感性分析结果:")
            print(sensitivity_df.to_string(index=False))
            
            # 保存敏感性分析结果（统一使用配置路径）
            paths = get_paths()
            base_results_dir = paths['results_dir']
            target_dir = os.path.join(base_results_dir, 'multibacktest')
            os.makedirs(target_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            sensitivity_path = os.path.join(target_dir, f'sensitivity_analysis_{timestamp}.csv')
            sensitivity_df.to_csv(sensitivity_path, index=False, encoding='utf-8-sig')
            print(f"敏感性分析结果已保存到: {sensitivity_path}")
    
    print("\n多品种回测验证完成!")
    return results


if __name__ == "__main__":
    run_comprehensive_multibacktest()
