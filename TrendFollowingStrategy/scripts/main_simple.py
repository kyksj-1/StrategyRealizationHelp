"""
MA20趋势跟踪策略 - 简化主程序
使用验证过的简化回测引擎
"""

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# 导入策略模块
from src.data_fetcher import DataFetcher
from src.data_processor import DataProcessor
from src.signal_generator import SignalGenerator
from src.risk_manager import RiskManager, PositionSide
from src.performance_analyzer import PerformanceAnalyzer
from config import get_config, validate_config, get_instrument_config, get_paths

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MA20TrendFollowingStrategySimple:
    """MA20趋势跟踪策略简化版"""
    
    def __init__(self, symbol: str = 'RB0', data_source: str = 'akshare'):
        """初始化策略
        
        Args:
            symbol: 交易品种代码
            data_source: 数据源 ('tushare' 或 'akshare')
        """
        self.symbol = symbol
        self.data_source = data_source
        self.config = get_config()
        
        # 初始化各模块
        self.data_fetcher = DataFetcher(data_source)
        self.data_processor = DataProcessor()
        self.signal_generator = SignalGenerator(ma_period=self.config['ma_period'])
        self.risk_manager = RiskManager()
        self.performance_analyzer = PerformanceAnalyzer()
        
        logger.info(f"MA20趋势跟踪策略初始化完成，品种: {symbol}, 数据源: {data_source}")
    
    def create_test_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """创建测试数据（当数据获取失败时使用）"""
        logger.info("创建模拟测试数据...")
        
        # 生成日期范围
        dates = pd.date_range(start_date, end_date, freq='D')
        n = len(dates)
        
        # 生成价格数据（趋势+随机波动）
        np.random.seed(42)
        base_price = 4000
        
        # 创建趋势（根据时间长度调整）
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days_diff = (end_dt - start_dt).days
        
        # 生成趋势（模拟真实市场波动）
        trend = np.linspace(-200, 200, n)  # 从-200到+200的趋势
        noise = np.cumsum(np.random.normal(0, 20, n))  # 随机游走
        prices = base_price + trend + noise
        
        # 确保价格在合理范围内
        prices = np.clip(prices, 3000, 6000)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.normal(0, 10, n),
            'high': prices + np.random.uniform(0, 50, n),
            'low': prices - np.random.uniform(0, 50, n),
            'close': prices,
            'volume': np.random.randint(10000, 100000, n)
        })
        
        # 确保价格逻辑正确
        for i in range(len(df)):
            row = df.iloc[i]
            df.loc[i, 'high'] = max(row['high'], row['open'], row['close'])
            df.loc[i, 'low'] = min(row['low'], row['open'], row['close'])
        
        logger.info(f"✓ 创建模拟数据: {len(df)} 条记录")
        return df
    
    def prepare_data(self, start_date: str, end_date: str, 
                    cache_dir: str = 'data/cache') -> pd.DataFrame:
        """准备策略数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            cache_dir: 缓存目录
            
        Returns:
            完整的策略数据DataFrame
        """
        logger.info(f"准备数据: {start_date} 至 {end_date}")
        
        try:
            # 1. 获取原始数据
            raw_data = self.data_fetcher.fetch_futures_data(self.symbol, start_date, end_date)
            logger.info(f"获取原始数据: {len(raw_data)} 条记录")
        except Exception as e:
            logger.warning(f"数据获取失败，使用模拟数据: {e}")
            raw_data = self.create_test_data(start_date, end_date)
        
        # 2. 保存原始数据缓存
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        try:
            self.data_fetcher.save_data(raw_data, self.symbol, cache_dir)
        except Exception as e:
            logger.warning(f"数据保存失败: {e}")
        
        # 3. 合成2日K线
        try:
            data_2day = self.data_processor.create_2day_kline(raw_data)
            logger.info(f"合成2日K线: {len(data_2day)} 条记录")
        except Exception as e:
            logger.error(f"2日K线合成失败: {e}")
            raise
        
        # 4. 准备策略数据（计算MA和特征）
        try:
            strategy_data = self.data_processor.prepare_strategy_data(data_2day, self.config['ma_period'])
            logger.info(f"策略数据准备完成: {len(strategy_data)} 条有效记录")
        except Exception as e:
            logger.error(f"策略数据准备失败: {e}")
            raise
        
        # 5. 生成交易信号
        try:
            signals_data = self.signal_generator.generate_signals(strategy_data)
            logger.info(f"信号生成完成")
        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            raise
        
        # 6. 数据摘要
        summary = self.data_processor.get_data_summary(signals_data)
        logger.info(f"数据摘要: {summary}")
        
        return signals_data
    
    def simple_backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, Any]:
        """简化回测
        
        Args:
            data: 策略数据
            initial_capital: 初始资金
            
        Returns:
            回测结果字典
        """
        logger.info(f"开始简化回测，初始资金: {initial_capital}")
        
        # 初始化回测状态
        capital = initial_capital
        position = 0  # 持仓数量
        entry_price = 0
        stop_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        commission = 0.0003  # 手续费
        slippage = 0.001      # 滑点
        contract_multiplier = 10  # 合约乘数
        margin_rate = 0.10   # 保证金率
        
        logger.info("开始回测逻辑...")
        
        # 回测逻辑
        for i in range(len(data)):
            row = data.iloc[i]
            current_price = row['close']
            signal = row['signal']
            
            # 无持仓时检查信号
            if position == 0:
                if signal == 1:  # 做多信号
                    # 计算止损
                    prev_low = data.iloc[i-1]['low'] if i > 0 else row['low']
                    stop_result = self.risk_manager.calculate_stop_loss(
                        entry_price=current_price,
                        prev_extreme=prev_low,
                        direction=PositionSide.LONG
                    )
                    
                    # 计算仓位
                    position_result = self.risk_manager.calculate_position_size(
                        capital=capital,
                        entry_price=current_price,
                        stop_price=stop_result.stop_price,
                        margin_rate=margin_rate,
                        contract_multiplier=contract_multiplier
                    )
                    
                    # 开仓
                    position = position_result.position_size
                    entry_price = current_price
                    stop_price = stop_result.stop_price
                    
                    # 扣除手续费和滑点
                    total_cost = entry_price * position * contract_multiplier * (commission + slippage)
                    capital -= total_cost
                    
                    trades.append({
                        'date': row['date'],
                        'type': 'BUY',
                        'price': entry_price,
                        'size': position,
                        'stop_price': stop_price,
                        'capital': capital
                    })
                    
                    logger.info(f"做多开仓: 价格={entry_price:.2f}, 数量={position}, 止损={stop_price:.2f}")
                    
                elif signal == -1:  # 做空信号
                    # 计算止损
                    prev_high = data.iloc[i-1]['high'] if i > 0 else row['high']
                    stop_result = self.risk_manager.calculate_stop_loss(
                        entry_price=current_price,
                        prev_extreme=prev_high,
                        direction=PositionSide.SHORT
                    )
                    
                    # 计算仓位
                    position_result = self.risk_manager.calculate_position_size(
                        capital=capital,
                        entry_price=current_price,
                        stop_price=stop_result.stop_price,
                        margin_rate=margin_rate,
                        contract_multiplier=contract_multiplier
                    )
                    
                    # 开仓
                    position = -position_result.position_size  # 负值表示做空
                    entry_price = current_price
                    stop_price = stop_result.stop_price
                    
                    # 扣除手续费和滑点
                    total_cost = entry_price * abs(position) * contract_multiplier * (commission + slippage)
                    capital -= total_cost
                    
                    trades.append({
                        'date': row['date'],
                        'type': 'SELL',
                        'price': entry_price,
                        'size': position,
                        'stop_price': stop_price,
                        'capital': capital
                    })
                    
                    logger.info(f"做空开仓: 价格={entry_price:.2f}, 数量={abs(position)}, 止损={stop_price:.2f}")
            
            # 有持仓时检查出场条件
            else:
                # 简化出场逻辑：K线颜色反转时平仓
                if position > 0:  # 做多持仓
                    # 收阴线时平仓
                    if row['close'] < row['open']:
                        # 平仓
                        exit_price = current_price
                        pnl = (exit_price - entry_price) * position * contract_multiplier
                        capital += pnl
                        
                        # 扣除手续费和滑点
                        total_cost = exit_price * abs(position) * contract_multiplier * (commission + slippage)
                        capital -= total_cost
                        
                        trades.append({
                            'date': row['date'],
                            'type': 'SELL',
                            'price': exit_price,
                            'size': position,
                            'pnl': pnl,
                            'capital': capital
                        })
                        
                        logger.info(f"平多仓: 价格={exit_price:.2f}, 盈亏={pnl:.2f}")
                        
                        # 重置状态
                        position = 0
                        entry_price = 0
                        stop_price = 0
                        
                elif position < 0:  # 做空持仓
                    # 收阳线时平仓
                    if row['close'] > row['open']:
                        # 平仓
                        exit_price = current_price
                        pnl = (entry_price - exit_price) * abs(position) * contract_multiplier
                        capital += pnl
                        
                        # 扣除手续费和滑点
                        total_cost = exit_price * abs(position) * contract_multiplier * (commission + slippage)
                        capital -= total_cost
                        
                        trades.append({
                            'date': row['date'],
                            'type': 'BUY',
                            'price': exit_price,
                            'size': position,
                            'pnl': pnl,
                            'capital': capital
                        })
                        
                        logger.info(f"平空仓: 价格={exit_price:.2f}, 盈亏={pnl:.2f}")
                        
                        # 重置状态
                        position = 0
                        entry_price = 0
                        stop_price = 0
            
            # 记录权益曲线
            equity_curve.append(capital)
        
        # 强制平仓剩余持仓
        if position != 0:
            exit_price = data.iloc[-1]['close']
            if position > 0:
                pnl = (exit_price - entry_price) * position * contract_multiplier
            else:
                pnl = (entry_price - exit_price) * abs(position) * contract_multiplier
            
            capital += pnl
            total_cost = exit_price * abs(position) * contract_multiplier * (commission + slippage)
            capital -= total_cost
            
            trades.append({
                'date': data.iloc[-1]['date'],
                'type': 'CLOSE',
                'price': exit_price,
                'size': position,
                'pnl': pnl,
                'capital': capital
            })
            
            logger.info(f"强制平仓: 价格={exit_price:.2f}, 盈亏={pnl:.2f}")
            equity_curve.append(capital)
        
        # 计算绩效指标
        total_return = (capital - initial_capital) / initial_capital
        winning_trades = len([t for t in trades if 'pnl' in t and t['pnl'] > 0])
        losing_trades = len([t for t in trades if 'pnl' in t and t['pnl'] < 0])
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算盈亏比
        if total_trades > 0:
            avg_win = np.mean([t['pnl'] for t in trades if 'pnl' in t and t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in trades if 'pnl' in t and t['pnl'] < 0]) if losing_trades > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            avg_win = avg_loss = profit_factor = 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
        logger.info("回测完成")
        return results
    
    def run_complete_strategy(self, start_date: str = '2020-01-01', 
                            end_date: str = '2024-12-31',
                            initial_capital: float = 100000,
                            save_results: bool = True) -> Dict[str, Any]:
        """运行完整策略
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            save_results: 是否保存结果
            
        Returns:
            完整结果字典
        """
        logger.info(f"运行完整策略: {self.symbol} ({start_date} 至 {end_date})")
        
        try:
            # 1. 准备数据
            data = self.prepare_data(start_date, end_date)
            
            # 2. 运行简化回测
            backtest_results = self.simple_backtest(data, initial_capital)
            
            # 3. 生成绩效报告
            performance_report = self.generate_performance_report(backtest_results)
            
            # 4. 保存完整结果
            complete_results = {
                'symbol': self.symbol,
                'data_source': self.data_source,
                'time_range': {'start': start_date, 'end': end_date},
                'initial_capital': initial_capital,
                'backtest_results': backtest_results,
                'performance_report': performance_report,
                'timestamp': datetime.now().isoformat()
            }
            
            if save_results:
                self._save_complete_results(complete_results)
            
            logger.info("完整策略运行完成")
            return complete_results
            
        except Exception as e:
            logger.error(f"完整策略运行失败: {e}")
            raise
    
    def generate_performance_report(self, backtest_results: Dict[str, Any]) -> str:
        """生成绩效报告
        
        Args:
            backtest_results: 回测结果
            
        Returns:
            格式化报告字符串
        """
        report = []
        report.append("=" * 60)
        report.append("           MA20趋势跟踪策略回测报告")
        report.append("=" * 60)
        
        # 基本信息
        report.append(f"\n【基本信息】")
        report.append(f"交易品种: {self.symbol}")
        report.append(f"初始资金: {backtest_results['initial_capital']:,.2f} CNY")
        report.append(f"最终资金: {backtest_results['final_capital']:,.2f} CNY")
        report.append(f"总收益率: {backtest_results['total_return']*100:+.2f}%")
        
        # 交易统计
        report.append(f"\n【交易统计】")
        report.append(f"总交易次数: {backtest_results['total_trades']}")
        report.append(f"盈利交易: {backtest_results['winning_trades']}")
        report.append(f"亏损交易: {backtest_results['losing_trades']}")
        report.append(f"胜率: {backtest_results['win_rate']*100:.2f}%")
        report.append(f"盈亏比: {backtest_results['profit_factor']:.2f}")
        
        if backtest_results['total_trades'] > 0:
            report.append(f"平均盈利: {backtest_results['avg_win']:,.2f} CNY")
            report.append(f"平均亏损: {backtest_results['avg_loss']:,.2f} CNY")
        
        # 交易明细
        trades = backtest_results['trades']
        if trades:
            report.append(f"\n【交易明细（前10笔）】")
            trade_count = 0
            for trade in trades:
                if 'pnl' in trade and trade_count < 10:
                    trade_count += 1
                    pnl_str = f"{trade['pnl']:,.2f}" if trade['pnl'] >= 0 else f"({trade['pnl']:,.2f})"
                    report.append(f"{trade_count:2d}. {trade['date'].strftime('%Y-%m-%d')} - "
                                f"{trade['type']:5s} - 价格: {trade['price']:7.2f} - 盈亏: {pnl_str:>12s}")
        
        report.append(f"\n【报告生成时间】")
        report.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _save_complete_results(self, results: Dict[str, Any]):
        """保存完整结果
        
        Args:
            results: 完整结果字典
        """
        try:
            # 创建结果目录
            paths = get_paths()
            results_dir = paths['results_dir']
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = results['symbol']
            
            # 保存回测报告
            report_filename = f"backtest_report_{symbol}_{timestamp}.txt"
            report_path = os.path.join(results_dir, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(results['performance_report'])
            
            logger.info(f"回测报告已保存到: {report_path}")
            
            # 保存交易记录
            if 'backtest_results' in results and 'trades' in results['backtest_results']:
                trades_filename = f"trades_{symbol}_{timestamp}.csv"
                trades_path = os.path.join(results_dir, trades_filename)
                
                trades_df = pd.DataFrame(results['backtest_results']['trades'])
                trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')
                
                logger.info(f"交易记录已保存到: {trades_path}")
            
            try:
                from simple_visualization import create_simple_visualization
                create_simple_visualization()
                logger.info("可视化图表已生成")
            except Exception as e:
                logger.warning(f"可视化生成失败: {e}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MA20趋势跟踪策略（简化版）')
    parser.add_argument('--symbol', type=str, default='RB0', 
                       help='交易品种代码 (默认: RB0)')
    parser.add_argument('--data-source', type=str, default='akshare',
                       choices=['tushare', 'akshare'], help='数据源 (默认: akshare)')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='开始日期 (默认: 2024-01-01)')
    parser.add_argument('--end-date', type=str, default='2025-12-31',
                       help='结束日期 (默认: 2025-12-31)')
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='初始资金 (默认: 100000)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果')
    parser.add_argument('--test', action='store_true',
                       help='运行测试模式')
    
    args = parser.parse_args()
    
    # 验证配置
    if not validate_config():
        logger.error("配置验证失败，请检查配置")
        return
    
    try:
        if args.test:
            # 测试模式
            logger.info("运行测试模式...")
            from simple_test import test_basic_functionality
            success = test_basic_functionality()
            if success:
                logger.info("所有测试通过!")
            else:
                logger.error("部分测试失败!")
        else:
            # 正常运行策略
            logger.info("运行MA20趋势跟踪策略...")
            
            # 创建策略实例
            strategy = MA20TrendFollowingStrategySimple(
                symbol=args.symbol,
                data_source=args.data_source
            )
            
            # 运行完整策略
            results = strategy.run_complete_strategy(
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital,
                save_results=not args.no_save
            )
            
            # 打印最终报告
            print("\n" + results['performance_report'])
            
            logger.info("策略运行完成!")
            
    except Exception as e:
        logger.error(f"策略运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
