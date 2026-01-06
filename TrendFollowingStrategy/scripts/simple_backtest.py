"""
MA20趋势跟踪策略 - 简化回测测试
验证策略逻辑而不使用Backtrader
注意：
- 本回测仅用于验证策略逻辑，使用的数据是生成的！
- 真实数据请使用main_simple.py或者multibacktest.py进行回测。
"""

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """创建测试数据"""
    # 生成2023年上半年的模拟数据
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='2D')  # 2日K线
    n = len(dates)
    
    # 生成价格数据（趋势+随机波动）
    np.random.seed(42)
    base_price = 4000
    trend = np.linspace(0, 200, n)  # 上升趋势
    noise = np.cumsum(np.random.normal(0, 20, n))  # 随机游走
    prices = base_price + trend + noise
    
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
    
    return df

def simple_backtest(data, initial_capital=100000, ma_period=20, commission=0.0003, slippage=0.001):
    """简化回测函数"""
    logger.info("开始简化回测...")
    
    # 准备数据
    from src.data_processor import DataProcessor
    processor = DataProcessor()
    
    # 计算MA
    data_with_ma = processor.calculate_ma(data, period=ma_period)
    
    # 生成信号
    from signal_generator import SignalGenerator
    generator = SignalGenerator(ma_period=ma_period)
    signals_data = generator.generate_signals(data_with_ma)
    
    # 初始化回测状态
    capital = initial_capital
    position = 0  # 持仓数量
    entry_price = 0
    stop_price = 0
    trades = []
    equity_curve = [initial_capital]
    
    # 回测逻辑
    for i in range(len(signals_data)):
        row = signals_data.iloc[i]
        current_price = row['close']
        signal = row['signal']
        
        # 无持仓时检查信号
        if position == 0:
            if signal == 1:  # 做多信号
                # 计算止损
                from src.risk_manager import RiskManager, PositionSide
                risk_manager = RiskManager()
                
                # 使用前一根K线的极值
                prev_low = signals_data.iloc[i-1]['low'] if i > 0 else row['low']
                stop_result = risk_manager.calculate_stop_loss(
                    entry_price=current_price,
                    prev_extreme=prev_low,
                    direction=PositionSide.LONG
                )
                
                # 计算仓位
                position_result = risk_manager.calculate_position_size(
                    capital=capital,
                    entry_price=current_price,
                    stop_price=stop_result.stop_price,
                    margin_rate=0.10,
                    contract_multiplier=10.0
                )
                
                # 开仓
                position = position_result.position_size
                entry_price = current_price
                stop_price = stop_result.stop_price
                
                # 扣除手续费
                commission_cost = entry_price * position * 10 * commission
                capital -= commission_cost
                
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
                from risk_manager import RiskManager, PositionSide
                risk_manager = RiskManager()
                
                # 使用前一根K线的极值
                prev_high = signals_data.iloc[i-1]['high'] if i > 0 else row['high']
                stop_result = risk_manager.calculate_stop_loss(
                    entry_price=current_price,
                    prev_extreme=prev_high,
                    direction=PositionSide.SHORT
                )
                
                # 计算仓位
                position_result = risk_manager.calculate_position_size(
                    capital=capital,
                    entry_price=current_price,
                    stop_price=stop_result.stop_price,
                    margin_rate=0.10,
                    contract_multiplier=10.0
                )
                
                # 开仓
                position = -position_result.position_size  # 负值表示做空
                entry_price = current_price
                stop_price = stop_result.stop_price
                
                # 扣除手续费
                commission_cost = entry_price * abs(position) * 10 * commission
                capital -= commission_cost
                
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
                    pnl = (exit_price - entry_price) * position * 10
                    capital += pnl
                    
                    # 扣除手续费
                    commission_cost = exit_price * abs(position) * 10 * commission
                    capital -= commission_cost
                    
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
                    pnl = (entry_price - exit_price) * abs(position) * 10
                    capital += pnl
                    
                    # 扣除手续费
                    commission_cost = exit_price * abs(position) * 10 * commission
                    capital -= commission_cost
                    
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
    
    # 计算绩效指标
    total_return = (capital - initial_capital) / initial_capital
    winning_trades = len([t for t in trades if 'pnl' in t and t['pnl'] > 0])
    losing_trades = len([t for t in trades if 'pnl' in t and t['pnl'] < 0])
    total_trades = winning_trades + losing_trades
    
    # 计算胜率
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
    
    return results

def main():
    """主函数"""
    logger.info("开始MA20趋势跟踪策略简化回测测试...")
    
    # 创建测试数据
    test_data = create_test_data()
    logger.info(f"✓ 创建测试数据: {len(test_data)} 条记录")
    
    # 运行简化回测
    results = simple_backtest(test_data)
    
    # 打印结果
    print("\n" + "="*50)
    print("           简化回测结果")
    print("="*50)
    print(f"初始资金: {results['initial_capital']:,.2f} CNY")
    print(f"最终资金: {results['final_capital']:,.2f} CNY")
    print(f"总收益率: {results['total_return']*100:+.2f}%")
    print(f"总交易次数: {results['total_trades']}")
    print(f"盈利交易: {results['winning_trades']}")
    print(f"亏损交易: {results['losing_trades']}")
    print(f"胜率: {results['win_rate']*100:.2f}%")
    print(f"盈亏比: {results['profit_factor']:.2f}")
    print(f"平均盈利: {results['avg_win']:,.2f} CNY")
    print(f"平均亏损: {results['avg_loss']:,.2f} CNY")
    print("="*50)
    
    # 显示前几个交易
    if results['trades']:
        print(f"\n前5个交易:")
        for i, trade in enumerate(results['trades'][:5]):
            if 'pnl' in trade:
                print(f"{i+1}. {trade['date'].strftime('%Y-%m-%d')} - {trade['type']} - "
                      f"价格: {trade['price']:.2f} - 盈亏: {trade['pnl']:,.2f}")
            else:
                print(f"{i+1}. {trade['date'].strftime('%Y-%m-%d')} - {trade['type']} - "
                      f"价格: {trade['price']:.2f} - 开仓")
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"\n✅ 简化回测测试完成!")
    print(f"策略在测试期间实现了 {results['total_return']*100:+.2f}% 的收益率")
    print(f"共进行了 {results['total_trades']} 笔交易，胜率 {results['win_rate']*100:.2f}%")
