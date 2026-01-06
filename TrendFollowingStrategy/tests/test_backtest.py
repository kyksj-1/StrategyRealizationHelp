"""
MA20趋势跟踪策略 - 回测功能测试
验证回测引擎的基本功能
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

def test_backtest_engine():
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

def test_backtest_engine():
    """测试回测引擎"""
    logger.info("开始测试MA20趋势跟踪策略回测引擎...")
    
    # 1. 创建测试数据
    logger.info("1. 创建测试数据...")
    test_data = create_test_data()
    logger.info(f"✓ 创建测试数据: {len(test_data)} 条记录")
    
    # 2. 准备策略数据
    logger.info("2. 准备策略数据...")
    from src.data_processor import DataProcessor
    processor = DataProcessor()
    
    # 计算MA20
    data_with_ma = processor.calculate_ma(test_data, period=20)
    logger.info(f"✓ MA20计算完成")
    
    # 3. 生成信号
    logger.info("3. 生成交易信号...")
    from src.signal_generator import SignalGenerator
    generator = SignalGenerator(ma_period=20)
    signals_data = generator.generate_signals(data_with_ma)
    
    buy_signals = (signals_data['signal'] == 1).sum()
    sell_signals = (signals_data['signal'] == -1).sum()
    logger.info(f"✓ 信号生成: 做多{buy_signals}个, 做空{sell_signals}个")
    
    # 4. 运行回测
    logger.info("4. 运行回测...")
    from src.backtest_engine import BacktestEngine
    
    engine = BacktestEngine('RB0')
    
    try:
        # 运行回测
        results = engine.run_backtest(signals_data, initial_capital=100000)
        
        # 提取结果
        basic_info = results.get('basic_info', {})
        final_value = basic_info.get('final_value', 0)
        total_return = basic_info.get('total_return', 0)
        total_trades = basic_info.get('total_trades', 0)
        
        logger.info(f"✓ 回测完成:")
        logger.info(f"  初始资金: 100,000 CNY")
        logger.info(f"  最终资产: {final_value:,.2f} CNY")
        logger.info(f"  总收益率: {total_return*100:+.2f}%")
        logger.info(f"  总交易次数: {total_trades}")
        
        # 打印简要报告
        engine.print_backtest_report(results)
        
        return {
            'success': True,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
        
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def test_risk_management():
    """测试风险管理功能"""
    logger.info("\n5. 测试风险管理功能...")
    from src.risk_manager import RiskManager, PositionSide
    
    risk_manager = RiskManager()
    
    # 测试做多止损
    stop_result = risk_manager.calculate_stop_loss(
        entry_price=4200.0,
        prev_extreme=4000.0,
        direction=PositionSide.LONG
    )
    logger.info(f"✓ 做多止损: 进场价4200.0, 止损价{stop_result.stop_price:.2f}")
    
    # 测试做空止损
    stop_result = risk_manager.calculate_stop_loss(
        entry_price=4200.0,
        prev_extreme=4400.0,
        direction=PositionSide.SHORT
    )
    logger.info(f"✓ 做空止损: 进场价4200.0, 止损价{stop_result.stop_price:.2f}")
    
    # 测试强制止损（超过6%容忍度）
    stop_result = risk_manager.calculate_stop_loss(
        entry_price=4200.0,
        prev_extreme=3600.0,  # 14.3%止损距离，超过6%容忍度
        direction=PositionSide.LONG
    )
    logger.info(f"✓ 强制止损: 进场价4200.0, 止损价{stop_result.stop_price:.2f} (强制3%止损)")
    
    # 测试仓位计算
    position_result = risk_manager.calculate_position_size(
        capital=100000.0,
        entry_price=4200.0,
        stop_price=4000.0,
        margin_rate=0.10,
        contract_multiplier=10.0
    )
    logger.info(f"✓ 仓位计算: 建议{position_result.position_size}手, 风险比例{position_result.risk_pct_of_capital:.2%}")

if __name__ == "__main__":
    try:
        # 运行回测测试
        backtest_results = test_backtest_engine()
        
        # 运行风险管理测试
        test_risk_management()
        
        if backtest_results['success']:
            print(f"\n🎉 回测引擎测试完成!")
            print(f"最终资产: {backtest_results['final_value']:,.2f} CNY")
            print(f"总收益率: {backtest_results['total_return']*100:+.2f}%")
            print(f"总交易次数: {backtest_results['total_trades']}")
            print(f"做多信号: {backtest_results['buy_signals']}")
            print(f"做空信号: {backtest_results['sell_signals']}")
        else:
            print(f"\n❌ 回测测试失败: {backtest_results['error']}")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
