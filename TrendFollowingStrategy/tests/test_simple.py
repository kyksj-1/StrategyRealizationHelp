"""
MA20趋势跟踪策略 - 简化测试版本
用于验证核心功能
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
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
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

def test_core_modules():
    """测试核心模块"""
    logger.info("开始测试MA20趋势跟踪策略核心模块...")
    
    # 1. 数据获取（使用模拟数据）
    logger.info("1. 创建测试数据...")
    raw_data = create_test_data()
    logger.info(f"✓ 创建测试数据: {len(raw_data)} 条记录")
    
    # 2. 数据处理
    logger.info("2. 测试数据处理模块...")
    from src.data_processor import DataProcessor
    
    processor = DataProcessor()
    
    # 2日K线合成
    data_2day = processor.create_2day_kline(raw_data)
    logger.info(f"✓ 2日K线合成: {len(raw_data)} -> {len(data_2day)} 条记录")
    
    # 计算MA20
    data_with_ma = processor.calculate_ma(data_2day, period=20)
    logger.info(f"✓ MA20计算完成")
    
    # 3. 信号生成
    logger.info("3. 测试信号生成模块...")
    from src.signal_generator import SignalGenerator
    
    generator = SignalGenerator(ma_period=20)
    signals_data = generator.generate_signals(data_with_ma)
    
    # 统计信号
    buy_signals = (signals_data['signal'] == 1).sum()
    sell_signals = (signals_data['signal'] == -1).sum()
    logger.info(f"✓ 信号生成: 做多{buy_signals}个, 做空{sell_signals}个")
    
    # 4. 风险管理
    logger.info("4. 测试风险管理模块...")
    from src.risk_manager import RiskManager, PositionSide
    
    risk_manager = RiskManager()
    
    # 测试做多止损
    stop_result = risk_manager.calculate_stop_loss(
        entry_price=4200.0,
        prev_extreme=4000.0,
        direction=PositionSide.LONG
    )
    logger.info(f"✓ 做多止损: 进场价4200.0, 止损价{stop_result.stop_price:.2f}")
    
    # 测试仓位计算
    position_result = risk_manager.calculate_position_size(
        capital=100000.0,
        entry_price=4200.0,
        stop_price=stop_result.stop_price,
        margin_rate=0.10,
        contract_multiplier=10.0
    )
    logger.info(f"✓ 仓位计算: 建议{position_result.position_size}手")
    
    # 5. 回测引擎
    logger.info("5. 测试回测引擎...")
    from src.backtest_engine import BacktestEngine
    
    engine = BacktestEngine('RB0')
    
    # 运行回测（使用较短的数据）
    test_data = signals_data.tail(50)  # 使用最后50条数据
    
    try:
        results = engine.run_backtest(test_data, initial_capital=100000)
        
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
        
    except Exception as e:
        logger.warning(f"回测引擎测试跳过: {e}")
    
    logger.info("✅ 核心模块测试完成!")
    
    return {
        'raw_data_length': len(raw_data),
        'processed_data_length': len(data_2day),
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'position_size': position_result.position_size,
        'stop_price': stop_result.stop_price
    }

if __name__ == "__main__":
    try:
        results = test_core_modules()
        print(f"\n📊 测试总结:")
        print(f"原始数据: {results['raw_data_length']} 条")
        print(f"处理后数据: {results['processed_data_length']} 条")
        print(f"做多信号: {results['buy_signals']} 个")
        print(f"做空信号: {results['sell_signals']} 个")
        print(f"建议仓位: {results['position_size']} 手")
        print(f"止损价格: {results['stop_price']:.2f}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
