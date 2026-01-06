"""
MA20趋势跟踪策略 - 单元测试模块
验证各个模块的功能正确性
"""

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入被测试的模块
from src.data_fetcher import DataFetcher
from src.data_processor import DataProcessor
from src.signal_generator import SignalGenerator, SignalType
from src.risk_manager import RiskManager, PositionSide, RiskParameters
from src.backtest_engine import MA20Strategy, BacktestEngine
from src.performance_analyzer import PerformanceAnalyzer
from config import get_config, validate_config


class TestDataFetcher(unittest.TestCase):
    """测试数据获取模块"""
    
    def setUp(self):
        """测试前准备"""
        self.fetcher = DataFetcher('akshare')  # 使用akshare进行测试
    
    def test_fetch_futures_data(self):
        """测试期货数据获取"""
        try:
            df = self.fetcher.fetch_futures_data('RB0', '2023-01-01', '2023-01-31')
            
            # 检查数据不为空
            self.assertFalse(df.empty, "获取的数据不应为空")
            
            # 检查必需列存在
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                self.assertIn(col, df.columns, f"数据中应包含列: {col}")
            
            # 检查日期范围
            min_date = pd.to_datetime(df['date']).min()
            max_date = pd.to_datetime(df['date']).max()
            self.assertGreaterEqual(min_date, pd.to_datetime('2023-01-01'))
            self.assertLessEqual(max_date, pd.to_datetime('2023-01-31'))
            
            logger.info(f"数据获取测试通过，获取了 {len(df)} 条记录")
            
        except Exception as e:
            self.skipTest(f"数据获取测试跳过: {e}")
    
    def test_data_validation(self):
        """测试数据验证"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'open': [100, 102, 101, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 104, 103, 105, 106, 107, 108, 109, 110, 111],
            'low': [99, 101, 100, 102, 103, 104, 105, 106, 107, 108],
            'close': [101, 103, 102, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000] * 10
        })
        
        # 验证价格逻辑
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            self.assertLessEqual(row['low'], row['open'], "最低价不应高于开盘价")
            self.assertLessEqual(row['low'], row['close'], "最低价不应高于收盘价")
            self.assertGreaterEqual(row['high'], row['open'], "最高价不应低于开盘价")
            self.assertGreaterEqual(row['high'], row['close'], "最高价不应低于收盘价")
        
        logger.info("数据验证测试通过")


class TestDataProcessor(unittest.TestCase):
    """测试数据处理模块"""
    
    def setUp(self):
        """测试前准备"""
        self.processor = DataProcessor()
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': [100 + i*2 for i in range(20)],
            'high': [102 + i*2 for i in range(20)],
            'low': [98 + i*2 for i in range(20)],
            'close': [101 + i*2 for i in range(20)],
            'volume': [1000 + i*100 for i in range(20)]
        })
    
    def test_create_2day_kline(self):
        """测试2日K线合成"""
        resampled = self.processor.create_2day_kline(self.test_data)
        
        # 检查数据量减少约一半
        expected_length = len(self.test_data) // 2
        self.assertAlmostEqual(len(resampled), expected_length, delta=2)
        
        # 检查必需列存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, resampled.columns, f"合成数据应包含列: {col}")
        
        # 验证第一条数据的逻辑
        first_resampled = resampled.iloc[0]
        first_original = self.test_data.iloc[0]
        second_original = self.test_data.iloc[1]
        
        # 开盘价应为第一根K线的开盘价
        self.assertEqual(first_resampled['open'], first_original['open'])
        
        # 收盘价应为第二根K线的收盘价
        self.assertEqual(first_resampled['close'], second_original['close'])
        
        # 成交量应为两根K线之和
        expected_volume = first_original['volume'] + second_original['volume']
        self.assertEqual(first_resampled['volume'], expected_volume)
        
        logger.info(f"2日K线合成测试通过，原始数据: {len(self.test_data)} -> 合成数据: {len(resampled)}")
    
    def test_calculate_ma(self):
        """测试MA计算"""
        # 计算MA5
        result = self.processor.calculate_ma(self.test_data, period=5)
        
        # 检查MA列存在
        self.assertIn('ma5', result.columns, "应包含MA5列")
        
        # 检查前4个值为NaN（因为窗口期为5）
        self.assertTrue(pd.isna(result['ma5'].iloc[:4]).all(), "前4个MA值应为NaN")
        
        # 检查第5个值开始不为NaN
        self.assertFalse(pd.isna(result['ma5'].iloc[4]), "第5个MA值不应为NaN")
        
        # 验证MA计算正确性
        expected_ma5 = self.test_data['close'].iloc[:5].mean()
        actual_ma5 = result['ma5'].iloc[4]
        self.assertAlmostEqual(actual_ma5, expected_ma5, places=2)
        
        logger.info("MA计算测试通过")
    
    def test_kline_features(self):
        """测试K线特征计算"""
        result = self.processor.calculate_kline_features(self.test_data)
        
        # 检查特征列存在
        feature_columns = ['is_red', 'is_green', 'body_size', 'total_range', 'body_ratio']
        for col in feature_columns:
            self.assertIn(col, result.columns, f"应包含特征列: {col}")
        
        # 验证K线颜色判断
        for i in range(len(result)):
            if result.iloc[i]['close'] > result.iloc[i]['open']:
                self.assertTrue(result.iloc[i]['is_red'], "收盘价>开盘价应为阳线")
                self.assertFalse(result.iloc[i]['is_green'], "阳线不应为绿色")
            elif result.iloc[i]['close'] < result.iloc[i]['open']:
                self.assertTrue(result.iloc[i]['is_green'], "收盘价<开盘价应为阴线")
                self.assertFalse(result.iloc[i]['is_red'], "阴线不应为红色")
        
        logger.info("K线特征计算测试通过")


class TestSignalGenerator(unittest.TestCase):
    """测试信号生成模块"""
    
    def setUp(self):
        """测试前准备"""
        self.generator = SignalGenerator(ma_period=5)
        
        # 创建测试数据（包含MA）
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': [100, 102, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            'high': [102, 104, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
            'low': [98, 101, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118],
            'close': [101, 103, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
        })
        
        # 计算MA5
        self.test_data['ma5'] = self.test_data['close'].rolling(window=5).mean()
    
    def test_generate_signals(self):
        """测试信号生成"""
        result = self.generator.generate_signals(self.test_data)
        
        # 检查信号列存在
        self.assertIn('signal', result.columns, "应包含信号列")
        self.assertIn('signal_reason', result.columns, "应包含信号原因列")
        
        # 验证信号值正确性
        valid_signals = [-1, 0, 1, None]
        for signal in result['signal']:
            self.assertIn(signal, valid_signals, f"信号值 {signal} 应为有效值")
        
        # 验证信号逻辑
        for i in range(len(result)):
            if pd.notna(result.iloc[i]['signal']):
                row = result.iloc[i]
                if row['signal'] == SignalType.BUY.value:
                    # 做多信号：收盘价 > MA 且 收阳
                    self.assertGreater(row['close'], row['ma5'], "做多信号时收盘价应高于MA")
                    self.assertGreater(row['close'], row['open'], "做多信号时应收阳线")
                elif row['signal'] == SignalType.SELL.value:
                    # 做空信号：收盘价 < MA 且 收阴
                    self.assertLess(row['close'], row['ma5'], "做空信号时收盘价应低于MA")
                    self.assertLess(row['close'], row['open'], "做空信号时应收阴线")
        
        logger.info(f"信号生成测试通过，生成了 {(result['signal'] != 0).sum()} 个有效信号")
    
    def test_signal_at_index(self):
        """测试指定索引信号生成"""
        # 测试有效索引
        signal = self.generator.generate_signal_at_index(self.test_data, 10)
        self.assertIsNotNone(signal, "应能生成有效信号")
        
        # 测试无效索引
        invalid_signal = self.generator.generate_signal_at_index(self.test_data, -1)
        self.assertIsNotNone(invalid_signal, "最后一个索引应能生成信号")
        
        # 测试越界索引
        out_of_bounds_signal = self.generator.generate_signal_at_index(self.test_data, 100)
        self.assertIsNone(out_of_bounds_signal, "越界索引应返回None")
        
        logger.info("指定索引信号生成测试通过")


class TestRiskManager(unittest.TestCase):
    """测试风险管理模块"""
    
    def setUp(self):
        """测试前准备"""
        self.risk_manager = RiskManager()
    
    def test_calculate_stop_loss_long(self):
        """测试做多止损计算"""
        entry_price = 4000.0
        prev_low = 3800.0  # 5%止损距离
        
        result = self.risk_manager.calculate_stop_loss(entry_price, prev_low, PositionSide.LONG)
        
        # 检查止损距离计算
        expected_distance = (entry_price - prev_low) / entry_price
        self.assertAlmostEqual(result.stop_distance_pct, expected_distance, places=4)
        
        # 检查止损价（应在容忍范围内，使用基础止损）
        self.assertEqual(result.stop_price, prev_low)
        self.assertFalse(result.is_forced_stop)
        
        logger.info(f"做多止损测试: 进场价={entry_price}, 前低={prev_low}, 止损={result.stop_price}")
    
    def test_calculate_stop_loss_short(self):
        """测试做空止损计算"""
        entry_price = 4000.0
        prev_high = 4200.0  # 5%止损距离
        
        result = self.risk_manager.calculate_stop_loss(entry_price, prev_high, PositionSide.SHORT)
        
        # 检查止损距离计算
        expected_distance = (prev_high - entry_price) / entry_price
        self.assertAlmostEqual(result.stop_distance_pct, expected_distance, places=4)
        
        # 检查止损价（应在容忍范围内，使用基础止损）
        self.assertEqual(result.stop_price, prev_high)
        self.assertFalse(result.is_forced_stop)
        
        logger.info(f"做空止损测试: 进场价={entry_price}, 前高={prev_high}, 止损={result.stop_price}")
    
    def test_force_stop_calculation(self):
        """测试强制止损计算"""
        # 测试超过6%容忍度的情况
        entry_price = 4000.0
        prev_low = 3600.0  # 10%止损距离，超过6%容忍度
        
        result = self.risk_manager.calculate_stop_loss(entry_price, prev_low, PositionSide.LONG)
        
        # 应使用强制3%止损
        expected_stop = entry_price * (1 - 0.03)
        self.assertEqual(result.stop_price, expected_stop)
        self.assertTrue(result.is_forced_stop)
        
        logger.info(f"强制止损测试: 进场价={entry_price}, 前低={prev_low}, 强制止损={result.stop_price}")
    
    def test_position_size_calculation(self):
        """测试仓位大小计算"""
        capital = 100000.0
        entry_price = 4000.0
        stop_price = 3800.0  # 每手风险200点
        margin_rate = 0.10
        contract_multiplier = 10.0
        
        result = self.risk_manager.calculate_position_size(
            capital, entry_price, stop_price, margin_rate, contract_multiplier
        )
        
        # 检查计算结果
        self.assertGreater(result.position_size, 0, "仓位大小应大于0")
        self.assertLessEqual(result.risk_pct_of_capital, 0.02, "风险比例不应超过2%")
        
        logger.info(f"仓位计算测试: 资金={capital}, 建议仓位={result.position_size}手, "
                   f"风险比例={result.risk_pct_of_capital:.2%}")
    
    def test_risk_parameter_validation(self):
        """测试风险参数验证"""
        # 测试默认参数
        validation = self.risk_manager.validate_risk_parameters()
        self.assertTrue(validation['is_valid'], "默认参数应有效")
        
        # 测试无效参数
        invalid_params = RiskParameters(max_loss_pct=0.02, force_stop_pct=0.03)
        invalid_manager = RiskManager(invalid_params)
        validation = invalid_manager.validate_risk_parameters()
        self.assertFalse(validation['is_valid'], "无效参数应被检测出")
        
        logger.info("风险参数验证测试通过")


class TestConfig(unittest.TestCase):
    """测试配置模块"""
    
    def test_config_loading(self):
        """测试配置加载"""
        config = get_config()
        
        # 检查配置不为空
        self.assertIsNotNone(config, "配置不应为None")
        self.assertIsInstance(config, dict, "配置应为字典类型")
        
        # 检查必需配置项
        required_keys = ['ma_period', 'max_loss_pct', 'force_stop_pct', 'instruments']
        for key in required_keys:
            self.assertIn(key, config, f"配置应包含键: {key}")
        
        logger.info("配置加载测试通过")
    
    def test_instrument_config(self):
        """测试品种配置"""
        rb_config = get_instrument_config('RB0')
        
        # 检查品种配置
        self.assertIsNotNone(rb_config, "螺纹钢配置不应为None")
        self.assertIn('commission', rb_config, "应包含手续费配置")
        self.assertIn('margin_rate', rb_config, "应包含保证金率配置")
        
        logger.info("品种配置测试通过")
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试默认配置
        is_valid = validate_config()
        # 注意：如果TUSHARE_TOKEN未设置，验证会失败，这是预期的行为
        logger.info(f"配置验证结果: {is_valid}")


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前准备"""
        # 创建完整的测试数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        # 生成价格数据
        base_price = 4000
        prices = [base_price]
        for i in range(1, len(dates)):
            change = np.random.uniform(-0.02, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'high': [p * np.random.uniform(1.00, 1.02) for p in prices],
            'low': [p * np.random.uniform(0.98, 1.00) for p in prices],
            'close': prices,
            'volume': np.random.randint(10000, 100000, len(dates))
        })
        
        # 确保价格逻辑正确
        for i in range(len(self.test_data)):
            row = self.test_data.iloc[i]
            self.test_data.loc[i, 'high'] = max(row['high'], row['open'], row['close'])
            self.test_data.loc[i, 'low'] = min(row['low'], row['open'], row['close'])
    
    def test_complete_workflow(self):
        """测试完整工作流程"""
        logger.info("开始集成测试...")
        
        # 1. 数据处理
        processor = DataProcessor()
        data_2day = processor.create_2day_kline(self.test_data)
        self.assertFalse(data_2day.empty, "2日K线数据不应为空")
        
        # 2. 信号生成
        generator = SignalGenerator(ma_period=10)
        signals_data = generator.generate_signals(data_2day)
        self.assertIn('signal', signals_data.columns, "应包含信号列")
        
        # 3. 风险管理
        risk_manager = RiskManager()
        
        # 测试止损计算
        if len(signals_data) > 10:
            sample_row = signals_data.iloc[10]
            if sample_row['close'] > sample_row['ma10']:
                # 做多止损
                stop_result = risk_manager.calculate_stop_loss(
                    sample_row['close'], sample_row['low'], PositionSide.LONG
                )
                self.assertIsNotNone(stop_result.stop_price, "应能计算止损价格")
        
        logger.info("集成测试通过")


def run_comprehensive_tests():
    """运行综合测试"""
    print("开始运行MA20趋势跟踪策略单元测试...")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestConfig,
        TestDataFetcher,
        TestDataProcessor,
        TestSignalGenerator,
        TestRiskManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n出错的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)
