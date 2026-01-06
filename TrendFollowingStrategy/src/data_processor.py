"""
MA20趋势跟踪策略 - 数据处理器
实现2日K线合成、技术指标计算等功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
from config import get_config, RESAMPLE_CONFIG

# 设置日志
logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器 - 负责K线合成和技术指标计算"""
    
    def __init__(self):
        """初始化数据处理器"""
        self.config = get_config()
        self.resample_config = RESAMPLE_CONFIG
    
    def create_2day_kline(self, df: pd.DataFrame) -> pd.DataFrame:
        """将日K线合成为2日K线
        
        合成规则：
        - Open: 两日中第一根的开盘价
        - High: 两日中的最高价
        - Low: 两日中的最低价
        - Close: 两日中最后一根的收盘价
        - Volume: 两日成交量之和
        - Amount: 两日成交金额之和（如果有）
        
        Args:
            df: 日K线数据DataFrame，必须包含['date', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            2日K线数据DataFrame
        """
        logger.info("开始合成2日K线数据...")
        
        # 验证输入数据
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"输入数据必须包含列: {required_columns}")
        
        # 确保日期格式正确
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 设置日期为索引
        df_indexed = df.set_index('date')
        
        # 使用resample进行2日K线合成
        aggregation_rules = self.resample_config['aggregation_rules'].copy()
        
        # 只保留数据中存在的列
        available_columns = [col for col in aggregation_rules.keys() if col in df_indexed.columns]
        aggregation_rules = {col: aggregation_rules[col] for col in available_columns}
        
        # 执行重采样
        resampled = df_indexed.resample('2D').agg(aggregation_rules)
        
        # 删除空行（节假日等）
        resampled = resampled.dropna()
        
        # 重置索引
        resampled = resampled.reset_index()
        
        # 验证合成结果
        self._validate_resampled_data(df, resampled)
        
        logger.info(f"2日K线合成完成: {len(resampled)} 条记录")
        return resampled
    
    def calculate_ma(self, df: pd.DataFrame, period: int = 20, price_col: str = 'close') -> pd.DataFrame:
        """计算移动平均线
        
        Args:
            df: 数据DataFrame
            period: MA周期
            price_col: 价格列名
            
        Returns:
            添加了MA的DataFrame
        """
        logger.info(f"计算MA{period}...")
        
        if price_col not in df.columns:
            raise ValueError(f"数据中不存在列: {price_col}")
        
        # 计算简单移动平均线
        ma_col = f'ma{period}'
        df[ma_col] = df[price_col].rolling(window=period, min_periods=period).mean()
        
        logger.info(f"MA{period}计算完成")
        return df
    
    def calculate_kline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算K线特征
        
        Args:
            df: 数据DataFrame
            
        Returns:
            添加了K线特征的DataFrame
        """
        logger.info("计算K线特征...")
        
        # K线颜色（阴阳）
        df['is_red'] = df['close'] > df['open']  # 阳线
        df['is_green'] = df['close'] < df['open']  # 阴线
        df['is_doji'] = df['close'] == df['open']  # 十字星
        
        # K线实体大小
        df['body_size'] = abs(df['close'] - df['open'])
        df['total_range'] = df['high'] - df['low']
        df['body_ratio'] = df['body_size'] / df['total_range']
        
        # 上影线和下影线
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        logger.info("K线特征计算完成")
        return df
    
    def calculate_price_position(self, df: pd.DataFrame, ma_period: int = 20) -> pd.DataFrame:
        """计算价格相对位置
        
        Args:
            df: 数据DataFrame
            ma_period: MA周期
            
        Returns:
            添加了价格位置的DataFrame
        """
        logger.info("计算价格相对位置...")
        
        ma_col = f'ma{ma_period}'
        if ma_col not in df.columns:
            df = self.calculate_ma(df, ma_period)
        
        # 价格在MA上方/下方
        df['above_ma'] = df['close'] > df[ma_col]
        df['below_ma'] = df['close'] < df[ma_col]
        
        # 距离MA的百分比
        df['distance_to_ma'] = (df['close'] - df[ma_col]) / df[ma_col]
        
        logger.info("价格相对位置计算完成")
        return df
    
    def prepare_strategy_data(self, df: pd.DataFrame, ma_period: int = 20) -> pd.DataFrame:
        """准备策略所需的所有数据
        
        Args:
            df: 原始数据DataFrame
            ma_period: MA周期
            
        Returns:
            完整的策略数据DataFrame
        """
        logger.info("准备策略数据...")
        
        # 1. 计算MA
        df = self.calculate_ma(df, ma_period)
        
        # 2. 计算K线特征
        df = self.calculate_kline_features(df)
        
        # 3. 计算价格位置
        df = self.calculate_price_position(df, ma_period)
        
        # 4. 删除包含NaN的行（前period-1行）
        df = df.dropna()
        
        logger.info(f"策略数据准备完成: {len(df)} 条有效记录")
        return df
    
    def _validate_resampled_data(self, original_df: pd.DataFrame, resampled_df: pd.DataFrame) -> None:
        """验证重采样数据的正确性
        
        Args:
            original_df: 原始日K线数据
            resampled_df: 合成的2日K线数据
        """
        logger.info("验证2日K线合成结果...")
        
        # 检查1: 数据量应该约为原来的一半
        expected_ratio = len(resampled_df) / len(original_df)
        if not (0.4 <= expected_ratio <= 0.6):
            logger.warning(f"数据量比例异常: {expected_ratio:.2f} (期望约0.5)")
        
        # 检查2: 价格逻辑验证
        invalid_high = resampled_df['high'] < resampled_df[['open', 'close']].max(axis=1)
        invalid_low = resampled_df['low'] > resampled_df[['open', 'close']].min(axis=1)
        
        if invalid_high.any():
            logger.error(f"发现 {invalid_high.sum()} 条high价格异常")
        
        if invalid_low.any():
            logger.error(f"发现 {invalid_low.sum()} 条low价格异常")
        
        # 检查3: 时间连续性（简单检查）
        if len(resampled_df) > 1:
            date_diffs = resampled_df['date'].diff().dropna()
            expected_diff = pd.Timedelta(days=2)
            
            # 允许有一定的偏差（节假日等）
            abnormal_diffs = date_diffs[date_diffs != expected_diff]
            if len(abnormal_diffs) > len(resampled_df) * 0.1:  # 超过10%认为异常
                logger.warning(f"时间间隔异常的数据占比: {len(abnormal_diffs)/len(resampled_df):.2%}")
        
        # 检查4: 随机抽样验证（前5条数据）
        if len(resampled_df) >= 2:
            logger.info("抽样验证前几条2日K线数据:")
            for i in range(min(3, len(resampled_df))):
                row = resampled_df.iloc[i]
                logger.info(f"日期: {row['date']}, O: {row['open']:.2f}, H: {row['high']:.2f}, "
                          f"L: {row['low']:.2f}, C: {row['close']:.2f}, V: {row['volume']:.0f}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据摘要信息
        
        Args:
            df: 数据DataFrame
            
        Returns:
            摘要信息字典
        """
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'trading_days': len(df)
            },
            'price_stats': {
                'highest': df['high'].max(),
                'lowest': df['low'].min(),
                'avg_close': df['close'].mean(),
                'price_range': df['high'].max() - df['low'].min()
            },
            'volume_stats': {
                'total_volume': df['volume'].sum(),
                'avg_volume': df['volume'].mean(),
                'max_volume': df['volume'].max()
            }
        }
        
        # 如果有K线特征，添加统计
        if 'is_red' in df.columns:
            red_ratio = df['is_red'].sum() / len(df)
            summary['kline_stats'] = {
                'red_ratio': red_ratio,
                'green_ratio': 1 - red_ratio - (df['is_doji'].sum() / len(df) if 'is_doji' in df.columns else 0),
                'doji_ratio': df['is_doji'].sum() / len(df) if 'doji' in df.columns else 0
            }
        
        return summary


def test_data_processor():
    """测试数据处理器"""
    print("测试数据处理器...")
    
    # 创建测试数据
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    np.random.seed(42)
    
    # 生成模拟价格数据
    base_price = 4000
    prices = [base_price]
    
    for i in range(1, len(dates)):
        # 随机价格变动 -2% 到 +2%
        change = np.random.uniform(-0.02, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 创建测试DataFrame
    test_data = pd.DataFrame({
        'date': dates,
        'open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'high': [p * np.random.uniform(1.00, 1.02) for p in prices],
        'low': [p * np.random.uniform(0.98, 1.00) for p in prices],
        'close': prices,
        'volume': np.random.randint(10000, 100000, len(dates))
    })
    
    # 确保价格逻辑正确
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        test_data.loc[i, 'high'] = max(row['high'], row['open'], row['close'])
        test_data.loc[i, 'low'] = min(row['low'], row['open'], row['close'])
    
    processor = DataProcessor()
    
    # 测试2日K线合成
    print("\n1. 测试2日K线合成:")
    resampled = processor.create_2day_kline(test_data)
    print(f"原始数据: {len(test_data)} 条")
    print(f"合成后: {len(resampled)} 条")
    print(resampled.head())
    
    # 测试MA计算
    print("\n2. 测试MA20计算:")
    with_ma = processor.calculate_ma(resampled, 20)
    print(with_ma[['date', 'close', 'ma20']].head())
    
    # 测试K线特征
    print("\n3. 测试K线特征计算:")
    with_features = processor.calculate_kline_features(resampled)
    print(with_features[['date', 'close', 'is_red', 'body_ratio']].head())
    
    # 测试完整数据准备
    print("\n4. 测试完整数据准备:")
    strategy_data = processor.prepare_strategy_data(test_data, 20)
    print(f"策略数据列: {list(strategy_data.columns)}")
    print(strategy_data.head())
    
    # 测试数据摘要
    print("\n5. 测试数据摘要:")
    summary = processor.get_data_summary(strategy_data)
    print(f"数据摘要: {summary}")
    
    print("\n数据处理器测试完成!")


if __name__ == "__main__":
    test_data_processor()