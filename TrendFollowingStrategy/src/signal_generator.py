"""
MA20趋势跟踪策略 - 信号生成模块
根据MA20和K线颜色生成交易信号
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 设置日志
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型枚举"""
    BUY = 1      # 做多信号
    SELL = -1    # 做空信号
    HOLD = 0     # 持仓观望
    WAIT = None  # 空仓观望


@dataclass
class TradingSignal:
    """交易信号数据结构"""
    signal_type: SignalType
    price: float
    date: pd.Timestamp
    ma_value: float
    confidence: float = 1.0
    reason: str = ""


class SignalGenerator:
    """信号生成器 - 基于MA20和K线颜色生成交易信号"""
    
    def __init__(self, ma_period: int = 20):
        """初始化信号生成器
        
        Args:
            ma_period: MA周期，默认20
        """
        self.ma_period = ma_period
        self.ma_col = f'ma{ma_period}'
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号
        
        信号规则：
        1. 做多信号：收盘价 > MA20 且 当前K线收阳（Close > Open）
        2. 做空信号：收盘价 < MA20 且 当前K线收阴（Close < Open）
        3. 其他情况：空仓观望
        
        Args:
            df: 包含价格和MA数据的DataFrame
            
        Returns:
            添加了信号列的DataFrame
        """
        logger.info(f"开始生成交易信号，MA周期: {self.ma_period}")
        
        # 验证输入数据
        required_columns = ['close', 'open', self.ma_col]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"输入数据必须包含列: {required_columns}")
        
        # 创建数据副本避免修改原数据
        result_df = df.copy()
        
        # 初始化信号列
        result_df['signal'] = SignalType.WAIT.value
        result_df['signal_reason'] = ''
        result_df['signal_confidence'] = 0.0
        
        # 判断价格在MA上方还是下方
        result_df['above_ma'] = result_df['close'] > result_df[self.ma_col]
        result_df['below_ma'] = result_df['close'] < result_df[self.ma_col]
        
        # 判断K线颜色
        result_df['is_red'] = result_df['close'] > result_df['open']  # 阳线
        result_df['is_green'] = result_df['close'] < result_df['open']  # 阴线
        
        # 生成做多信号：均线上方且收阳
        long_condition = result_df['above_ma'] & result_df['is_red']
        result_df.loc[long_condition, 'signal'] = SignalType.BUY.value
        result_df.loc[long_condition, 'signal_reason'] = '收盘价>MA20且收阳线'
        result_df.loc[long_condition, 'signal_confidence'] = 1.0
        
        # 生成做空信号：均线下方且收阴
        short_condition = result_df['below_ma'] & result_df['is_green']
        result_df.loc[short_condition, 'signal'] = SignalType.SELL.value
        result_df.loc[short_condition, 'signal_reason'] = '收盘价<MA20且收阴线'
        result_df.loc[short_condition, 'signal_confidence'] = 1.0
        
        # 统计信号数量
        buy_signals = (result_df['signal'] == SignalType.BUY.value).sum()
        sell_signals = (result_df['signal'] == SignalType.SELL.value).sum()
        total_signals = buy_signals + sell_signals
        
        logger.info(f"信号生成完成:")
        logger.info(f"  做多信号: {buy_signals} 个")
        logger.info(f"  做空信号: {sell_signals} 个")
        logger.info(f"  总信号: {total_signals} 个")
        logger.info(f"  信号频率: {total_signals/len(result_df)*100:.2f}%")
        
        return result_df
    
    def generate_signal_at_index(self, df: pd.DataFrame, index: int) -> Optional[TradingSignal]:
        """在指定索引位置生成信号（用于实时交易）
        
        Args:
            df: 数据DataFrame
            index: 索引位置
            
        Returns:
            交易信号，如果没有信号返回None
        """
        if index < 0 or index >= len(df):
            return None
        
        row = df.iloc[index]
        
        # 检查是否有足够的历史数据计算MA
        if pd.isna(row.get(self.ma_col)):
            return None
        
        # 生成信号
        signal_value = 0
        reason = ""
        
        if row['close'] > row[self.ma_col] and row['close'] > row['open']:
            signal_value = SignalType.BUY.value
            reason = f"收盘价({row['close']:.2f})>MA{self.ma_period}({row[self.ma_col]:.2f})且收阳线"
        elif row['close'] < row[self.ma_col] and row['close'] < row['open']:
            signal_value = SignalType.SELL.value
            reason = f"收盘价({row['close']:.2f})<MA{self.ma_period}({row[self.ma_col]:.2f})且收阴线"
        
        if signal_value != 0:
            return TradingSignal(
                signal_type=SignalType(signal_value),
                price=row['close'],
                date=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
                ma_value=row[self.ma_col],
                confidence=1.0,
                reason=reason
            )
        
        return None
    
    def add_signal_filters(self, df: pd.DataFrame, 
                          min_body_ratio: float = 0.3,
                          min_volume_ratio: float = 1.2) -> pd.DataFrame:
        """添加信号过滤器
        
        Args:
            df: 包含信号的数据DataFrame
            min_body_ratio: 最小K线实体比例
            min_volume_ratio: 最小成交量比例（相对于过去5日平均）
            
        Returns:
            过滤后的DataFrame
        """
        logger.info("添加信号过滤器...")
        
        # 计算K线实体比例
        if 'body_ratio' not in df.columns:
            df['body_size'] = abs(df['close'] - df['open'])
            df['total_range'] = df['high'] - df['low']
            df['body_ratio'] = df['body_size'] / df['total_range']
        
        # 计算成交量移动平均
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        
        # 过滤条件
        strong_body = df['body_ratio'] >= min_body_ratio
        high_volume = df['volume'] >= (df['volume_ma5'] * min_volume_ratio)
        
        # 应用过滤器
        original_signals = df['signal'].copy()
        
        # 过滤做多信号
        buy_mask = (df['signal'] == SignalType.BUY.value)
        df.loc[buy_mask & (~strong_body | ~high_volume), 'signal'] = SignalType.WAIT.value
        
        # 过滤做空信号
        sell_mask = (df['signal'] == SignalType.SELL.value)
        df.loc[sell_mask & (~strong_body | ~high_volume), 'signal'] = SignalType.WAIT.value
        
        # 更新原因
        filtered_signals = (original_signals != df['signal']).sum()
        logger.info(f"过滤了 {filtered_signals} 个弱信号")
        
        return df
    
    def get_signal_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取信号统计信息
        
        Args:
            df: 包含信号的数据DataFrame
            
        Returns:
            统计信息字典
        """
        # 信号计数
        buy_signals = (df['signal'] == SignalType.BUY.value).sum()
        sell_signals = (df['signal'] == SignalType.SELL.value).sum()
        wait_signals = df['signal'].isna().sum() + (df['signal'] == SignalType.WAIT.value).sum()
        
        # 信号频率
        total_records = len(df)
        
        # 信号分布（按年份/月份）
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['year'] = df_copy['date'].dt.year
        df_copy['month'] = df_copy['date'].dt.month
        
        yearly_signals = df_copy.groupby('year')['signal'].value_counts().unstack(fill_value=0)
        monthly_signals = df_copy.groupby('month')['signal'].value_counts().unstack(fill_value=0)
        
        stats = {
            'total_signals': buy_signals + sell_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'wait_signals': wait_signals,
            'signal_frequency': (buy_signals + sell_signals) / total_records,
            'buy_frequency': buy_signals / total_records,
            'sell_frequency': sell_signals / total_records,
            'yearly_distribution': yearly_signals.to_dict() if not yearly_signals.empty else {},
            'monthly_distribution': monthly_signals.to_dict() if not monthly_signals.empty else {}
        }
        
        return stats
    
    def plot_signal_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """绘制信号分布图
        
        Args:
            df: 包含信号的数据DataFrame
            save_path: 保存路径，如果为None则不保存
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 信号时间序列
            ax1 = axes[0, 0]
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            
            # 绘制价格和MA
            ax1.plot(df_copy['date'], df_copy['close'], label='收盘价', alpha=0.7)
            ax1.plot(df_copy['date'], df_copy[self.ma_col], label=f'MA{self.ma_period}', alpha=0.7)
            
            # 标记信号点
            buy_signals = df_copy[df_copy['signal'] == SignalType.BUY.value]
            sell_signals = df_copy[df_copy['signal'] == SignalType.SELL.value]
            
            ax1.scatter(buy_signals['date'], buy_signals['close'], 
                       color='red', marker='^', s=50, label='做多信号', zorder=5)
            ax1.scatter(sell_signals['date'], sell_signals['close'], 
                       color='green', marker='v', s=50, label='做空信号', zorder=5)
            
            ax1.set_title('价格走势与交易信号')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 信号统计
            ax2 = axes[0, 1]
            signal_counts = [
                (df['signal'] == SignalType.BUY.value).sum(),
                (df['signal'] == SignalType.SELL.value).sum(),
                df['signal'].isna().sum() + (df['signal'] == SignalType.WAIT.value).sum()
            ]
            labels = ['做多信号', '做空信号', '观望']
            colors = ['red', 'green', 'gray']
            
            ax2.pie(signal_counts, labels=labels, colors=colors, autopct='%1.1f%%')
            ax2.set_title('信号分布')
            
            # 3. 月度信号分布
            ax3 = axes[1, 0]
            df_copy['month'] = df_copy['date'].dt.month
            monthly_counts = df_copy.groupby(['month', 'signal']).size().unstack(fill_value=0)
            
            if not monthly_counts.empty:
                monthly_counts.plot(kind='bar', ax=ax3, color=['gray', 'red', 'green'])
                ax3.set_title('月度信号分布')
                ax3.set_xlabel('月份')
                ax3.set_ylabel('信号数量')
                ax3.legend(['观望', '做多', '做空'])
            
            # 4. 信号置信度分布
            ax4 = axes[1, 1]
            if 'signal_confidence' in df.columns:
                confidence_data = df[df['signal_confidence'] > 0]['signal_confidence']
                if len(confidence_data) > 0:
                    ax4.hist(confidence_data, bins=20, alpha=0.7, color='blue')
                    ax4.set_title('信号置信度分布')
                    ax4.set_xlabel('置信度')
                    ax4.set_ylabel('频次')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"信号分布图已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib或seaborn未安装，无法绘制信号分布图")
        except Exception as e:
            logger.error(f"绘制信号分布图失败: {e}")


def test_signal_generator():
    """测试信号生成器"""
    print("测试信号生成器...")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='2D')
    
    # 生成价格数据（趋势+随机波动）
    n = len(dates)
    trend = np.linspace(4000, 4500, n)
    noise = np.random.normal(0, 50, n)
    prices = trend + noise
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.normal(0, 20, n),
        'high': prices + np.random.uniform(0, 100, n),
        'low': prices - np.random.uniform(0, 100, n),
        'close': prices,
        'volume': np.random.randint(10000, 100000, n)
    })
    
    # 确保价格逻辑正确
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        test_data.loc[i, 'high'] = max(row['high'], row['open'], row['close'])
        test_data.loc[i, 'low'] = min(row['low'], row['open'], row['close'])
    
    # 计算MA20
    test_data['ma20'] = test_data['close'].rolling(window=20).mean()
    
    # 测试信号生成
    generator = SignalGenerator(ma_period=20)
    
    print("\n1. 测试信号生成:")
    signals_df = generator.generate_signals(test_data)
    print(f"信号数据列: {list(signals_df.columns)}")
    
    # 显示前几条信号
    signal_rows = signals_df[signals_df['signal'] != 0].head(10)
    if not signal_rows.empty:
        print("前10个信号:")
        print(signal_rows[['date', 'close', 'ma20', 'signal', 'signal_reason']].to_string())
    
    print("\n2. 测试信号统计:")
    stats = generator.get_signal_statistics(signals_df)
    print(f"总信号数: {stats['total_signals']}")
    print(f"做多信号: {stats['buy_signals']}")
    print(f"做空信号: {stats['sell_signals']}")
    print(f"信号频率: {stats['signal_frequency']:.2%}")
    
    print("\n3. 测试信号过滤器:")
    filtered_df = generator.add_signal_filters(signals_df)
    filtered_stats = generator.get_signal_statistics(filtered_df)
    print(f"过滤后总信号: {filtered_stats['total_signals']}")
    print(f"过滤后信号频率: {filtered_stats['signal_frequency']:.2%}")
    
    print("\n4. 测试单个信号生成:")
    # 在数据末尾生成信号
    last_signal = generator.generate_signal_at_index(filtered_df, -1)
    if last_signal:
        print(f"最新信号: {last_signal.signal_type.name}")
        print(f"信号价格: {last_signal.price:.2f}")
        print(f"信号原因: {last_signal.reason}")
    else:
        print("当前无信号")
    
    print("\n信号生成器测试完成!")


if __name__ == "__main__":
    test_signal_generator()