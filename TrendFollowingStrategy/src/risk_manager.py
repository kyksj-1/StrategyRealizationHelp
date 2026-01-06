"""
MA20趋势跟踪策略 - 风险管理模块
实现动态止损计算、仓位管理等功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 设置日志
logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """持仓方向"""
    LONG = 1     # 做多
    SHORT = -1   # 做空
    NONE = 0     # 无持仓


@dataclass
class RiskParameters:
    """风险参数"""
    max_loss_pct: float = 0.06      # 最大止损容忍度6%
    force_stop_pct: float = 0.03    # 强制止损3%
    risk_per_trade: float = 0.02    # 每笔交易风险2%
    max_position_pct: float = 0.8   # 最大仓位80%
    min_position_size: int = 1      # 最小仓位1手


@dataclass
class StopLossResult:
    """止损计算结果"""
    stop_price: float
    stop_distance_pct: float
    risk_amount: float
    is_forced_stop: bool
    calculation_reason: str


@dataclass
class PositionSizeResult:
    """仓位大小计算结果"""
    position_size: int
    required_margin: float
    risk_amount: float
    risk_pct_of_capital: float
    calculation_reason: str


class RiskManager:
    """风险管理员 - 负责止损计算和仓位管理"""
    
    def __init__(self, parameters: Optional[RiskParameters] = None):
        """初始化风险管理器
        
        Args:
            parameters: 风险参数，如果为None使用默认参数
        """
        self.parameters = parameters or RiskParameters()
    
    def calculate_stop_loss(self, entry_price: float, prev_extreme: float, 
                          direction: PositionSide, **kwargs) -> StopLossResult:
        """计算止损价格
        
        止损规则：
        做多时：
        - 基础止损 = 前一根K线的最低价
        - 止损距离 = (进场价 - 基础止损) / 进场价
        - 如果止损距离 > 6%: 实际止损 = 进场价 × (1 - 3%)
        - 否则: 实际止损 = 基础止损
        
        做空时：同理，使用前一根K线的最高价
        
        Args:
            entry_price: 进场价格
            prev_extreme: 前一根K线的极值（做多用最低价，做空用最高价）
            direction: 持仓方向
            **kwargs: 其他参数
            
        Returns:
            止损计算结果
        """
        logger.info(f"计算{'做多' if direction == PositionSide.LONG else '做空'}止损价格...")
        
        if direction == PositionSide.LONG:
            # 做多止损计算
            base_stop = prev_extreme
            stop_distance_pct = (entry_price - base_stop) / entry_price
            
            if stop_distance_pct > self.parameters.max_loss_pct:
                # 超过6%容忍度，使用强制3%止损
                stop_price = entry_price * (1 - self.parameters.force_stop_pct)
                is_forced_stop = True
                reason = f"基础止损距离{stop_distance_pct:.2%}超过{self.parameters.max_loss_pct:.2%}，使用强制{self.parameters.force_stop_pct:.2%}止损"
            else:
                # 使用基础止损
                stop_price = base_stop
                is_forced_stop = False
                reason = f"基础止损距离{stop_distance_pct:.2%}在容忍范围内"
            
            # 确保止损价低于进场价
            if stop_price >= entry_price:
                stop_price = entry_price * (1 - self.parameters.force_stop_pct)
                reason = f"基础止损价{base_stop:.2f} >= 进场价{entry_price:.2f}，使用强制止损"
        
        elif direction == PositionSide.SHORT:
            # 做空止损计算
            base_stop = prev_extreme
            stop_distance_pct = (base_stop - entry_price) / entry_price
            
            if stop_distance_pct > self.parameters.max_loss_pct:
                # 超过6%容忍度，使用强制3%止损
                stop_price = entry_price * (1 + self.parameters.force_stop_pct)
                is_forced_stop = True
                reason = f"基础止损距离{stop_distance_pct:.2%}超过{self.parameters.max_loss_pct:.2%}，使用强制{self.parameters.force_stop_pct:.2%}止损"
            else:
                # 使用基础止损
                stop_price = base_stop
                is_forced_stop = False
                reason = f"基础止损距离{stop_distance_pct:.2%}在容忍范围内"
            
            # 确保止损价高于进场价
            if stop_price <= entry_price:
                stop_price = entry_price * (1 + self.parameters.force_stop_pct)
                reason = f"基础止损价{base_stop:.2f} <= 进场价{entry_price:.2f}，使用强制止损"
        
        else:
            raise ValueError(f"无效的持仓方向: {direction}")
        
        # 计算风险金额（每手）
        risk_amount = abs(entry_price - stop_price)
        
        result = StopLossResult(
            stop_price=stop_price,
            stop_distance_pct=abs(stop_distance_pct),
            risk_amount=risk_amount,
            is_forced_stop=is_forced_stop,
            calculation_reason=reason
        )
        
        logger.info(f"止损计算完成: {reason}")
        logger.info(f"止损价: {stop_price:.2f}, 止损距离: {abs(stop_distance_pct):.2%}")
        
        return result
    
    def calculate_position_size(self, capital: float, entry_price: float, 
                              stop_price: float, margin_rate: float = 0.10,
                              contract_multiplier: float = 10.0) -> PositionSizeResult:
        """计算仓位大小
        
        基于风险管理的仓位计算：
        1. 计算每手风险金额
        2. 根据账户资金的2%确定最大可开手数
        3. 检查保证金是否足够（不超过80%资金）
        
        Args:
            capital: 可用资金
            entry_price: 进场价格
            stop_price: 止损价格
            margin_rate: 保证金比例
            contract_multiplier: 合约乘数
            
        Returns:
            仓位大小计算结果
        """
        logger.info(f"计算仓位大小，资金: {capital:.2f}, 进场价: {entry_price:.2f}")
        
        # 每手风险金额
        risk_per_contract = abs(entry_price - stop_price) * contract_multiplier
        
        # 基于风险的最大可开手数（账户的2%）
        max_risk_amount = capital * self.parameters.risk_per_trade
        max_by_risk = int(max_risk_amount / risk_per_contract)
        
        # 基于保证金的最大可开手数（不超过80%资金）
        margin_per_contract = entry_price * contract_multiplier * margin_rate
        max_by_margin = int((capital * self.parameters.max_position_pct) / margin_per_contract)
        
        # 取两者较小值，但不小于最小仓位
        position_size = max(self.parameters.min_position_size, 
                           min(max_by_risk, max_by_margin))
        
        # 计算实际风险
        actual_risk_amount = risk_per_contract * position_size
        actual_risk_pct = actual_risk_amount / capital
        required_margin = margin_per_contract * position_size
        
        reason = f"风险限制: {max_by_risk}手, 保证金限制: {max_by_margin}手, "
        if max_by_risk <= max_by_margin:
            reason += "风险因素主导"
        else:
            reason += "保证金因素主导"
        
        result = PositionSizeResult(
            position_size=position_size,
            required_margin=required_margin,
            risk_amount=actual_risk_amount,
            risk_pct_of_capital=actual_risk_pct,
            calculation_reason=reason
        )
        
        logger.info(f"仓位计算完成: {position_size}手, 风险金额: {actual_risk_amount:.2f}, "
                   f"风险比例: {actual_risk_pct:.2%}")
        
        return result
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float,
                              extreme_price: float, direction: PositionSide,
                              trailing_pct: float = 0.02) -> float:
        """计算移动止损价格
        
        Args:
            entry_price: 进场价格
            current_price: 当前价格
            extreme_price: 极值价格（做多用最高价，做空用最低价）
            direction: 持仓方向
            trailing_pct: 移动止损百分比
            
        Returns:
            移动止损价格
        """
        if direction == PositionSide.LONG:
            # 做多移动止损：从最高价回撤2%
            trailing_stop = extreme_price * (1 - trailing_pct)
            # 确保不亏损（至少保本）
            trailing_stop = max(trailing_stop, entry_price)
        elif direction == PositionSide.SHORT:
            # 做空移动止损：从最低价反弹2%
            trailing_stop = extreme_price * (1 + trailing_pct)
            # 确保不亏损（至少保本）
            trailing_stop = min(trailing_stop, entry_price)
        else:
            raise ValueError(f"无效的持仓方向: {direction}")
        
        return trailing_stop
    
    def validate_risk_parameters(self) -> Dict[str, Any]:
        """验证风险参数的有效性
        
        Returns:
            验证结果
        """
        issues = []
        
        # 检查止损比例
        if self.parameters.max_loss_pct <= 0:
            issues.append("最大止损容忍度必须大于0")
        
        if self.parameters.force_stop_pct <= 0:
            issues.append("强制止损比例必须大于0")
        
        if self.parameters.force_stop_pct >= self.parameters.max_loss_pct:
            issues.append("强制止损比例应小于最大止损容忍度")
        
        # 检查风险比例
        if self.parameters.risk_per_trade <= 0 or self.parameters.risk_per_trade >= 0.1:
            issues.append("每笔交易风险比例应在0-10%之间")
        
        if self.parameters.max_position_pct <= 0 or self.parameters.max_position_pct > 1:
            issues.append("最大仓位比例应在0-100%之间")
        
        # 检查仓位大小
        if self.parameters.min_position_size < 1:
            issues.append("最小仓位不能小于1手")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'parameters': self.parameters
        }
    
    def get_risk_summary(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """获取风险统计摘要
        
        Args:
            trades_df: 交易记录DataFrame
            
        Returns:
            风险统计信息
        """
        if trades_df.empty:
            return {}
        
        # 基础统计
        total_trades = len(trades_df)
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] < 0).sum()
        
        # 盈亏统计
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # 风险指标
        max_loss = trades_df['pnl'].min()
        max_win = trades_df['pnl'].max()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 盈亏比
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 回撤相关
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max_win,
            'max_loss': max_loss,
            'max_drawdown': max_drawdown
        }
        
        return summary


def test_risk_manager():
    """测试风险管理器"""
    print("测试风险管理器...")
    
    # 创建风险管理器
    risk_manager = RiskManager()
    
    print("\n1. 测试做多止损计算:")
    entry_price = 4000.0
    prev_low = 3800.0
    
    result = risk_manager.calculate_stop_loss(entry_price, prev_low, PositionSide.LONG)
    print(f"进场价: {entry_price}, 前低: {prev_low}")
    print(f"止损价: {result.stop_price:.2f}")
    print(f"止损距离: {result.stop_distance_pct:.2%}")
    print(f"风险金额: {result.risk_amount:.2f}")
    print(f"是否强制止损: {result.is_forced_stop}")
    print(f"原因: {result.calculation_reason}")
    
    print("\n2. 测试做空止损计算:")
    entry_price = 4000.0
    prev_high = 4200.0
    
    result = risk_manager.calculate_stop_loss(entry_price, prev_high, PositionSide.SHORT)
    print(f"进场价: {entry_price}, 前高: {prev_high}")
    print(f"止损价: {result.stop_price:.2f}")
    print(f"止损距离: {result.stop_distance_pct:.2%}")
    print(f"原因: {result.calculation_reason}")
    
    print("\n3. 测试仓位大小计算:")
    capital = 100000.0
    entry_price = 4000.0
    stop_price = 3800.0
    margin_rate = 0.10
    contract_multiplier = 10.0
    
    position_result = risk_manager.calculate_position_size(
        capital, entry_price, stop_price, margin_rate, contract_multiplier
    )
    print(f"资金: {capital}, 进场价: {entry_price}, 止损价: {stop_price}")
    print(f"建议仓位: {position_result.position_size}手")
    print(f"所需保证金: {position_result.required_margin:.2f}")
    print(f"风险金额: {position_result.risk_amount:.2f}")
    print(f"风险比例: {position_result.risk_pct_of_capital:.2%}")
    print(f"原因: {position_result.calculation_reason}")
    
    print("\n4. 测试移动止损:")
    entry_price = 4000.0
    current_price = 4200.0
    extreme_price = 4300.0
    
    trailing_stop = risk_manager.calculate_trailing_stop(
        entry_price, current_price, extreme_price, PositionSide.LONG
    )
    print(f"进场价: {entry_price}, 当前价: {current_price}, 极值: {extreme_price}")
    print(f"移动止损价: {trailing_stop:.2f}")
    
    print("\n5. 测试风险参数验证:")
    validation = risk_manager.validate_risk_parameters()
    print(f"参数是否有效: {validation['is_valid']}")
    if not validation['is_valid']:
        print(f"问题: {validation['issues']}")
    
    print("\n风险管理器测试完成!")


if __name__ == "__main__":
    test_risk_manager()