import calendar
from typing import Literal, List, Tuple

def generate_dates_series(
    start_year: int,
    end_year: int,
    frequency: Literal["monthly", "quarterly", "semi_annual"] = "monthly"
) -> List[Tuple[str, str]]:
    """生成指定时间范围内的日期序列
    
    Args:
        start_year: 开始年份（包含）
        end_year: 结束年份（包含）
        frequency: 时间频率，可选值: "monthly", "quarterly", "semi_annual"
        
    Returns:
        包含开始日期和结束日期的元组列表
    """
    result = []
    
    # 定义不同频率的月份映射
    frequency_configs = {
        "monthly": {
            "periods": 12,  # 每年12个月
            "months_mapping": {i: (i, i) for i in range(1, 13)}  # 每月都是自身开始和结束
        },
        "quarterly": {
            "periods": 4,   # 每年4个季度
            "months_mapping": {
                1: (1, 3),   # 第一季度: 1月-3月
                2: (4, 6),   # 第二季度: 4月-6月
                3: (7, 9),   # 第三季度: 7月-9月
                4: (10, 12)  # 第四季度: 10月-12月
            }
        },
        "semi_annual": {
            "periods": 2,   # 每年2个半年
            "months_mapping": {
                1: (1, 6),   # 上半年: 1月-6月
                2: (7, 12)   # 下半年: 7月-12月
            }
        },
        "yearly": {
            "periods": 1,   # 每年1个年度
            "months_mapping": {
                1: (1, 12)   # 全年: 1月-12月
            }
        }
    }
    
    # 获取配置
    config = frequency_configs[frequency]
    periods = config["periods"]
    months_mapping = config["months_mapping"]
    
    # 生成日期序列
    for year in range(start_year, end_year + 1):
        for period in range(1, periods + 1):
            # 获取起始月份和结束月份
            start_month, end_month = months_mapping[period]
            
            # 生成开始日期（总是该期的第一天）
            sd = f'{year}-{start_month:02d}-01'
            
            # 生成结束日期（该期的最后一天）
            last_day = calendar.monthrange(year, end_month)[1]
            ed = f'{year}-{end_month:02d}-{last_day:02d}'
            
            result.append((sd, ed))
    
    return result
