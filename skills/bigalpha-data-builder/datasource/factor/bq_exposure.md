## 表描述

A 股 Barra 风格因子暴露表，扩展自 `jq_style_factor`。每行代表某只股票在某个交易日上的因子取值，包含 10 个 Barra 风格因子的暴露值、一级行业代码、市值与权重、当期收益率，以及 31 个行业哑变量（覆盖 sw2014 与 sw2021 两套申万行业分类）。

可直接用于：
- 因子收益率回归（风格因子 + 行业哑变量构成自变量）
- 风格暴露归因（持仓在各风格上的加权暴露）
- 行业中性化处理（按行业分组去均值/回归取残差）

## 字段

### 索引字段

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| date | np.datetime64 | 日期 | np.nan |
| instrument | pd.StringDtype | 证券代码 | np.nan |

### 风格因子（Barra 10 因子，截面标准化后的暴露值）

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| SIZE | np.double | 市值 | np.nan |
| BETA | np.double | 贝塔 | np.nan |
| MOMENTUM | np.double | 传统动量 | np.nan |
| RESVOL | np.double | 残差波动率 | np.nan |
| SIZENL | np.double | 非线性市值 | np.nan |
| BTOP | np.double | 账面市值比 | np.nan |
| LIQUIDTY | np.double | 流动性 | np.nan |
| EARNYILD | np.double | 盈利能力 | np.nan |
| GROWTH | np.double | 成长 | np.nan |
| LEVERAGE | np.double | 杠杆 | np.nan |

### 行业与权重

| 字段 | 类型 | 描述 | 默认值 |
|---|---|---|---|
| industry_level1_code | pd.StringDtype | 一级行业代码 | np.nan |
| float_market_cap | np.double | 流通市值 | np.nan |
| weights | np.double | 市值权重（截面归一化） | np.nan |
| ret | np.double | 当期收益率 | np.nan |

### 行业哑变量（31 个，0=不属于，1=属于）

> 同一时点每只股票仅在一个行业哑变量上取 1，其余为 0。`sw2014` 与 `sw2021` 在多数行业上一致，差异列在表注里。

| 字段 | 类型 | 描述 | 行业分类来源 |
|---|---|---|---|
| AGRIFOREST | np.int8 | 农林牧渔 | sw2014, sw2021 |
| MINING | np.int8 | 采掘 | sw2014 |
| CHEM | np.int8 | 化工 / 基础化工 | sw2014（化工）, sw2021（基础化工） |
| IRONSTEEL | np.int8 | 钢铁 | sw2014, sw2021 |
| NONFERMETAL | np.int8 | 有色金属 | sw2014, sw2021 |
| ELECTRONICS | np.int8 | 电子 | sw2014, sw2021 |
| AUTO | np.int8 | 汽车 | sw2014, sw2021 |
| HOUSEAPP | np.int8 | 家用电器 | sw2014, sw2021 |
| FOODBEVER | np.int8 | 食品饮料 | sw2014, sw2021 |
| TEXTILE | np.int8 | 纺织服饰 | sw2014, sw2021 |
| LIGHTINDUS | np.int8 | 轻工制造 | sw2014, sw2021 |
| HEALTH | np.int8 | 医药生物 | sw2014, sw2021 |
| UTILITIES | np.int8 | 公用事业 | sw2014, sw2021 |
| TRANSPORTATION | np.int8 | 交通运输 | sw2014, sw2021 |
| REALESTATE | np.int8 | 房地产 | sw2014, sw2021 |
| COMMETRADE | np.int8 | 商业贸易 / 商贸零售 | sw2014（商业贸易）, sw2021（商贸零售） |
| LEISERVICE | np.int8 | 休闲服务 / 社会服务 | sw2014（休闲服务）, sw2021（社会服务） |
| BANK | np.int8 | 银行 | sw2014, sw2021 |
| NONBANKFINAN | np.int8 | 非银金融 | sw2014, sw2021 |
| CONGLOMERATES | np.int8 | 综合 | sw2014, sw2021 |
| CONMAT | np.int8 | 建筑材料 | sw2014, sw2021 |
| BUILDDECO | np.int8 | 建筑装饰 | sw2014, sw2021 |
| ELECEQP | np.int8 | 电气设备 / 电力设备 | sw2014（电气设备）, sw2021（电力设备） |
| MACHIEQUIP | np.int8 | 机械设备 | sw2014, sw2021 |
| AERODEF | np.int8 | 国防军工 | sw2014, sw2021 |
| COMPUTER | np.int8 | 计算机 | sw2014, sw2021 |
| MEDIA | np.int8 | 传媒 | sw2014, sw2021 |
| TELECOM | np.int8 | 通信 | sw2014, sw2021 |
| COAL | np.int8 | 煤炭 | sw2021 |
| PETRO | np.int8 | 石油石化 | sw2021 |
| ENVP | np.int8 | 环保 | sw2021 |
| BEAUTY | np.int8 | 美容护理 | sw2021 |

> sw2014 与 sw2021 的主要差异：
> - sw2014 的 `MINING`（采掘）在 sw2021 中拆分为 `COAL`（煤炭）与 `PETRO`（石油石化）
> - sw2021 新增 `ENVP`（环保）、`BEAUTY`（美容护理）
> - 部分行业仅改名（化工→基础化工、商业贸易→商贸零售、休闲服务→社会服务、电气设备→电力设备）

## 示例

### 查询某日全市场风格因子暴露

```python
import dai
dai.query("""
SELECT date, instrument, SIZE, BETA, MOMENTUM, RESVOL, SIZENL,
       BTOP, LIQUIDTY, EARNYILD, GROWTH, LEVERAGE
FROM bq_exposure
WHERE date = '2026-05-20'
ORDER BY instrument
""").df()
```

### 因子收益率回归（风格因子 + 行业哑变量为自变量，ret 为因变量）

```python
import dai
df = dai.query("""
SELECT *
FROM bq_exposure
WHERE date = '2026-05-20'
""").df()

import statsmodels.api as sm
style_cols = ['SIZE', 'BETA', 'MOMENTUM', 'RESVOL', 'SIZENL',
              'BTOP', 'LIQUIDTY', 'EARNYILD', 'GROWTH', 'LEVERAGE']
industry_cols = ['AGRIFOREST', 'MINING', 'CHEM', 'IRONSTEEL', 'NONFERMETAL',
                 'ELECTRONICS', 'AUTO', 'HOUSEAPP', 'FOODBEVER', 'TEXTILE',
                 'LIGHTINDUS', 'HEALTH', 'UTILITIES', 'TRANSPORTATION', 'REALESTATE',
                 'COMMETRADE', 'LEISERVICE', 'BANK', 'NONBANKFINAN', 'CONGLOMERATES',
                 'CONMAT', 'BUILDDECO', 'ELECEQP', 'MACHIEQUIP', 'AERODEF',
                 'COMPUTER', 'MEDIA', 'TELECOM', 'COAL', 'PETRO', 'ENVP', 'BEAUTY']

X = df[style_cols + industry_cols]
y = df['ret']
w = df['weights']
model = sm.WLS(y, X, weights=w).fit()
factor_returns = model.params
```

### 计算组合在风格因子上的加权暴露

```python
import dai
dai.query("""
SELECT
    date,
    SUM(weights * SIZE)     AS portfolio_SIZE,
    SUM(weights * BETA)     AS portfolio_BETA,
    SUM(weights * MOMENTUM) AS portfolio_MOMENTUM,
    SUM(weights * BTOP)     AS portfolio_BTOP
FROM bq_exposure
WHERE date >= '2026-05-01' AND date <= '2026-05-20'
  AND instrument IN ('000001.SZ', '600519.SH', '300750.SZ')
GROUP BY date
ORDER BY date
""").df()
```

### 行业中性化：按行业分组对某个 alpha 因子去均值

```python
import dai
dai.query("""
SELECT
    date,
    instrument,
    industry_level1_code,
    SIZE - AVG(SIZE) OVER (PARTITION BY date, industry_level1_code) AS SIZE_neutral
FROM bq_exposure
WHERE date = '2026-05-20'
""").df()
```
