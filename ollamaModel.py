import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import akshare as ak

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 获取A股中信国安历史数据
df = ak.stock_zh_a_hist(symbol="000839", adjust="qfq")
df["日期"] = pd.to_datetime(df["日期"])

# 2. 构造目标变量（预测下一天收盘价）
df['Target'] = df['收盘'].shift(-1)

# 3. 技术指标与滞后特征
df['MA5'] = df['收盘'].rolling(window=5).mean()
df['MA10'] = df['收盘'].rolling(window=10).mean()
df['收盘-1'] = df['收盘'].shift(1)
df['收盘-2'] = df['收盘'].shift(2)
df['涨跌幅-1'] = df['涨跌幅'].shift(1)
df['成交量-1'] = df['成交量'].shift(1)
df = df.dropna().reset_index(drop=True)

features = [
    '开盘', '最高', '最低', '收盘', '成交量', '振幅', '涨跌幅', '涨跌额',
    '换手率', 'MA5', 'MA10', '收盘-1', '收盘-2', '涨跌幅-1', '成交量-1'
]
X = df[features]
y = np.log1p(df['Target'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 时间序列，不打乱，最后20%做测试
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

model = xgb.XGBRegressor(random_state=42)
param_grid = {
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}
cv = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=5),
    n_jobs=-1,
    verbose=2
)
cv.fit(X_train, y_train)

# 8. 预测与评估
y_pred = cv.predict(X_test)
print('最佳参数:', cv.best_params_)
print('R2:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

# 9. 可视化预测效果
plt.figure(figsize=(12, 6))
plt.plot(df["日期"].iloc[-len(y_test):], np.expm1(y_test.values), label='真实价格')
plt.plot(df["日期"].iloc[-len(y_test):], np.expm1(y_pred), label='预测价格')
plt.xlabel("日期")
plt.ylabel("收盘价")
plt.title("XGBoost预测 vs 实际")
plt.legend()
plt.show()

# 10. 特征重要性可视化
xgb.plot_importance(cv.best_estimator_, max_num_features=10, importance_type='gain')
plt.title("特征重要性Top10")
plt.show()

# 11. 预测下一个交易日收盘价
from pandas.tseries.offsets import BDay
last_date = df["日期"].iloc[-1]
next_day = last_date + BDay(1)  # 自动跳过周末/节假日
print("最后一条数据的日期：", last_date)
latest_features = df[features].iloc[-1].values.reshape(1, -1)
latest_features_scaled = scaler.transform(latest_features)
next_day_pred = cv.predict(latest_features_scaled)

predicted_price = np.expm1(next_day_pred[0])
print(f"预测下一个交易日 {next_day.date()} 收盘价为：{predicted_price:.2f}")
logging.info(f"预测下一个交易日 {next_day.date()} 收盘价为：{predicted_price:.2f}")