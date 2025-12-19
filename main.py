#----------------------------------------------------------------

# ライブラリのインポート
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns

#----------------------------------------------------------------

# データの読み込み
df = pd.read_excel("data/kadai.xlsx")
# df.info()

df1 = df.copy()

def LightGBM(X_train, Y_train, X_test, Y_test):
    """
    # 学習データとテストデータに分割
    train = df1[0:1776]
    test = df1[1776:2276]

    # インデックスのリセット
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # テストデータの最初の1レコードの"process_end_time"より前のtrainデータを抽出
    train = train[train["final_mes_time"] < test['process_end_time'][0]]

    # -----------------------------------------------------------------

    # LightGBM
    # 学習データ
    X_train = train.drop(['process_end_time', 'final_mes_time', 'OV'], axis=1)
    Y_train =train.OV

    # テストデータ
    X_test = test.drop(['process_end_time', 'final_mes_time', 'OV'], axis=1)
    Y_test = test.OV
    """

    # LightGBM用データセットの作成
    train_data = lgb.Dataset(X_train, label=Y_train)
    eval_data = lgb.Dataset(X_test, label=Y_test, reference=train_data)

    # ハイパーパラメータの設定
    params = {
        'task': 'train',            
        'boosting_type': 'gbdt',    # 勾配ブースティング
        'objective': 'regression',  # 目的：回帰
        'metric': 'rmse',           # 評価指標：RMSE
        'learning_rate': 0.05,      # 学習率
        'num_leaves': 10,           # 木の複雑さ
        'min_data_in_leaf': 20,     # 葉っぱに含まれる最小データ数を増やす
        'verbose': -1               # 不要なログを非表示
    }

    # モデルの学習
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, eval_data],             # 訓練・検証データをセット
        num_boost_round=10000,                          # 最大イテレーション回数
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),    # 100回スコアが改善しなかったら停止
            lgb.log_evaluation(0)                       # ログを非表示
        ]
    )

    # 予測と精度評価
    y_pred = model.predict(X_test)

    # RMSEを計算
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    # print(f"\nTest RMSE: {float(rmse):.4f}")

    # 特徴量重要度の可視化
    #lgb.plot_importance(model, figsize=(10, 6), importance_type='gain')
    #plt.show()

    return y_pred

    # -----------------------------------------------------------------

# 特徴量エンジニアリング
# Lag特徴量
lags = [1, 2, 3, 5]
for lag in lags:
      df1[f"OV_lag{lag}"] = df1['OV'].shift(lag)

# 差分特徴量
df1['OV_diff'] = df1['OV'].diff(1)

# 移動平均と移動標準偏差
windows = [3, 5]
for window in windows:
      df1[f'OV_roll_mean{window}'] = df1['OV'].rolling(window).mean().shift(1)
      df1[f'OV_roll_std{window}'] = df1['OV'].rolling(window).std().shift(1)

# shiftへの対応
df1 = df1.dropna().reset_index(drop=True)

# 変数選択
# テストデータ数設定
test_len = 500
total_rows = len(df1)
start_idx = total_rows - test_len
end_idx = total_rows

train_for_sel = df1.iloc[0:start_idx]

X_sel = train_for_sel.drop(['process_end_time', 'final_mes_time', 'OV'], axis=1)
y_sel = train_for_sel['OV']

# 変数選択用の仮モデル学習
print("変数選択を実行中...")
ds_sel = lgb.Dataset(X_sel, label=y_sel)
params_sel = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'seed': 42
}
model_sel = lgb.train(params_sel, ds_sel, num_boost_round=100)

# 重要度を取得 (gain: RMSEへの貢献度)
importance_df = pd.DataFrame()
importance_df['feature'] = X_sel.columns
importance_df['importance'] = model_sel.feature_importance(importance_type='gain')

# 重要度順にソート
importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

# 累積寄与率を計算
importance_df['cumulative_gain'] = importance_df['importance'].cumsum()
importance_df['cumulative_ratio'] = importance_df['cumulative_gain'] / importance_df['importance'].sum()

# 閾値の設定 (0.95 = 寄与率95%までの変数を残す)
threshold = 0.95

# 閾値を超える最初のインデックスを見つける
valid_indices = importance_df[importance_df['cumulative_ratio'] <= threshold].index
if len(valid_indices) < len(importance_df):
    cutoff_idx = valid_indices[-1] + 1
else:
    cutoff_idx = len(importance_df)

# 選ばれた変数リスト
selected_features = importance_df.iloc[:cutoff_idx + 1]['feature'].tolist()

print(f"\n全変数: {len(X_sel.columns)} -> 選択された変数: {len(selected_features)}")
print(f"削除された変数: {list(set(X_sel.columns) - set(selected_features))}")

# --- 累積寄与率の可視化 ---
plt.figure(figsize=(10, 6))
x_vals = range(len(importance_df))
plt.bar(x_vals, importance_df['importance'], align='center', label='Individual Importance')
plt.plot(x_vals, importance_df['cumulative_gain'], color='r', marker='o', label='Cumulative Importance')
# 第2軸で比率を表示
plt.twinx()
plt.plot(x_vals, importance_df['cumulative_ratio'], color='orange', linestyle='--', label='Cumulative Ratio')
plt.axhline(y=threshold, color='g', linestyle=':', label=f'Threshold {threshold}')
plt.title('Feature Selection by Cumulative Importance')
plt.show()

yHat = []
total_rows = len(df1)
test_len = 500

start_idx = total_rows - test_len
end_idx = total_rows

for  i in np.arange(start_idx, end_idx):
        train = df1[0:i]
        test = df1[i:i+1]

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        train = train[train["final_mes_time"] < test['process_end_time'].values[0]]

        X_train = train.drop(['process_end_time', 'final_mes_time', 'OV'], axis=1)
        Y_train = train.OV

        X_test = test.drop(['process_end_time', 'final_mes_time', 'OV'], axis=1)
        Y_test = test.OV

        yHat.append(LightGBM(X_train, Y_train, X_test, Y_test))



Y_t = df1["OV"].iloc[start_idx:end_idx].values
yh = np.array(yHat).flatten()

min_len = min(len(yh), len(Y_t))
yh = yh[:min_len]
Y_t = Y_t[:min_len]

#RMSEの表示
print("RMSE:")
print(np.sqrt(mean_squared_error(Y_t, yh)))

plt.figure(figsize=(12, 6))
plt.plot(Y_t, label='Actual')
plt.plot(yh, label='Predicted')
plt.legend()
plt.title('Prediction with Advanced Features')
plt.show()

# 変数の削除
# 重要度の低い変数の削除
yHat2 = []
for  i in np.arange(start_idx, end_idx):
        train = df1[0:i]
        test = df1[i:i+1]

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        train = train[train["final_mes_time"] < test['process_end_time'].values[0]]

        X_train = train.drop(['process_end_time', 'final_mes_time', 'OV'], axis=1)
        X_train = X_train[selected_features]
        Y_train = train.OV

        X_test = test.drop(['process_end_time', 'final_mes_time', 'OV'], axis=1)
        X_test = X_test[selected_features]
        Y_test = test.OV

        yHat2.append(LightGBM(X_train, Y_train, X_test, Y_test))

yh2 = np.array(yHat2).flatten()

min_len = min(len(yh2), len(Y_t))
yh2 = yh2[:min_len]
Y_t = Y_t[:min_len]

#RMSEの表示
print("RMSE:")
print(np.sqrt(mean_squared_error(Y_t, yh2)))

plt.figure(figsize=(12, 6))
plt.plot(Y_t, label='Actual')
plt.plot(yh2, label='Predicted')
plt.legend()
plt.title('Prediction with Advanced Features')
plt.show()