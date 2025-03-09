import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 📌 データの読み込み
def load_data():
    train = pd.read_csv("train.csv", parse_dates=["date"])
    test = pd.read_csv("test.csv", parse_dates=["date"])
    stores = pd.read_csv("stores.csv")
    oil = pd.read_csv("oil.csv", parse_dates=["date"])
    holidays = pd.read_csv("holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv("transactions.csv", parse_dates=["date"])
    
    return train, test, stores, oil, holidays, transactions

df_train, df_test, df_stores, df_oil, df_holidays, df_transactions = load_data()

# 📌 祝日データは National（全国）のみを使用
df_holidays = df_holidays[df_holidays["locale"] == "National"]

# 📌 カラム名の整理（`type` の競合を防ぐ）
df_stores.rename(columns={"type": "type_store"}, inplace=True)
df_holidays.rename(columns={"type": "type_holiday"}, inplace=True)

# 📌 データの結合
def merge_data(df):
    df = df.merge(df_stores, on="store_nbr", how="left")
    df = df.merge(df_oil, on="date", how="left")
    df = df.merge(df_transactions, on=["date", "store_nbr"], how="left")
    df = df.merge(df_holidays, on="date", how="left")
    return df

df_train = merge_data(df_train)
df_test = merge_data(df_test)

# 📌 欠損値の補完（原油価格データ）
df_train["dcoilwtico"].interpolate(inplace=True)
df_train["dcoilwtico"].fillna(df_train["dcoilwtico"].mean(), inplace=True)

df_test["dcoilwtico"].interpolate(inplace=True)
df_test["dcoilwtico"].fillna(df_test["dcoilwtico"].mean(), inplace=True)

# 📌 日付関連の特徴量
for df in [df_train, df_test]:
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=月曜, 6=日曜
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# 📌 カテゴリ変数のエンコーディング
le = LabelEncoder()
for df in [df_train, df_test]:
    df["store_encoded"] = le.fit_transform(df["store_nbr"])
    df["family_encoded"] = le.fit_transform(df["family"])

# 📌 One-Hot Encoding（店舗タイプ & クラスター）
df_train = pd.get_dummies(df_train, columns=["type_store", "cluster"], drop_first=False)
df_test = pd.get_dummies(df_test, columns=["type_store", "cluster"], drop_first=False)

# 📌 時系列特徴量の作成（過去の売上データを使う）
for lag in [7, 14, 30]:
    df_train[f"lag_{lag}"] = df_train["sales"].shift(lag)
    df_train[f"rolling_mean_{lag}"] = df_train["sales"].shift(lag).rolling(window=lag).mean()

for lag in [7, 14, 30]:
    df_test[f"lag_{lag}"] = 0  # 予測対象データには過去データがないため仮にゼロで埋める
    df_test[f"rolling_mean_{lag}"] = 0
df_train.fillna(0, inplace=True)  # 欠損値をNANをゼロで埋める
df_test.fillna(0, inplace=True)  

# 📌 祝日フラグの作成
df_train["is_holiday"] = df_train["type_holiday"].apply(lambda x: 1 if x == "Holiday" else 0).fillna(0)
df_test["is_holiday"] = df_test["type_holiday"].apply(lambda x: 1 if x == "Holiday" else 0).fillna(0)

# 📌 外れ値の除去（IQRを使用）
Q1 = df_train["sales"].quantile(0.25)
Q3 = df_train["sales"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_train = df_train[(df_train["sales"] >= lower_bound) & (df_train["sales"] <= upper_bound)]
#print("After outlier removal:", df_train.shape)

# 📌 重複データの削除
df_train.drop_duplicates(inplace=True)
df_test.drop_duplicates(inplace=True)

# 📌 最終確認
# print(df_train.head())
# print(df_train.info())
# print(df_test.head())
# print(df_test.info())

# print(df_train.columns)  # データフレームに存在するカラム一覧を確認

# if df_train.empty:
#     print("⚠ df_train is empty! Check preprocessing steps.")
# else:
#     print(df_train.head())

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_log_error

# 特徴量とターゲットの設定
features = [
    "store_encoded", "family_encoded", "day_of_week", "month", "year", "is_weekend", 
    "dcoilwtico", "transactions", "is_holiday"
] + [col for col in df_train.columns if "type_store_" in col or "cluster_" in col] + [
    "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"
]

target = "sales"

# 欠損値の処理（XGBoostは欠損値を自動処理するが、補完しておく）
df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)

# 学習データと検証データの分割
X_train, X_valid, y_train, y_valid = train_test_split(
    df_train[features], df_train[target], test_size=0.2, random_state=42
)

# XGBoost データセットの作成
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(df_test[features])

# XGBoost のパラメータ設定
params = {
    "objective": "reg:squarederror",  # 回帰問題
    "eval_metric": "mae",  # 評価指標（MAE: Mean Absolute Error）
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

# モデルの学習
watchlist = [(dtrain, "train"), (dvalid, "valid")]
model = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist, early_stopping_rounds=50, verbose_eval=50)

# 予測
predictions = model.predict(dtest)

# 結果の保存
df_test["sales_predictions"] = predictions
df_test[["id", "sales_predictions"]].to_csv("submission.csv", index=False)

# 検証データでの評価
y_valid_pred = model.predict(dvalid)
y_valid_pred = np.maximum(0, y_valid_pred)# 負の値を0にクリッピング
rmsle = np.sqrt(mean_squared_log_error(y_valid, y_valid_pred))
print(f"Validation RMSLE: {rmsle:.4f}")