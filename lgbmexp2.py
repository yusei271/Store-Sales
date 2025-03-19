import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

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

# 📌 カラム名の整理
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

# 📌 欠損値の補完
df_train["dcoilwtico"].fillna(df_train["dcoilwtico"].mean(), inplace=True)
df_test["dcoilwtico"].fillna(df_test["dcoilwtico"].mean(), inplace=True)
df_train["transactions"].fillna(0, inplace=True)
df_test["transactions"].fillna(0, inplace=True)

# 📌 日付関連の特徴量
for df in [df_train, df_test]:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# 📌 カテゴリ変数のエンコーディング
categorical_columns = ["store_nbr", "family", "city", "state", "locale", "locale_name", "description", "transferred"]
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col].astype(str))
    df_test[col] = le.transform(df_test[col].astype(str))
    label_encoders[col] = le

# 📌 One-Hot Encoding（店舗タイプ & クラスター）
df_train = pd.get_dummies(df_train, columns=["type_store", "cluster"], drop_first=True)
df_test = pd.get_dummies(df_test, columns=["type_store", "cluster"], drop_first=True)

# 📌 時系列特徴量の作成
for lag in [7, 14, 30]:
    df_train[f"lag_{lag}"] = df_train["sales"].shift(lag)

# NaN を埋める
df_train.fillna(0, inplace=True)
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

# 📌 重複データの削除
df_train.drop_duplicates(inplace=True)
df_test.drop_duplicates(inplace=True)

# 📌 説明変数と目的変数を分ける
TARGET = "sales"
features = df_train.drop(columns=[TARGET, "date", "type_holiday"]).columns

X = df_train[features]
y = np.log1p(df_train[TARGET])  # 💡 log(1+x) を適用

# 📌 訓練データと検証データに分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 LightGBM のデータセット形式に変換
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

# ✅ **カスタムメトリック（RMSLE）を定義**
def rmsle_eval(y_pred, dataset):
    y_true = dataset.get_label()
    return 'rmsle', np.sqrt(mean_squared_log_error(y_true, np.maximum(y_pred, 0))), False

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

# 📌 LightGBM のモデルを訓練
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],  # ✅ 学習データと検証データを指定
    valid_names=["train", "valid"],  # ✅ ログ表示のための名前
    feval=rmsle_eval,  # ✅ ここでカスタムメトリックを追加！
    callbacks=[
        lgb.early_stopping(50),  # ✅ `early_stopping_rounds` の代わりに `callbacks` を使用
        lgb.log_evaluation(100)  # ✅ 100回ごとにログを表示
    ]
)

# 📌 予測＆評価（RMSLE を最終確認）
# ✅ モデルが予測した値を `expm1` で元のスケールに戻す
y_pred = np.expm1(model.predict(X_valid, num_iteration=model.best_iteration))

# ✅ 検証データも `expm1` で元のスケールに戻す
y_valid_orig = np.expm1(y_valid)

# ✅ 負の値を0に補正
y_pred = np.maximum(y_pred, 0)
y_valid_orig = np.maximum(y_valid_orig, 0)

# ✅ RMSLE の再計算（正しいスケールで）
rmsle = np.sqrt(mean_squared_log_error(y_valid_orig, y_pred))

print(f"✅ LightGBM RMSLE (修正後): {rmsle:.4f}")