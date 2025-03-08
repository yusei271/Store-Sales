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

# 📌 最終確認
print(df_train.head())
print(df_train.info())
print(df_test.head())
print(df_test.info())