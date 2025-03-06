import pandas as pd
import numpy as np

# データ読み込み
train = pd.read_csv("train.csv", parse_dates=["date"])
stores = pd.read_csv("stores.csv")
oil = pd.read_csv("oil.csv", parse_dates=["date"])
holidays = pd.read_csv("holidays_events.csv", parse_dates=["date"])
transactions = pd.read_csv("transactions.csv", parse_dates=["date"])

# 祝日データは National（全国） のみを使用
holidays = holidays[holidays["locale"] == "National"]

# ① 店舗データの `type` → `type_store` にリネーム
stores.rename(columns={"type": "type_store"}, inplace=True)

# ② 祝日データの `type` → `type_holiday` にリネーム
holidays.rename(columns={"type": "type_holiday"}, inplace=True)

# データ結合
df_train = train.merge(stores, on="store_nbr", how="left")
df_train = df_train.merge(oil, on="date", how="left")
df_train = df_train.merge(transactions, on=["date", "store_nbr"], how="left")
df_train = df_train.merge(holidays, on="date", how="left")  # 全国祝日データのみ結合

# 曜日・月・年
# 日付関連の特徴量
df_train["day_of_week"] = df_train["date"].dt.dayofweek  # 0=月曜, 6=日曜
df_train["month"] = df_train["date"].dt.month
df_train["year"] = df_train["date"].dt.year
df_train["is_weekend"] = (df_train["day_of_week"] >= 5).astype(int)  # 週末フラグ

# カテゴリ変数のエンコーディング

from sklearn.preprocessing import LabelEncoder

# Label Encoding
le = LabelEncoder()
df_train["store_encoded"] = le.fit_transform(df_train["store_nbr"])
df_train["family_encoded"] = le.fit_transform(df_train["family"])

# One-Hot Encoding（店舗タイプ & クラスター）
df_train = pd.get_dummies(df_train, columns=["type_store", "cluster"], drop_first=False)

# 時系列特徴量

# ラグ特徴量（過去の売上）
df_train["lag_7"] = df_train["sales"].shift(7)
df_train["lag_14"] = df_train["sales"].shift(14)
df_train["lag_30"] = df_train["sales"].shift(30)

# 移動平均特徴量（過去の売上の平均）
df_train["rolling_mean_7"] = df_train["sales"].shift(7).rolling(window=7).mean()
df_train["rolling_mean_30"] = df_train["sales"].shift(7).rolling(window=30).mean()

# 祝日フラグの作成

df_train["is_holiday"] = df_train["type_holiday"].apply(lambda x: 1 if x == "Holiday" else 0)
df_train["is_holiday"] = df_train["is_holiday"].fillna(0)

# 最終確認

print(df_train.head())  # 先頭5行を表示
print(df_train.info())  # データ型 & 欠損値確認