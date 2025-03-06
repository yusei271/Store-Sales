import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
#from sklearn.decomposition import PCA

# データを読み込む
df_train = pd.read_csv("train.csv")
df_oil = pd.read_csv("oil.csv")  # 修正箇所
df_stores = pd.read_csv("stores.csv")  # 他のデータも必要なら読み込む
df_test = pd.read_csv("test.csv")
#追加
df_holidays = pd.read_csv('holidays_events.csv')
df_transactions = pd.read_csv('transactions.csv')
sample_subm = pd.read_csv('sample_submission.csv')

# 日付をdatetime型に変換
df_train['date'] = pd.to_datetime(df_train['date'], format="%Y-%m-%d")
print(df_train['date'].head())
df_oil['date'] = pd.to_datetime(df_oil['date'], format="%Y-%m-%d")
print(df_oil['date'].head())
df_test['date'] = pd.to_datetime(df_test['date'], format="%Y-%m-%d")
print(df_test['date'].head())
df_holidays['date'] = pd.to_datetime(df_holidays['date'], format="%Y-%m-%d")
print(df_holidays['date'].head())
#追加
df_holidays['date'] = pd.to_datetime(df_train['date'], format="%Y-%m-%d")
df_transactions['date'] = pd.to_datetime(df_train['date'], format="%Y-%m-%d")


# データをマージ
df_train = pd.merge(df_train, df_stores, on="store_nbr", how="left")
df_train = pd.merge(df_train, df_transactions, on=["store_nbr", "date"], how="left")
df_train = pd.merge(df_train, df_holidays, on="date", how="left")
df_train = pd.merge(df_train, df_oil, on="date", how="left")

df_test = pd.merge(df_test, df_stores, on="store_nbr", how="left")
df_test = pd.merge(df_test, df_transactions, on=["store_nbr", "date"], how="left")
df_test = pd.merge(df_test, df_holidays, on="date", how="left")
df_test = pd.merge(df_test, df_oil, on="date", how="left")
print(df_train.head())
print(df_test.head())
#print(df_train.head())

# 欠損値の割合を計算
print(np.round(df_train.isna().sum(axis=0) / len(df_train), 4) * 100)
print(np.round(df_test.isna().sum(axis=0) / len(df_test), 4) * 100)
#欠損値の個数の確認
print(df_oil.isnull().sum())
#欠損値の補完
df_train.dcoilwtico = df_train.dcoilwtico.interpolate()
df_train.loc[df_train.dcoilwtico.isna(), "dcoilwtico"] = df_train.dcoilwtico.mean()
#欠損値の補完後の確認
print(df_train.head())

# 相関の確認 print(df_train.corr())

# salesの外れ値をIQRで除去する方法
Q1 = df_train['sales'].quantile(0.25)
Q3 = df_train['sales'].quantile(0.75)
IQR = Q3 - Q1
# 外れ値の条件を定義
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# 外れ値を除去
df_train_no_outliers = df_train[(df_train['sales'] >= lower_bound) & (df_train['sales'] <= upper_bound)]
# 外れ値除去後のデータを表示
print(df_train_no_outliers['sales'].describe())


# 重複データの確認と削除
print("\n重複データの数（trainデータ）:")
print(df_train.duplicated().sum())  # 重複している行の数
print("\n重複データの数（testデータ）:")
print(df_test.duplicated().sum())  # 重複している行の数
# 重複データの削除
df_train = df_train.drop_duplicates()  # 重複データ削除
df_test = df_test.drop_duplicates()  # 重複データ削除
# 重複データ削除後の確認
print("\n重複データ削除後（trainデータ）:")
print(df_train.duplicated().sum())  # 削除後の重複行の数
print("\n重複データ削除後（testデータ）:")
print(df_test.duplicated().sum())  # 削除後の重複行の数