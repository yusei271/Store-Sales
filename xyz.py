import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

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
#df_stores['store_nbr'] = pd.to_datetime(df_stores['store_nbr'], format="%Y-%m-%d")
#print(df_stores['store_nbr'].head())
#df_test['store_nbr'] = pd.to_datetime(df_test['store_nbr'], format="%Y-%m-%d")
#print(df_test['store_nbr'].head())

# データをマージ
df_train = pd.merge(df_train, df_stores, on="store_nbr", how="left")
df_train = pd.merge(df_train, df_transactions, on=["store_nbr", "date"], how="left")
df_train = pd.merge(df_train, df_holidays, on="date", how="left")
df_train = pd.merge(df_train, df_oil, on="date", how="left")
#df_train = df_train.merge(df_oil, on='date')
#df_train = df_train.merge(df_stores, on='store_nbr')
#df_test = df_test.merge(df_oil, on='date')
#df_test = df_test.merge(df_stores, on='store_nbr')
df_test = pd.merge(df_test, df_stores, on="store_nbr", how="left")
df_test = pd.merge(df_test, df_transactions, on=["store_nbr", "date"], how="left")
df_test = pd.merge(df_test, df_holidays, on="date", how="left")
df_test = pd.merge(df_test, df_oil, on="date", how="left")
print(df_train.head())
print(df_test.head())

print(df_train.head())
# 欠損値の割合を計算
print(np.round(df_train.isna().sum(axis=0) / len(df_train), 4) * 100)
print(np.round(df_test.isna().sum(axis=0) / len(df_test), 4) * 100)
#欠損値の個数の確認
print(df_oil.isnull().sum())

#欠損値の補完
df_train.dcoilwtico = df_train.dcoilwtico.interpolate()
df_train.loc[df_train.dcoilwtico.isna(), "dcoilwtico"] = df_train.dcoilwtico.mean()
print(df_train.head())

print(df_train.corr())

# --- 外れ値・異常値の削除 ---
# 売上データ (sales) の外れ値を削除
#q1 = df_train['sales'].quantile(0.25)
#q3 = df_train['sales'].quantile(0.75)
# iqr = q3 - q1
#lower_bound = q1 - 1.5 * iqr
#upper_bound = q3 + 1.5 * iqr
#df_train = df_train[(df_train['sales'] >= lower_bound) & (df_train['sales'] <= upper_bound)]
# --- 重複データの削除 ---
#df_train = df_train.drop_duplicates()

# 確認
#print("After cleaning:")
#print(df_train.isna().sum())
#print(df_train.duplicated().sum())
#print(df_train.describe())

# --- エンコーディング (カテゴリカルデータの変換) ---
# Label Encoding
#categorical_columns = ['type', 'cluster']  # カテゴリカル変数を指定
#label_encoders = {}
#for col in categorical_columns:
#    le = LabelEncoder()
#    df_train[col] = le.fit_transform(df_train[col])
#    label_encoders[col] = le

print(df_train.head(10))

# 必要な列の確認・抽出（例：'type_x ', 'Sales'）
# state: 州, Sales: 売上額
store_sales = df_train.groupby('type_x')['sales'].sum().reset_index()
# グラフの作成
plt.figure(figsize=(10, 6))
sns.barplot(x='type_x', y='sales', data=store_sales, palette='viridis')
# グラフのカスタマイズ
plt.title('Sales by type_x', fontsize=16)
plt.xlabel('type_x', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
# グラフを表示
plt.show()


# 必要な列の確認・抽出（例：'date ', 'Sales'）
# state: 州, Sales: 売上額
store_sales = df_train.groupby('date')['sales'].sum().reset_index()
# グラフの作成
plt.figure(figsize=(10, 6))
sns.barplot(x='date', y='sales', data=store_sales, palette='viridis')
# グラフのカスタマイズ
plt.title('Sales by date', fontsize=16)
plt.xlabel('date', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
# グラフを表示
plt.show()


# 必要な列を抽出（都市名と売上額の列を確認して変更）
city_column = 'city'  # 都市名が格納されている列名
sales_column = 'sales'  # 売上額が格納されている列名
# 都市ごとの売上額を集計
city_sales = df_train.groupby(city_column)[sales_column].sum().reset_index()
print(city_sales)
# 棒グラフの作成
plt.figure(figsize=(12, 6))
plt.bar(city_sales[city_column], city_sales[sales_column], color='skyblue')
plt.xlabel('City', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.title('Total Sales by City', fontsize=16)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
# グラフを表示
plt.show()



# 必要な列の確認・抽出（例：'state', 'Sales'）
# state: 州, Sales: 売上額
store_sales = df_train.groupby('state')['sales'].sum().reset_index()
#store_sales = df_train.groupby('state')['sales'].mean().reset_index()
# グラフの作成
plt.figure(figsize=(10, 6))
sns.barplot(x='state', y='sales', data=store_sales, palette='viridis')
# グラフのカスタマイズ
plt.title('Sales by state', fontsize=16)
plt.xlabel('state', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
# グラフを表示
plt.show()


# 必要な列の確認・抽出（例：'cluster', 'Sales'）
# clustr: お店の種類, Sales: 売上額
store_sales = df_train.groupby('cluster')['sales'].sum().reset_index()
# グラフの作成
plt.figure(figsize=(10, 6))
sns.barplot(x='cluster', y='sales', data=store_sales, palette='viridis')
# グラフのカスタマイズ
plt.title('Sales by cluster', fontsize=16)
plt.xlabel('cluster', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
# グラフを表示
plt.show()


# 必要な列の確認・抽出（例：'family', 'Sales'）
# family:販売されている製品の種類 , Sales: 売上額
store_sales = df_train.groupby('family')['sales'].sum().reset_index()
# グラフの作成
plt.figure(figsize=(10, 6))
sns.barplot(x='family', y='sales', data=store_sales, palette='viridis')
# グラフのカスタマイズ
plt.title('Sales by family', fontsize=16)
plt.xlabel('family', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
# グラフを表示
plt.show()

# 必要な列の確認・抽出（例：'store_nbr', 'Sales'）
# store_nbr:店番号 , Sales: 売上額
store_sales = df_train.groupby('store_nbr')['sales'].sum().reset_index()
# グラフの作成
plt.figure(figsize=(10, 6))
sns.barplot(x='store_nbr', y='sales', data=store_sales, palette='viridis')
# グラフのカスタマイズ
plt.title('Sales by store_nbr', fontsize=16)
plt.xlabel('store_nbr', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
# グラフを表示
plt.show()




# # 必要な列の確認・抽出
# # dcoilwtico:原油価格, Sales: 売上額
# store_sales = df_train.groupby('dcoilwtico')['sales'].sum().reset_index()
# # 散布図の描画
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='dcoilwtico', y='sales', data=store_sales, alpha=0.5)
# # 散布図にタイトルとラベルを追加
# plt.title('Scatter Plot of Sales vs Oil Price', fontsize=14)
# plt.xlabel('Oil Price')
# plt.ylabel('Sales')
# plt.grid(True)
# plt.show()


# データを読み込む (ファイル名を適切に置き換えてください)
# 例: "train.csv" がデータセットの名前だと仮定
data = df_train

# 必要なカラムのみ抽出
# "onpromotion" がプロモーション中の商品数、"sales" が売上額を仮定
x = data["onpromotion"]
y = data["sales"]

# 散布図を描画
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5, color='blue', s=10)  # alphaで透明度を調整
plt.title("Connection between sales and promotion products.")
plt.xlabel("Number of promotion products (onpromotion)")
plt.ylabel("Sales")
plt.grid(True)
plt.show()


# データを読み込む (ファイル名を適切に置き換えてください)
# 例: "train.csv" がデータセットの名前だと仮定
data = df_train

# 必要なカラムのみ抽出
# "onpromotion" がプロモーション中の商品数、"sales" が売上額を仮定
x = data["dcoilwtico"]
y = data["sales"]

# 散布図を描画
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5, color='blue', s=10)  # alphaで透明度を調整
plt.title("Connection between  Sales and Oil")
plt.xlabel("oil price (dcoilwtico)")
plt.ylabel("Sales")
plt.grid(True)
plt.show()
# 必要な列の確認・抽出
# number of the promotion products:プロモーションされた製品数, Sales: 売上額
#store_sales = df_train.groupby('number of the promotion products')['sales'].sum().reset_index()
# 散布図の描画
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='number of the promotion products', y='sales', data=store_sales, alpha=0.5)
# # 散布図にタイトルとラベルを追加
# plt.title('Scatter Plot of Sales vs number of the promotion products', fontsize=14)
# plt.xlabel('number of the promotion products')
# plt.ylabel('Sales')
# plt.grid(True)
# plt.show()



# --- 標準化 (Standardization) ---
# scaler = StandardScaler()
# numerical_columns = ['sales', 'onpromotion', 'dcoilwtico']  # 数値カラムを指定
# df_train[numerical_columns] = scaler.fit_transform(df_train[numerical_columns])

# --- 次元削減 (PCA) ---
# pca = PCA(n_components=2)  # 主成分数を2に設定
# reduced_features = pca.fit_transform(df_train[numerical_columns])
# df_train['pca_1'] = reduced_features[:, 0]
# df_train['pca_2'] = reduced_features[:, 1]

# 確認
# print("After cleaning, encoding, and PCA:")
# print(df_train.head())
# print("Explained variance ratio:", pca.explained_variance_ratio_)

#グラフ作成
#x=df_train['city']
#y=df_train['sales']
#plt.bar(x,y)
#plt.title('都市名（横軸）と売上額（縦軸）の相関')
#plt.xlabel('都市名')
#plt.xticks(rotation=30)
#plt.ylabel('売上額')
#plt.grid(True)
#plt.show()