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