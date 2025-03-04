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


