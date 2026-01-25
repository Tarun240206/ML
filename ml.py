import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
#1
df = pd.read_excel(
    r"C:\Users\ADMIN\Downloads\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

print(df.head())
X = df.iloc[:, 1:4].values
Y = df.iloc[:, 4].values
rank_X = np.linalg.matrix_rank(X)
X_pinv = np.linalg.pinv(X)
price_c = np.matmul(X_pinv,Y)
print("Feature matrix: ",X)
print("Output vector(payment):",Y)
print("Rank of X: ",rank_X)
print("Pseudo-inverse of X: ",X_pinv)
print("Price of candies,mangoes and milk packers: ",price_c)

#2

Y2 = (df["Payment (Rs)"] > 200).astype(int)

status_classifier = np.matmul(X_pinv,Y2)
pred_y = np.matmul(X,status_classifier)

predicted_status = (pred_y >= 0.5).astype(int)
labels = np.array(["POOR", "RICH"])
predicted_status = labels[predicted_status]

print(Y2)
print(predicted_status)

#3

df_2 = pd.read_excel(r"C:\Users\ADMIN\Downloads\Lab Session Data.xlsx",
    sheet_name="IRCTC Stock Price")
print(df_2.head())

irctc_all_price = df_2.iloc[:, 3].values
print("IRCTC price",irctc_all_price)

def cal_mean(price):
    sum = 0
    for i in price:
        sum += i
    mean = sum/len(price)
    return mean

def cal_var(price):
    mean = cal_mean(price)
    sum = 0
    for i in price:
        sum += (i - mean) ** 2
    variance = sum/len(price)
    return variance

def time_complexity_package_functions(price):
    start = time.time()
    for i in range(10):
        np.mean(price)
        np.var(price)
    end = time.time()
    avg_time = (end - start)/10

    return avg_time

def cal_time_complexity(price):
    start = time.time()
    for i in range(10):
        cal_mean(price)
        cal_var(price)
    end = time.time()
    avg_time = (end - start)/10
    return avg_time

wednesday_information = df_2[df_2["Day"] == "Wed"]
wednesday_price = wednesday_information["Price"]
population_price_mean = np.mean(irctc_all_price)
wednesday_price_mean = np.mean(wednesday_price)

change_in_stock_price = df_2.iloc[:, 8]
loss_rates = change_in_stock_price.apply(lambda x: x < 0)
days_of_loss = loss_rates.sum()
total_days = len(change_in_stock_price)
probability_of_loss = days_of_loss/total_days

wednesday_change_in_stock = wednesday_information["Chg%"]
wednesday_loss_rate = wednesday_change_in_stock.apply(lambda x: x < 0)
wed_days_of_loss = wednesday_loss_rate.sum()
probability_of_loss_wednesday = wed_days_of_loss/len(wednesday_change_in_stock)

wednesday_profit_rate = wednesday_change_in_stock.apply(lambda x: x > 0)
wed_days_of_profit = wednesday_profit_rate.sum()
probability_of_profit_wednesday = wed_days_of_profit/len(wednesday_change_in_stock)

plt.scatter(df_2["Day"],df_2["Chg%"])
plt.xlabel("Days of the week")
plt.ylabel("Change in Stock")

April_information = df_2[df_2["Month"] == "Apr"]
April_price = April_information["Price"]
April_price_mean = np.mean(April_price)

mean_package = np.mean(irctc_all_price)
variance_package = np.var(irctc_all_price)

my_mean = cal_mean(irctc_all_price)
my_variance = cal_var(irctc_all_price)

mean_accuracy = my_mean - mean_package
variance_accuracy = my_variance - variance_package

print("Mean accuracy: ",mean_accuracy)
print("variance_accuracy: ",variance_accuracy)
print("Time taken by my functions: ",cal_time_complexity(irctc_all_price))
print("Time taken by package functions: ",time_complexity_package_functions(irctc_all_price))
print("Wedneday price mean: ",wednesday_price_mean)
print("population price mean: ",population_price_mean)
print("The change in both the mean value(population and wednesday price mean): ",(population_price_mean-wednesday_price_mean))
print("April price mean: ",April_price_mean)
print("The change in both the mean value(population and April price mean): ",(April_price_mean-population_price_mean))
print("The probability of loss is: ",probability_of_loss)
print("The probability of loss on wednesday is: ",probability_of_loss_wednesday)
plt.show()

#4
df_3 = pd.read_excel(r"C:\Users\ADMIN\Downloads\Lab Session Data.xlsx",
    sheet_name="thyroid0387_UCI")
print(df_3.head())
print("Dataset Shape (rows, cols):", df_3.shape)
print("\nFirst 5 rows:\n", df_3.head())

print(df_3.info())

print(df_3.columns.tolist())

for col in df_3.columns:
    unique_count = df_3[col].nunique(dropna=True)
    if unique_count <= 15:  
        print(f"\nColumn: {col}")
        print("Unique values:", df_3[col].dropna().unique())


cat_cols = df_3.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = df_3.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical Columns:", cat_cols)
print("Numeric Columns:", num_cols)

print("\nFor NOMINAL categorical variables -> One-Hot Encoding")
print("For ORDINAL categorical variables -> Label Encoding")

print("\nCategorical columns found in your dataset:")
for col in cat_cols:
    print(f"- {col}  (Unique values: {df_3[col].nunique(dropna=True)})")

print("\nNOTE: To decide ordinal vs nominal, check if values have  order.")

if len(num_cols) > 0:
    numeric_range = df_3[num_cols].agg(["min", "max"])
    print(numeric_range)
else:
    print("No numeric columns found.")

missing_count = df_3.isna().sum()
missing_percent = (missing_count / len(df_3)) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing %": missing_percent
}).sort_values(by="Missing Count", ascending=False)

print(missing_table)


def count_outliers_iqr(series):
    series = series.dropna()
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), lower, upper

if len(num_cols) > 0:
    for col in num_cols:
        out_count, lower, upper = count_outliers_iqr(df_3[col])
        print(f"{col}: Outliers = {out_count}, Lower Bound = {lower:.3f}, Upper Bound = {upper:.3f}")
else:
    print("No numeric columns found for outlier detection.")


if len(num_cols) > 0:
    stats_table = pd.DataFrame({
        "Mean": df_3[num_cols].mean(),
        "Variance": df_3[num_cols].var(),     
    })
    print(stats_table)
else:
    print("No numeric columns found.")

#5
r1_and_r2 = df_3.iloc[0:2]
binary_columns = df_3.columns[df_3.isin(['t', 'f']).all()]
binary_information = r1_and_r2[binary_columns].replace({'t': 1, 'f': 0})
vector_1 = binary_information.iloc[0].values
vector_2 = binary_information.iloc[1].values
f11,f10,f01,f00 = 0,0,0,0
for i,j in zip(vector_1,vector_2):
    if i == 1 and j == 1:
        f11 += 1
    if i == 1 and j == 0:
        f10 += 1
    if i == 0 and j == 1:
        f01 += 1
    if i == 0 and j == 0:
        f00 += 1
jaccard_coefficient = f11/(f11+f10+f01)
simple_matching_coefficient = (f11+f00)/(f11+f10+f01+f00)
print("Jaccord Coefficient(JC): ",jaccard_coefficient)
print("Simple matching Coefficient(SMC): ",simple_matching_coefficient)

#6
num_df_3=df_3.select_dtypes(include=[np.number])
A=num_df_3.iloc[0].values
B=num_df_3.iloc[1].values

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(cosine_similarity(A,B))

#7
num_df = df_3.select_dtypes(include=[np.number]).iloc[:20]
binary_df = (num_df > num_df.mean()).astype(int)

def J_C(x, y):
    f11, f10, f01, f00 = 0, 0, 0, 0

    for i, j in zip(x, y):
        if i == 1 and j == 1:
            f11 += 1
        elif i == 1 and j == 0:
            f10 += 1
        elif i == 0 and j == 1:
            f01 += 1
        elif i == 0 and j == 0:
            f00 += 1

    if (f11 + f10 + f01) == 0:
        return 0

    j_c = f11 / (f11 + f10 + f01)
    return j_c

def S_M_C(x, y):
    f11, f10, f01, f00 = 0, 0, 0, 0

    for i, j in zip(x, y):
        if i == 1 and j == 1:
            f11 += 1
        elif i == 1 and j == 0:
            f10 += 1
        elif i == 0 and j == 1:
            f01 += 1
        elif i == 0 and j == 0:
            f00 += 1

    if (f11 + f10 + f01 + f00) == 0:
        return 0

    s_m_c = (f11+f00)/(f11+f10+f01+f00)
    return s_m_c

def C_O_S(x,y):
    return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))


results = []

for i, j in combinations(range(20), 2):
    results.append({
        "Vector Pair": f"({i+1}, {j+1})",
        "Jaccard (JC)": J_C(binary_df.iloc[i], binary_df.iloc[j]),
        "SMC": S_M_C(binary_df.iloc[i], binary_df.iloc[j]),
        "Cosine (COS)": C_O_S(num_df.iloc[i], num_df.iloc[j])
    })

similarity_df = pd.DataFrame(results)
print(similarity_df.head(10))


n = 20
JC = np.zeros((n, n))
SMC = np.zeros((n, n))
COS = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        JC[i, j] = J_C(binary_df.iloc[i], binary_df.iloc[j])
        SMC[i, j] = S_M_C(binary_df.iloc[i], binary_df.iloc[j])
        COS[i, j] = C_O_S(num_df.iloc[i], num_df.iloc[j])
    

plt.figure(figsize=(8, 6))
sns.heatmap(JC, cmap="Blues", annot=True)
plt.title("Jaccard Coefficient Heatmap")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(SMC, cmap="Greens", annot=True)
plt.title("Simple Matching Coefficient Heatmap")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(COS, cmap="Reds", annot=True)
plt.title("Cosine Similarity Heatmap")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()







