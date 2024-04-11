import pandas as pd

dataSetPath = 'dataset.csv'
df = pd.read_csv(dataSetPath, encoding='utf-8')

count_row = df.shape[0]  # Gives number of rows
count_col = df.shape[1]  # Gives number of columns

print("Row Counts =", count_row)
print("Columns Counts =", count_col)