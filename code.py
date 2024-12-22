#saját google drive betöltése
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pip install openpyxl

file_path = '/content/drive/MyDrive/Datasets/sales_data.csv'
sales_data = pd.read_csv(file_path)
print(sales_data.head())

# Hiányzó adatok keresése
print(sales_data.isnull().sum())

# Duplikált sorok eltávolítása
sales_data = sales_data.drop_duplicates()

# Alap statisztikai adatok
print(sales_data.describe())

# Eladások eloszlása (példa)
plt.figure(figsize=(10, 6))
sns.histplot(sales_data['Sales'], kde=True)
plt.title('Sales Distribution')
plt.show()
