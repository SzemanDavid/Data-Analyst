# Loading my Google Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

!pip install openpyxl

file_path = '/content/drive/MyDrive/Python/Sales-data/sales_data_sample.csv'
sales_data_sample = pd.read_csv(file_path, encoding='latin1')
print(sales_data_sample.head())

# Searching for missing data
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



# Gépi tanulás modell választás, tanítás és tesztelés
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Adatok előkészítése
X = df[['month', 'day_of_week', 'product_category']]  # Kiválasztott jellemzők
y = df['sales']  # Eladások

# Képzés és tesztelés
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell betanítása
model = LinearRegression()
model.fit(X_train, y_train)

# Előrejelzés
y_pred = model.predict(X_test)

# Modell értékelése
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
