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

######## adat kezelések ######
# 2. Adattisztítás
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
data['Total Revenue'] = data['Quantity Ordered'] * data['Price Each']

# 3. Elemzések
# Összbevétel
total_revenue = data['Total Revenue'].sum()
print(f"Összbevétel: {total_revenue}")

# Legjobb termékek
top_products = data.groupby('Product')['Total Revenue'].sum().sort_values(ascending=False)
print(top_products.head())

# Időbeli trendek
data['Month'] = data['Order Date'].dt.month
monthly_revenue = data.groupby('Month')['Total Revenue'].sum()
print(monthly_revenue)

# 4. Vizualizáció
# Havi bevétel
monthly_revenue.plot(kind='bar', title='Havi Bevétel')
plt.show()

# Legjobb termékek vizualizáció
top_products.head(10).plot(kind='bar', title='Top 10 Termékek')
plt.show()

# 5. Riport exportálása
data.to_excel('sales_report.xlsx', index=False)

#####################################################



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
