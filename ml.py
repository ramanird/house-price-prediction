import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
df= pd.read_csv('data.csv')

X = df.drop(['date','street','city','statezip','country'],axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, color='yellow', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Prediction')
plt.xlabel('ACTUAL PRICE')
plt.ylabel('PREDICTED PRICE')
plt.title('ACTUAL VS PREDICTED HOUSE PRICE')
plt.legend()
plt.grid(True)
plt.show()