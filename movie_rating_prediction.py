import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:\\Users\\manik\\OneDrive\\Documents\\Data_sets\\imbd_movies_india.csv",encoding='latin')
df.head()

df.dropna(inplace=True)


X = df[['Genre', 'Director', 'Actor 1','Actor 2','Actor 3']]
y = df['Rating']


X_encoded = pd.get_dummies(X, columns=['Genre', 'Director', 'Actor 1','Actor 2','Actor 3'], drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import mean_squared_error
# Evaluate the model's performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
