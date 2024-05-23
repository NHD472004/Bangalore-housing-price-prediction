import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv('./data/bhk.csv', usecols=['location', 'total_sqft', 'bath', 'balcony', 'price', 'bhk'])

X = df.drop('price', axis='columns')
y = df['price']

X_dummies = pd.get_dummies(df['location']).astype(int)
# print(X_dummies.head())
X_concat = pd.concat([df, X_dummies.drop('other', axis='columns')], axis='columns')
X_final = X_concat.drop(['location', 'price'], axis='columns')
# print(X_final.head())


X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
# print(X_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(X_final.head())


def predict_price(location, sqrt, bath, balcony, bhk):
    loc_index = np.where(X_final.columns == location)[0][0]
    
    x = np.zeros(len(X_final.columns))
    x[0] = sqrt
    x[1] = bath
    x[2] = balcony
    x[3] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
        
    return model.predict([x])[0]

print(predict_price('Indira Nagar', 1000, 3, 2, 3))

with open('./models/model_predict_house_price.pkl', 'wb') as file:
    pickle.dump(model, file)