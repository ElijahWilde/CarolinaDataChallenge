# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:42:51 2019

@author: Eli
"""
# Import Dependencies ---------------------------------------------------------
print("Importing dependencies...")
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Import Data -----------------------------------------------------------------
print("Importing data...")
df = pd.read_csv (r'C:\Users\Eli\Desktop\Python\sales_week_anon_201909241613.csv')

# Format Data -----------------------------------------------------------------
print("Formating data...")
df = df.drop('item_id', axis=1)
df = df.drop('store_id', axis=1)
df.insert(0, 'units_sold', df.sales_units)
df = df.drop('sales_units', axis=1)

def subtract2000(numba):
    return (numba - 2000)

mean = df['units_sold'].mean()
def simplify(numba):
    return int(numba > mean)

df['"year"'].apply(subtract2000)





#CHOOSE YOUR OWN ADVENTURE: drop columns to see how it affects the networks performance
#Dont forget to change the "shape" variable
#df = df.drop('week', axis=1)
#df = df.drop('product_pyramid', axis=1)
#df = df.drop('end_use', axis=1)
#df = df.drop('solid_novelty', axis=1)
#df = df.drop('offer_type', axis=1)
#df = df.drop('color_family', axis=1)
#df = df.drop('"year"', axis=1)




encodableColumn= 0
columnsToEncode = []
for column in df.columns:
    if (type(df[column][0]) == str):
        encodableColumn += 1
        columnsToEncode.append(column)
if (encodableColumn):
    df = pd.get_dummies(df, prefix_sep="__", columns=columnsToEncode)

x = df.drop('units_sold', axis=1)
y = df['units_sold']

min_max_scaler = preprocessing.MinMaxScaler()
x_scale = min_max_scaler.fit_transform(x)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(x_scale, y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

model = Sequential([
    Dense(32, activation='relu', input_shape=(47,)), #CHANGE THIS VARIABLE IF YOU DROP THINGS
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
    
# Training network ------------------------------------------------------------
print("Training network...")

model.compile(optimizer='sgd',
          loss='binary_crossentropy',
          metrics=['accuracy'])

train = model.fit(X_train, Y_train,
          batch_size=32, epochs=10,
          validation_data=(X_val, Y_val))

print("model is {}% accurate!".format(model.evaluate(X_test, Y_test)[1]))

