import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

model_path = '''Write the file path'''
model_data = pd.read_csv(model_path)
y = model_data.target
features = ['''features''']
x = model_data['''features''']

train_x, val_x, train_y, val_y =train_test_split(x ,y ,random_state=1)

#model1 without any node
model1 = DecisionTreeRegressor(random_state=1)
model1.fit(train_x, train_y)
val_predictions = model1.predict(val_x)
val_mae1 = mean_absolute_error(val_y, val_predictions)
print(val_mae1)

#model2 with a specifies node
model2 = DecisionTreeRegressor(max_leaf_nodes=100 ,random_state=1)
model2.fit(train_x, train_y)
val_predictions = model2.predict(val_x)
val_mae2 = mean_absolute_error(val_y, val_predictions)
print(val_mae2)

#THE RANDOM FORET MODEL
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_x, train_y)
val_predictions = rf_model.predict(val_x)
rf_val_mae = mean_absolute_error(val_predictions, val_y)
print(rf_val_mae)