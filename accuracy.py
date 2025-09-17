import pandas as pd
model_path = '''Write the file path'''
model_data = pd.read_csv(model_path)
y = model_data.target
features = ['''features''']
x = model_data['''features''']

#EVALUATING THE ACCURACY OF THE MODEL
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y =train_test_split(x ,y ,random_state=1)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)
model.fit(train_x, train_y)

val_predictions = model.predict(val_x)

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)