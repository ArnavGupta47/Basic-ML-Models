import pandas as pd

#defining where is the data
model_path = '''Write the file path'''
model_data = pd.read_csv(model_path)

#Prints all the columns in the dataset
print(str(model_data.columns))

#deciding the target
y = model_data.target
features = ['''features''']
x = model_data['''features''']

#review data
x.describe()
x.head()

#MAKING THE MODEL
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)#so that every time the data is randomised in a fixed pattern
model.fit(x, y)

#predictions
predictions = model.predict('''what to predict?''')
print(predictions)