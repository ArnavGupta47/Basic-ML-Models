import pandas as pd
model_path = '''Write the file path'''
model_data = pd.read_csv(model_path)
y = model_data.target
features = ['''features''']
x = model_data['''features''']

from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y =train_test_split(x ,y ,random_state=1)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5,50,500,5000]#i for i in range(1, 10001) -->this takes too much time
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
d1=dict()
for i in candidate_max_leaf_nodes:
    mae = get_mae(i, train_x, val_x, train_y, val_y)
    d1.update({i: mae})
lowest_mae=min(d1.values())
for a, mae in d1.items():
    if mae == lowest_mae:
        best_leaf_node=a        

print(f"{best_leaf_node} is the best leaf node for this model.") # type: ignore
