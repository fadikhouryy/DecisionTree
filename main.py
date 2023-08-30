import pandas as pd
df = pd.read_csv(r'C:\Users\Trainee.MOMEN-KITTANEH\Desktop\Customer_Behaviour.csv')
#print(df.head())
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cat_col = ['Gender']
for cols in cat_col:
    df[cols] = encoder.fit_transform(df[cols])

#print(df.info())

features = ['Gender','Age','EstimatedSalary']
target =['Purchased']

X=df[features]
#print(X)
y=df[target]
#print(y)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree = tree.fit(X,y)
import matplotlib.pyplot as plt
#from sklearn import tree
#tree.plot_tree(tree, feature_names=features)
print(tree.score(X,y))

print(tree.predict([[0,20,20000]]))