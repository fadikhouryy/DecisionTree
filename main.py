import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import sys

my_csv_file = sys.argv[1]
df=pd.read_csv(my_csv_file)

encoder = LabelEncoder()
cat_coll = ['Gender']
for cols in cat_coll:
    df[cols] = encoder.fit_transform(df[cols])


test_file = sys.argv[2]
test = pd.read_csv(test_file)
for cols in cat_coll:
    test[cols] = encoder.fit_transform(test[cols])



features = ['Gender','Age','EstimatedSalary']
target =['Purchased']

X=df[features]
y= df[target]



def model_fit(X,y,Gender,Age,EstimatedSalary,test):
    tree = DecisionTreeClassifier()
    tree = tree.fit(X,y)
    predict_value = tree.predict([[Gender,Age,EstimatedSalary]])

    print(test[['User ID','Gender','Age','EstimatedSalary','Purchased']])
    return predict_value


print(model_fit(X,y,0,29,135000,test))















