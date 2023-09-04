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
y=df[target]


tree = DecisionTreeClassifier()
tree = tree.fit(X,y)



user = test.drop(['Gender','Age','EstimatedSalary','Purchased'], axis='columns')

for i in user.iloc[:100]:
    drop = test.drop(["User ID", 'Purchased'], axis='columns')
    predicted = tree.predict(drop.iloc[:100])

predicted = pd.DataFrame(predicted)
result = (user,predicted)






f = open("demofile2.txt", "a")
for i in list(result):
   f.write(f"{i} \n")


