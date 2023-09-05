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

X = df[features]
y = df[target]




def model_fit(user,featurs,target,test):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier()
    tree = tree.fit(featurs,target)
    for i in user.iloc[:100]:
        predicted_value = tree.predict(test.iloc[:100])
    predicted_value = pd.DataFrame(predicted_value)
    clas =  pd.concat([user , predicted_value],axis=1,join="inner")
    clas.rename(columns={0: 'class'}, inplace=True)
    return clas

user = test.drop(['Gender' , 'Age' , 'EstimatedSalary' , 'Purchased'], axis='columns')
drop = test.drop(['User ID' , 'Purchased'], axis='columns')

print(model_fit(user,X,y,drop))










