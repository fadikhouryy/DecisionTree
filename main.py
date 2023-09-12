import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
from sklearn.metrics import accuracy_score

#loading training data
my_csv_file = sys.argv[1]
df=pd.read_csv(my_csv_file)

encoder = LabelEncoder()
cat_coll = ['Gender']
for cols in cat_coll:
    df[cols] = encoder.fit_transform(df[cols])


#loading data for prediction
test_file = sys.argv[2]
df_test = pd.read_csv(test_file)
cat_col =['Gender']
for cols in cat_col:
    df_test[cols] = encoder.fit_transform(df_test[cols])


#split the training data
Feature = ['User ID' , 'Gender','Age','EstimatedSalary']
X = df[Feature]
y = df['Purchased']

#train_test_split the training data

X_train , X_test , y_train , y_test = train_test_split(X , y ,test_size=0.247,random_state=42)

#fiting the data
clf =DecisionTreeClassifier()
clf.fit(X_train,y_train)

#predict data
predicted_data = clf.predict(df_test)

predicted_data = pd.DataFrame(predicted_data)

new_predicted_df =  pd.concat([df_test , predicted_data],axis=1,join="inner")
print(new_predicted_df)
#confusion_matrix
confusion_matrix = metrics.confusion_matrix(y_test, predicted_data)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

#store result in file
file =open("result_file.txt",'a')
for i in list(new_predicted_df):
   file.write(f"{i} \n")



