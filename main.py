import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import sys

my_csv_file = sys.argv[1]
with open(my_csv_file,'r') as f:
    f_cont = f.read()

df=pd.read_csv(my_csv_file)




encoder = LabelEncoder()
cat_col = ['Gender']
for cols in cat_col:
    df[cols] = encoder.fit_transform(df[cols])



features = ['Gender','Age','EstimatedSalary']
target =['Purchased']

X=df[features]

y=df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


tree = DecisionTreeClassifier()

tree = tree.fit(X_train,y_train)

predicted=tree.predict(X_test)

confusion_matrix = metrics.confusion_matrix(y_test, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


