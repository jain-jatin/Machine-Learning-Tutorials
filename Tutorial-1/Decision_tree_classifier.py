from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

#X = df.drop(['playtennis'], axis=1)
#Y = df['playtennis']

#le = LabelEncoder()
#test = X.column.values

#for col in test:
 #   le.fit(X[col])
 #   X[col] = le.transform(X[col])
   
#le.fit(Y)
#Y=le.transform(Y)

x,x_test,y,y_test=train_test_split(X,Y,test_size=0.37,random_state=0)

clf = DecisionTreeClassifier(criterion='entropy',random_state=0, max_depth=3)
clf = clf.fit(x, y)

y_predict = clf.predict(x_test);
a = confusion_matrix(y_test, y_predict)
print(a)

t=0
for i in range(0,len(a)):
     t+=a[i][i]
   
print("Accuracy is: "+str(t*100.0/len(y_test))+"%")
 
export_graphviz(clf, out_file="tree.dot") 