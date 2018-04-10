import pandas as pd
import numpy as np;
from sklearn.preprocessing import LabelEncoder
def gauss(x,mi,va):
  return np.exp(-((x-mi)**2)/(2*va))/np.sqrt(2*np.pi*va);
  
def labelencoder(data1):
  le = LabelEncoder()
  for col in data1.columns:
    if data1[col].dtype == 'object':
      le.fit(data1[col])
      data1[col] = le.transform(data1[col])
  return data1

data = pd.read_csv('data1.csv', index_col=None);
data = labelencoder(data)
X = data.drop('sex',axis=1);
Y = data['sex'];

#y no. of diff outputs
#m is rows = 8
#n is columns = 3
X = X.values
Y=Y.values
y=len(set(Y));
m=len(X);
n=len(X[0]);

#c contains count for diff outputs
c=np.zeros((y,1));

for d in Y:
  c[d]+=1;

mean = np.zeros((y,n));
var  = np.zeros((y,n));

for i in range(n):
  for j in range(m):
    mean[Y[j]][i]+=X[j][i];
mean = mean/c;

for i in range(n):
  for j in range(m):
    var[Y[j]][i]+=(X[j][i]-mean[Y[j]][i])**(2);
var = var/(c-1);

#testing
a=[[6, 130, 8],[5,120,7],[5.5,70,8]];

ans = np.zeros((len(a),1));
pro = c/m; #probability of different outputs
for k in range(len(a)):
  con_prob = np.zeros((y,n));
  post = np.zeros((y,1));
  for i in range(y):
    for j in range(n):
      con_prob[i][j]=gauss(a[k][j],mean[i][j],var[i][j]);
  post =pro*np.prod(con_prob,axis=1).reshape(y,1);
  evidence = post.sum();
  ymap=post/evidence;
  ans[k][0]=np.argmax(ymap);
print(np.hstack((a,ans)));