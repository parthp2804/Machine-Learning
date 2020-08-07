### Load the Data
import pandas as pd 
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
diabetes = pd.read_csv('diabetes.csv')
diabetes.columns = col_names


### features and labels
feature = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = diabetes[feature]
y = diabetes['label']

### splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

### Fit the model
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train,y_train)
#predict 
#y_pred = regressor.predict(X_test)
pred = regressor.predict([[4,95,33.8,59,150,70,0.647]])
prob = regressor.predict_proba([[4,95,33.8,59,150,70,0.647]])
print('prediction:', pred)
print('probability:', prob)
### Evaluate the model using Confusion matrix
from sklearn import metrics
cnf = metrics.confusion_matrix(y_test,y_pred)
print(cnf)

### Accuracy of the model
print('Accuracy:',metrics.accuracy_score(y_test,y_pred))
