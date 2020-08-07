### Loading and exploring Data
from sklearn import datasets

wine_data = datasets.load_wine()
features =  wine_data.feature_names
labels = wine_data.target_names


### Split the train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine_data.data,wine_data.target, test_size = 0.3, random_state = 101)

### fit the model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)

### predict the model
y_pred = model.predict(X_test)


### Accuracy of the model
from sklearn import metrics
print('Accuracy:', metrics.accuracy_score(y_test,y_pred))