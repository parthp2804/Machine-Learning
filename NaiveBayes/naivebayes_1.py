# Assigning data as in features and label

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

### Encode them into numbers
from sklearn import preprocessing

encode  = preprocessing.LabelEncoder()
weather_en = encode.fit_transform(weather)
print(weather_en)
temp_en = encode.fit_transform(temp)
print(temp_en)
play_en = encode.fit_transform(play)
print(play_en)

### Combine all the features
X = list(zip(weather_en,temp_en))
print(X)

### Fit the model
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X,play_en)

### Predict the output
predicted = model.predict([[0,2]])
print("Predicted value:",predicted)