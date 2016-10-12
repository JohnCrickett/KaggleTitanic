import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('./data/train.csv')

features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Embarked', 'Gender']

train, test = cross_validation.train_test_split(data, test_size = 0.7, random_state=0)

model = RandomForestClassifier(n_estimators=100)
model.fit(train[features], train['Survived'])


test_data = pd.read_csv('./data/test.csv')

test_data['Survived'] = model.predict(test_data[features])

print(test_data['Survived'])

test_data.to_csv("./predictions.csv", columns= ('PassengerId', 'Survived'), index=None)

