import pandas
import numpy

# Load the train file into a dataframe
train_df = pandas.read_csv('./data/original/train.csv', header=0)

# convert Sex to Gender code
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Remove the Name, Ticket and Sex
train_df = train_df.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Fare'], axis=1)

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(numpy.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

print(train_df.head(10))

#train_df = train_df[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Gender', 'Survived']]
train_df = train_df[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Embarked', 'Gender', 'Survived']]

print(train_df.head(10))

train_df.to_csv('./data/train.csv', sep=',', encoding='utf-8', index=None)



# Load the test file into a dataframe
test_df = pandas.read_csv('./data/original/test.csv', header=0)

# convert Sex to Gender code
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Remove the Name, Ticket and Sex
test_df = test_df.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Fare'], axis=1)

# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

Ports = list(enumerate(numpy.unique(test_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

print(test_df.head(10))

test_df = test_df[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Embarked', 'Gender']]

print(test_df.head(10))

test_df.to_csv('./data/test.csv', sep=',', encoding='utf-8', index=None)