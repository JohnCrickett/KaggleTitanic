from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, ReluLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import pandas


def normalise(value, data_high, data_low):
    return (value - data_low) / (data_high - data_low)


def build_network(num_features):
    network = FeedForwardNetwork()
    inLayer = LinearLayer(num_features)
    hiddenLayer = SigmoidLayer(num_features * 7)
    #hiddenLayer = ReluLayer(num_features * 2)
    outLayer = LinearLayer(1)

    network.addInputModule(inLayer)
    network.addModule(hiddenLayer)
    network.addOutputModule(outLayer)

    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)
    network.addConnection(in_to_hidden)
    network.addConnection(hidden_to_out)
    network.sortModules()
    return network




if __name__ == '__main__':
    data = pandas.read_csv('./data/train.csv', delimiter=',')
    num_features = len(data.columns) - 3

    network = build_network(num_features)

    # normalise the data
    # calculate range for each column
    max_age = data['Age'].max()
    min_age = data['Age'].min()
    max_class = data['Pclass'].max()
    min_class = data['Pclass'].min()
    min_age_nrange = min_age
    max_age_nrange = max_age
    max_class_nrange = max_class
    min_class_nrange = min_class

    data['Age'] = data['Age'].apply(normalise, args=(max_age_nrange, min_age_nrange))
    data['Pclass'] = data['Pclass'].apply(normalise, args=(max_class_nrange, min_class_nrange))

    # build the training data set
    ds = SupervisedDataSet(num_features, 1)
    for tuple_row in data.itertuples():
        row = list(tuple_row)
        ds.addSample((row[3:-1]), tuple(row[-1:]))

    trainer = BackpropTrainer(network, ds)

    for i in range(500):
        error = trainer.train()
        print('\rTraining iteration: {0} error {1}\t\t\t\t\t'.format(i, error), end='')

    print()

    correct = 0

    for tuple_row in data.itertuples():
        row = list(tuple_row)
        result = network.activate((row[3:-1]))
        result = 1.0 if result >= 0.5 else 0.0
        if result == row[-1]:
            correct += 1

    accuracy = correct / len(data)
    print("Score: " + str(accuracy))