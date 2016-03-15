from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, ReluLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import numpy
import pandas


#data = numpy.loadtxt('./data/train.csv', delimiter=',',skiprows=1)
#num_features = len(data[0]) - 2
data = pandas.read_csv('./data/train.csv', delimiter=',')
num_features = len(data.columns) - 3

network = FeedForwardNetwork()
inLayer = LinearLayer(num_features)
#hiddenLayer = SigmoidLayer(num_features * 2)
hiddenLayer = ReluLayer(num_features)
outLayer = LinearLayer(1)

network.addInputModule(inLayer)
network.addModule(hiddenLayer)
network.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
network.addConnection(in_to_hidden)
network.addConnection(hidden_to_out)
network.sortModules()


# normalise the data?



# build the training data set
ds = SupervisedDataSet(num_features, 1)
#for row in data:
#    ds.addSample((row[1:-1]), tuple(row[-1:]))
for tuple_row in data.itertuples():
    #ds.addSample((row[1:-1]), tuple(row[-1:]))
    row = list(tuple_row)
#    print(row[3:])
    ds.addSample((row[3:-1]), tuple(row[-1:]))


trainer = BackpropTrainer(network, ds)

for i in range(5):
    error = trainer.train()
    print('Training iteration: {0} error {1}\r                   '.format(i, error), end='')

print()
# calc error on TS
test = data[1:51]

for tuple_row in data.itertuples():
    row = list(tuple_row)
    result = network.activate((row[3:-1]))
    print('Result: ' + str(result) + ' expected:' + str(row[-1:]))
