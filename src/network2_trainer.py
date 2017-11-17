import mnist_loader
print "Loading data..."
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print "Complete!"
import network2
size = input("Please set the size of network: ")
net = network2.Network(list(size), cost=network2.CrossEntropyCost)
epochs, mini_batch_size, eta, lamda = input("Please set the iteration times, \
mini-batch size, learning rate and lamda: ")
# net.large_weight_initializer()
print "Well done, start training..."
net.SGD(training_data, epochs, mini_batch_size, eta, evaluation_data=test_data,
lmbda=lamda, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
monitor_training_cost=True,  monitor_training_accuracy=True)
print "Complete!"
