import mnist_loader
print "Loading data..."
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print "Complete!"
import network
size = input("Please set the size of network: ")
net = network.Network(list(size))
epochs, mini_batch_size, eta = input("Please set the iteration times, mini-batch size and learning rate: ")
print "Well done, start training..."
net.SGD(training_data, epochs, mini_batch_size, eta, test_data=test_data)
print "Complete!"
