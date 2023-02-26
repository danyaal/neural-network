import numpy
import matplotlib.pyplot
import imageio.v2 as imageio
import tkinter
from tkinter import filedialog

from neural_network import NeuralNetwork
from util.helpers import bcolors

##################
# Input functions
##################
def getTrainingSize():
    while True:
        print(bcolors.WARNING + "Do you want to train with:" + bcolors.ENDC)
        print("A) Large training set (60,000 images)")
        print("B) Small training set (100 images)")
        input_training_size = input("[A/B] : ")

        if input_training_size in ['A', 'B']:
          break
    
    return input_training_size

def getTestSize():
  while True:
      print(bcolors.WARNING + "Do you want to test with:" + bcolors.ENDC)
      print("A) Large test set (10,000 images)")
      print("B) Small test set (10 images)")
      print("C) My own image")
      input_test_size = input("[A/B/C] : ")

      if input_test_size in ['A', 'B', 'C']:
          break

  return input_test_size

def getLearningRate():
  while True:
      # input_learning_rate = 0.3
      input_learning_rate = float(input(bcolors.WARNING + "Please set a learning rate (0.3 recommended): " + bcolors.ENDC))

      if input_learning_rate < 1 and input_learning_rate > 0:
          break

  return input_learning_rate

def getContinue():
    while True:
        print(bcolors.WARNING + "Would you like to continue?" + bcolors.ENDC)
        print("A) Yes")
        print("B) No")
        input_continue = input("[A/B] : ")

        if input_continue in ['A', 'B']:
          break
    
    return input_continue

##################
# Main
##################
print("\n")
print(bcolors.HEADER + "**************************************" + bcolors.ENDC)
print(bcolors.HEADER + "Welcome to Danyaal's Neural Network!!" + bcolors.ENDC)
print(bcolors.HEADER + "**************************************" + bcolors.ENDC)
print("In this script, we will train a neural network to recognize handwritten numbers")
print("\n")

input_training_size = getTrainingSize()
print("\n")
input_test_size = getTestSize()
print("\n")
input_learning_rate = getLearningRate()
print("\n")

input_nodes = 784;
hidden_nodes = 100;
output_nodes = 10;

learning_rate = input_learning_rate;

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);

##################
# Train
##################
print("Training...")
print("\n")
training_file = "numbers_assets/mnist_dataset/mnist_train.csv" if input_training_size == "A" else "numbers_assets/mnist_dataset/mnist_train_100.csv"
training_data_file = open(training_file, "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# go through all records in the training data set
for record in training_data_list:
    all_values = record.split(',')
    target_label = all_values[0]

    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create the target output values (all 0.01, except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(target_label)] = 0.99

    n.train(inputs, targets)
    pass

##################
# Test
##################
print("Testing...")
print("\n")

if input_test_size != 'C':
    test_file = "numbers_assets/mnist_dataset/mnist_test.csv" if input_test_size == "A" else "numbers_assets/mnist_dataset/mnist_test_10.csv"
    test_data_file = open(test_file, "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        all_values = record.split(',')
        # correct answer is teh first value
        correct_label = int(all_values[0])
        print(bcolors.OKCYAN + "image label:", correct_label, bcolors.ENDC)
        # scale and shift inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        print(bcolors.OKGREEN + "network guess:", label, bcolors.ENDC)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches the correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answero, add 0 to scorecard
            scorecard.append(0)
            pass
        print("\n")
        pass

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print (bcolors.HEADER + "neural network performance =", scorecard_array.sum() / scorecard_array.size, bcolors.ENDC)
else:
    print("\n")

    while True:
      print(bcolors.WARNING + "Please upload a 28x28 png: " + bcolors.ENDC)

      # collect user input
      root = tkinter.Tk()
      root.wm_withdraw() # this completely hides the root window
      img_filename = filedialog.askopenfilename(initialdir=".", title="Select an image", filetypes=(("png files", "*.png"),))
      root.destroy()

      # load image data from png files into an array
      print("\n")
      print ("loading ... ", img_filename)
      img_array = imageio.imread(img_filename, as_gray=True)
          
      # reshape from 28x28 to list of 784 values, invert values
      img_data  = 255.0 - img_array.reshape(784)

      # then scale data to range from 0.01 to 1.0
      img_data = (img_data / 255.0 * 0.99) + 0.01

      # plot image
      matplotlib.pyplot.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None')
      matplotlib.pyplot.show(block=False)

      # query the network
      outputs = n.query(img_data)

      # the index of the highest value corresponds to the label
      network_guess = numpy.argmax(outputs)
      print(bcolors.OKGREEN + "network guess:", network_guess, bcolors.ENDC)
      matplotlib.pyplot.show()

      print("\n")

      if getContinue() == "B":
        print("\n")
        break
      else:
        print("\n")
