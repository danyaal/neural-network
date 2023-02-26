import numpy
import matplotlib.pyplot
import imageio.v2 as imageio
import tkinter
from tkinter import filedialog
import glob
import random

from neural_network import NeuralNetwork
from util.helpers import bcolors

##################
# Input functions
##################
def getLearningRate():
  while True:
      # input_learning_rate = 0.3
      input_learning_rate = float(input(bcolors.WARNING + "Please set a learning rate: " + bcolors.ENDC))

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
print("In this script, we will train a neural network to recognize faces!")
print("\n")

input_learning_rate = getLearningRate()
print("\n")

input_nodes = 784;
hidden_nodes = 100;
output_nodes = 2;

learning_rate = input_learning_rate;

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);

##################
# Train
##################
print("Training...")
print("\n")

# our own image test data set
training_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('face_assets/face_dataset/*.png'):
    # load image data from png files into an array
    img_array = imageio.imread(image_file_name, as_gray=True)

    # use the filename to set the correct label
    label = 1 if "smile" in image_file_name else 0
    
    # reshape from 28x28 to list of 784 values, invert values
    img_data = 255.0 - img_array.reshape(784)
    
    # append label and image data  to test data set
    record = numpy.append(label, img_data)
    training_dataset.append(record)

    # matplotlib.pyplot.imshow(record[1:].reshape(28,28), cmap='Greys', interpolation='None')
    # matplotlib.pyplot.show()
    
    pass

random.shuffle(training_dataset)

# go through all records in the training data set
for record in training_dataset:
    # all_values = record.split(',')
    target_label = record[0]
    # inputs = record[1:]
    inputs = (numpy.asfarray(record[1:]) / 255.0 * 0.99) + 0.01

    # create the target output values (all 0.01, except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(target_label)] = 0.99

    n.train(inputs, targets)
    pass

##################
# Test
##################
print("Testing custom images...")
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

  # data is remaining values
  inputs = numpy.asfarray(img_data)

  # query the network
  outputs = n.query(inputs)

  # the index of the highest value corresponds to the label
  network_guess = numpy.argmax(outputs)
  expression = "smile" if network_guess == 1 else "frown"
  print(bcolors.OKGREEN + "network guess:", expression, bcolors.ENDC)
  matplotlib.pyplot.show()

  print("\n")

  if getContinue() == "B":
     print("\n")
     break
  else:
     print("\n")

