import numpy
from matplotlib import pyplot as plt
import scipy.special
import scipy.misc
import imageio
from PIL import Image, ImageFilter
def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva

class neuralNetwork:
    #initialize the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes #set input nodes
        self.hnodes = hiddennodes #set hidden nodes
        self.onodes = outputnodes #set output nodes
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))
        #learning rate
        self.lr = learningrate
        # sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def train(self,inputs_list,targets_list):
        #convert input & target lists to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        #above moves through the network and get value
        #check the error deriving from above
        output_errors = targets - final_outputs
        # the back propagation of errors : errorshidden = (weightsT)*hidden_output * errors_output
        # hidden layer error is the output_errors, split by weights, recombined
        #at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        #change weights for hidden to output accordingly
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update the weights for the links between the hidden and output layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    def query(self,inputs_list):
        #convert inputs to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculate signals into hidden layer, w_ih * inputs
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer, sigmoid on hidden_inputs
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer, w_ho*hidden_outputs
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer, sigmoid on final_inputs
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass
#number of input,hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
#learning rate
learning_rate = 0.3
#Create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#Load training data
data_file = open("mnist_train.csv","r")
data_list = data_file.readlines()
data_file.close()
#train neural network
#go through all the numbers in data set
for record in data_list:
    #split record by commas
    all_values = record.split(',')
    #scale the inputs, transform each pixel into a corressponding value between 0.01 and 0.99
    inputs = ((numpy.asfarray(all_values[1:])/255.0) * 0.99)+0.01
    #create the target values we want
    #our target values cannot be zero, but we can set 0.01 as essentially as our 0.
    #we make a matrix of 0.01's
    targets = numpy.zeros(output_nodes) + 0.01
    #get the value of our intended number and give it the highest value
    targets[int(all_values[0])] = 0.99
    #train the neural network
    n.train(inputs,targets)
    print('loading')
    pass

# load the mnist test data CSV file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass

    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

#guess my hand writing
#label and get file name
label = 3
img_data = imageprepare('./3.png')
record = numpy.append(label,img_data)
our_own_dataset = []
our_own_dataset.append(record)

#plot image
plt.imshow(our_own_dataset[0][1:].reshape(28,28), cmap='Greys', interpolation='None')

# real answer
correct_answer = our_own_dataset[0][0]
#data
inputs = our_own_dataset[0][1:]
#query the network
outputs = n.query(inputs)
print(outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print("network says ", label)
# append correct or incorrect to list
print(outputs[correct_label],outputs[label])
if (label == correct_answer):
    print ("match!")
else:
    print ("no match!")
    pass
plt.show()
# let's write some numbers
# will take some time , but the network should say the picture is an image of a 7
#awesome! , let's try 8
# bear with me!!!
# 8 with a probability of 52% !!!
