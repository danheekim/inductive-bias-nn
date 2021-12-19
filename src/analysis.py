"""
Author: Danhee Kim
Date: December 2021
Description: This file contains the necessary code to create and train the models
implemented in this thesis.
It is assumed that CUDA is not available to the user. If you have access to CUDA,
you may execute this file by uncommenting lines with the comment
"uncomment if cuda is available:" above it.
"""
import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('using GPU device:', torch.cuda.get_device_name(0))
else:
    print('using CPU...')
    device = torch.device("cpu")

STITCHED_TRAIN_DATA = data.get_train_data()
STITCHED_TRAIN_LABELS = data.get_train_label()

STITCHED_TEST_DATA = data.get_test_data()
STITCHED_TEST_LABELS = data.get_test_label()

STITCHED_TRAIN_DATA = STITCHED_TRAIN_DATA.to(device, dtype = torch.float32)
STITCHED_TRAIN_LABELS = STITCHED_TRAIN_LABELS.to(device, dtype = torch.long)
STITCHED_TEST_DATA = STITCHED_TEST_DATA.to(device, dtype = torch.float32)
STITCHED_TEST_LABELS = STITCHED_TEST_LABELS.to(device, dtype = torch.long)

mean = STITCHED_TRAIN_DATA.mean()
std = STITCHED_TRAIN_DATA.std()

mean = mean.to(device)
std = std.to(device)

def get_error(scores, labels):
    """
    This method returns the error of our model by calculating the total
    number of correctly classified examples divided by how many examples
    the model has already seen.
    """
    bs = scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()

    return 1-num_matches.float()/bs

#######################################################################
####################### APPROACH ONE ##################################
#######################################################################

# creating our 3 layer fully connected network
class three_layer_net(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_size1, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size, bias=True)
        self.layer2 = nn.Linear(hidden_size, hidden_size1, bias=True)
        self.layer3 = nn.Linear(hidden_size1, output_size, bias = True)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x

my_three_net = three_layer_net(7840, 1000, 1000, 2)
my_three_net = my_three_net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_three_net.parameters(), lr=0.001)
bs = 20

def eval_on_test_set():
  """
  This method is used to evaluate the test error done periodically
  when training our model.
  """
  running_error = 0
  num_batches = 0

  with torch.no_grad():
    for i in range(0, STITCHED_TEST_DATA.size(0), bs):
      minibatch_data = STITCHED_TEST_DATA[i:i+bs]
      minibatch_label = STITCHED_TEST_LABELS[i:i+bs]

      minibatch_data = minibatch_data.to(device)
      minibatch_label = minibatch_label.to(device)

      inputs = minibatch_data.view(bs, 7840)
      float_inputs = (inputs - mean)/std

      scores = my_three_net(float_inputs)

      error = get_error(scores, minibatch_label)

      running_error += error.item()

      num_batches += 1

    total_error = running_error/num_batches
    print('error rate on test set =', total_error*100, 'percent')

### TRAINING ###
for epoch in range(50):
    running_loss = 0
    running_error = 0
    num_batches = 0

    shuffled_indices=torch.randperm(STITCHED_TRAIN_DATA.size(0))

    for count in range(0,STITCHED_TRAIN_DATA.size(0),bs):
        optimizer.zero_grad()

        # creating minibatch:
        indices = shuffled_indices[count:count+bs]
        minibatch_data = STITCHED_TRAIN_DATA[indices]
        minibatch_label= STITCHED_TRAIN_LABELS[indices]

        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)

        inputs = minibatch_data.view(bs,7840)
        inputs_float = (inputs - mean)/std
        inputs_float.requires_grad_()

        inputs_float.to(device)

        scores = my_three_net(inputs_float)

        loss = criterion(scores, minibatch_label)
        loss.backward()
        optimizer.step()

        num_batches+=1
        with torch.no_grad():
            running_loss += loss.item()
            error = get_error( scores , minibatch_label)
            running_error += error.item()

    total_loss = running_loss/num_batches
    total_error = running_error/num_batches

    print('epoch =', epoch, '\t loss =', total_loss , '\t error =', total_error*100 ,'percent')
    eval_on_test_set()
    print('')

#######################################################################
####################### APPROACH TWO ##################################
#######################################################################

# creating our modified mlp-mixer model
class my_mlp(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1, bias = True)
        self.layer2 = nn.Linear(hidden_size1 * 10, hidden_size2, bias = True)
        self.layer3 = nn.Linear(hidden_size2, output_size, bias = True)

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, 10, 784)
        x = self.layer1(x)
        x = F.relu(x)
        x = x.view(bs, -1) # stacking embeddings
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)

        return x

mlp = my_mlp(784, 500, 5000, 2)

mlp = mlp.to(device)
mean = STITCHED_TRAIN_DATA.mean()
std = STITCHED_TRAIN_DATA.std()

mean = mean.to(device)
std = std.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.001)

def eval_on_test_set_mlp():

  running_error = 0
  num_batches = 0

  with torch.no_grad():
    for i in range(0, STITCHED_TEST_DATA.size(0), bs):
      minibatch_data = STITCHED_TEST_DATA[i:i+bs]
      minibatch_label = STITCHED_TEST_LABELS[i:i+bs]

      minibatch_data = minibatch_data.to(device)
      minibatch_label = minibatch_label.to(device)

      inputs = minibatch_data.view(bs, 7840)
      float_inputs = (inputs - mean)/std

      scores = mlp(float_inputs)

      error = get_error(scores, minibatch_label)

      running_error += error.item()

      num_batches += 1

    total_error = running_error/num_batches
    print('error rate on test set =', total_error*100, 'percent')

### TRAINING ###

for epoch in range(150):
    running_loss = 0
    running_error = 0
    num_batches = 0

    shuffled_indices=torch.randperm(STITCHED_TRAIN_DATA.size(0))

    for count in range(0,STITCHED_TRAIN_DATA.size(0),bs):
        optimizer.zero_grad()

        # creating minibatch:
        indices = shuffled_indices[count:count+bs]
        minibatch_data = STITCHED_TRAIN_DATA[indices]
        minibatch_label = STITCHED_TRAIN_LABELS[indices]

        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)

        inputs = minibatch_data.view(-1, 7840)
        inputs_float = (inputs - mean)/std
        inputs_float.requires_grad_()

        scores = mlp(inputs_float)

        loss = criterion(scores, minibatch_label)
        loss.backward()
        optimizer.step()

        num_batches+=1

        with torch.no_grad():

            running_loss += loss.item()

            error = get_error(scores , minibatch_label)
            running_error += error.item()

    total_loss = running_loss/num_batches
    total_error = running_error/num_batches

    print('epoch =',epoch, '\t loss =', total_loss , '\t error =', total_error*100 ,'percent')
    eval_on_test_set_mlp()
    print('')
