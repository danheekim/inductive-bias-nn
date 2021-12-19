"""
Author: Danhee Kim
Date: December 2021
Description: This file contains the necessary code to generate the training
and testing data for the thesis: Exploring Inductive Bias in Neural Networks.
The data consists of 10 randomly stitched together "4", "7", and "9" digits
from the MNIST dataset. The labels associated with each data item are a binary
classification which indicates whether or not there are 3 or more consecutive
digits in a row.
"""
import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# getting MNIST data:
mnist_train = datasets.MNIST(root = './data', train = True, download = True, transform = None)
mnist_test = datasets.MNIST(root = './data', train = False, download = True, transform = None)

train_data = mnist_train.data
train_label = mnist_train.targets
test_data = mnist_test.data
test_label = mnist_test.targets

def get_label(x):
	"""
	This method returns the desired label of the given data item used in our
	experiment.

	Input: x is our data
	Returns 0 or 1 depending on if 3 or more of the same digits are
	repeated or not.
		0: no repeats
		1: repeat exists
	"""
	current_label = None
	count = 0
	for label in x:
		if count >= 3:
			return 1
		if current_label == label:
			count += 1
		else:
			current_label = label
			count = 1
	return 0

# extracting the 4's, 7's, and 9's from MNIST:
my_train_of_fours = []
my_train_of_sevens = []
my_train_of_nines = []

for i in range(len(train_label)):
    if train_label[i] == 4:
        my_train_of_fours.append(i)

for i in range(len(train_label)):
    if train_label[i] == 7:
        my_train_of_sevens.append(i)

for i in range(len(train_label)):
    if train_label[i] == 9:
        my_train_of_nines.append(i)

train_of_fours = train_data[my_train_of_fours]
train_of_sevens = train_data[my_train_of_sevens]
train_of_nines = train_data[my_train_of_nines]

# dictionary to access how many 4's, 7's, and 9's exist in MNIST
train_num_size_map = {
	4: train_of_fours.size(0),
	7: train_of_sevens.size(0),
	9: train_of_nines.size(0),
}

# gets the minimum size of 4's, 7's, and 9's -- used to avoid oversampling
min_train_size = train_num_size_map[min(train_num_size_map)]

# gets random indices of each digit of the minimum size
random_idx_4_list = random.sample(range(train_num_size_map[4]), min_train_size)
random_idx_7_list = random.sample(range(train_num_size_map[7]), min_train_size)
random_idx_9_list = random.sample(range(train_num_size_map[9]), min_train_size)

# now we have a random sample of 4's, 7's, and 9's of the same size (and their MNIST labels)
train_data_4s = train_of_fours[random_idx_4_list]
train_data_7s = train_of_sevens[random_idx_7_list]
train_data_9s = train_of_nines[random_idx_9_list]

my_labels_4 = torch.from_numpy(np.full((train_data_4s.size(0)), 4))
my_labels_7 = torch.from_numpy(np.full((train_data_7s.size(0)), 7))
my_labels_9 = torch.from_numpy(np.full((train_data_9s.size(0)), 9))

# stacking our 4's, 7's, and 9's
training_data = torch.cat([train_data_4s, train_data_7s, train_data_9s], dim = 0)
training_labels = torch.cat([my_labels_4, my_labels_7, my_labels_9])

############################

# extracting 4's, 7's, and 9's from the MNIST test dataset:
my_test_data = []
for i in range(len(test_label)):
	if test_label[i] == 4 or test_label[i] == 7 or test_label[i] == 9:
		my_test_data.append(i)
testing_data = test_data[my_test_data]
testing_labels = test_label[my_test_data]

############################

# getting 10 random indices from my train and test data (which just consists of 4's, 7's, and 9's)
random_train_indices = torch.randint(training_data.size()[0], (10,))
random_test_indices = torch.randint(testing_data.size()[0], (10,))

# stitched_train_labels gets the labels to the corresponding random_indices generated
    # this should get overwritten to either be 0 or 1 (related to if we get 3 consecutive 4's (or 9's) in a row)
# getting stitched
stitched_train_data = training_data[random_train_indices]
stitched_train_labels = training_labels[random_train_indices]

stitched_test_data = testing_labels[random_test_indices]
stitched_test_labels = testing_labels[random_test_indices]

# CREATING OUR STITCHED TRAIN DATA + LABELS — this is the data used in our models!!

TRAIN_DATA_SIZE = 200000
my_stitched_train_data = []
my_stitched_train_labels = []

for i in range(TRAIN_DATA_SIZE):
    random_train_indices = torch.randint(training_data.size()[0], (10,))
    stitched_train_pre_labels = training_labels[random_train_indices]
    stitched_train_partial_data = training_data[random_train_indices]

    my_stitched_train_labels.append(stitched_train_pre_labels)
    my_stitched_train_data.append(stitched_train_partial_data)

STITCHED_TRAIN_DATA = torch.stack(my_stitched_train_data)
STITCHED_TRAIN_DATA = STITCHED_TRAIN_DATA.view(TRAIN_DATA_SIZE, 280, 28)

stitch_train_label = torch.stack(my_stitched_train_labels) # 500 x 10

revised_train_labels = []
for i in range(0, stitch_train_label.size(0)):
    revised_train_labels.append(get_label(stitch_train_label[i]))

STITCHED_TRAIN_LABELS = torch.tensor(revised_train_labels) # should be 500


#############

# CREATING OUR STITCHED TEST DATA + LABELS

TEST_DATA_SIZE = 20000
my_stitched_test_data = []
my_stitched_test_labels = []

for i in range(TEST_DATA_SIZE):
    random_test_indices = torch.randint(testing_data.size()[0], (10,))
    stitched_test_pre_labels = testing_labels[random_test_indices]
    stitched_test_partial_data = testing_data[random_test_indices]

    my_stitched_test_labels.append(stitched_test_pre_labels)
    my_stitched_test_data.append(stitched_test_partial_data)

STITCHED_TEST_DATA = torch.stack(my_stitched_test_data)
STITCHED_TEST_DATA = STITCHED_TEST_DATA.view(TEST_DATA_SIZE, 280, 28)

stitch_test_label = torch.stack(my_stitched_test_labels) # 500 x 10

revised_test_labels = []
for i in range(0, stitch_test_label.size(0)):
    revised_test_labels.append(get_label(stitch_test_label[i]))

STITCHED_TEST_LABELS = torch.tensor(revised_test_labels) # should be 500 # should be 500

def get_train_data():
	return STITCHED_TRAIN_DATA

def get_train_label():
	return STITCHED_TRAIN_LABELS

def get_test_data():
	return STITCHED_TEST_DATA

def get_test_label():
	return STITCHED_TEST_LABELS

print("##########")
print("Train data size: " + str(STITCHED_TRAIN_DATA.size(0)))
print("Ratio of label 0 to label 1 in train data: " + str(1-(torch.sum(STITCHED_TRAIN_LABELS)/TRAIN_DATA_SIZE)))
print("	")
print("Test data size: " + str(STITCHED_TEST_DATA.size(0)))
print("Ratio of label 0 to label 1 in test data: " + str(1-(torch.sum(STITCHED_TEST_LABELS)/TEST_DATA_SIZE)))
print("##########")
