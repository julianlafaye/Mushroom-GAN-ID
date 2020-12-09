# %% Model Evaluation
print("___  ___          _      _   _____           _ ")
print("|  \/  |         | |    | | |  ___|         | |")
print("| .  . | ___   __| | ___| | | |____   ____ _| |")
print("| |\/| |/ _ \ / _` |/ _ \ | |  __\ \ / / _` | |")
print("| |  | | (_) | (_| |  __/ | | |___\ V / (_| | |")
print("\_|  |_/\___/ \__,_|\___|_| \____/ \_/ \__,_|_|")
print("...............................................")
print("\n")

## %% Imports
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import numpy as np
import matplotlib.pyplot as plt
import os
import click
print("\n")
print("...................................................")
## %% Grab Test Images
PATH = "../images"
test_dir = os.path.join(PATH, 'test')

test_fungi_dir = os.path.join(test_dir, 'fungi')  # directory with our test fungi pictures
test_insect_dir = os.path.join(test_dir, 'insect')  # directory with our test insect pictures
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our test data

batch_size = 128
epochs = 5
img_h = 32
img_w = 32
print("Testing Set:")
test_data_gen = test_image_generator.flow_from_directory(batch_size = batch_size,
                                                              directory=test_dir,
                                                              target_size=(img_h, img_w),
                                                              class_mode='binary')
print("....................................................")
print("\n")                                      
print("     ,---,.  ,---,    ,---,.  ,---,    ,---,.  ,---,")
print("   ,'  .' |,'  .'|  ,'  .' |,'  .'|  ,'  .' |,'  .'|")
print(" ,---.'  ,---.'  |,---.'  ,---.'  |,---.'  ,---.'  |")
print(" |   |   |   |   ;|   |   |   |   ;|   |   |   |   ;")
print(" :   :  .:   :  .':   :  .:   :  .':   :  .:   :  .'")
print(" :   |.' :   |.'  :   |.' :   |.'  :   |.' :   |.'  ")
print(" `---'   `---'    `---'   `---'    `---'   `---'    ")
print("\n")

num = click.prompt('Enter Model Number', type=int)
model = tf.keras.models.load_model(f'../models/test_model/{num}/')

## %% Generate Predictions

print(" ______             _ _      _   _             ")
print(" | ___ \           | (_)    | | (_)            ")
print(" | |_/ / __ ___  __| |_  ___| |_ _ _ __   __ _ ")
print(" |  __/ '__/ _ \/ _` | |/ __| __| | '_ \ / _` |")
print(" | |  | | |  __/ (_| | | (__| |_| | | | | (_| |")
print(" \_|  |_|  \___|\__,_|_|\___|\__|_|_| |_|\__, |")
print("                                          __/ |")
print("                                         |___/ ")

pred = model.predict_classes(test_data_gen, batch_size=None)

pred = [0 if p < 0.5 else 1 for p in pred]
label = list(test_data_gen.labels)
filename = list(test_data_gen.filenames)
table = [pred, label, filename]

print(".                                                     ______                 ")
print("                                                      |  _  \                ")
print("                                                      | | | |___  _ __   ___ ")
print("                                                      | | | / _ \| '_ \ / _ \\")
print(" _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _| |/ / (_) | | | |  __/")
print("(_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_)___/ \___/|_| |_|\___|")
                                                                 
                                                                 

print("\n")                                      
print("     ,---,.  ,---,    ,---,.  ,---,    ,---,.  ,---,")
print("   ,'  .' |,'  .'|  ,'  .' |,'  .'|  ,'  .' |,'  .'|")
print(" ,---.'  ,---.'  |,---.'  ,---.'  |,---.'  ,---.'  |")
print(" |   |   |   |   ;|   |   |   |   ;|   |   |   |   ;")
print(" :   :  .:   :  .':   :  .:   :  .':   :  .:   :  .'")
print(" :   |.' :   |.'  :   |.' :   |.'  :   |.' :   |.'  ")
print(" `---'   `---'    `---'   `---'    `---'   `---'    ")
print("\n")

## %% Testing Evaluation
print("Evaluating....")
print("...................................................")
print("\n")
df = pd.DataFrame(table)
df = df.transpose()
df.columns = ["predictions", 'labels', "filenames"]
df['guesses'] = df['predictions'] == df['labels']
guess_bias = df[df["guesses"] == False]["labels"].mean()
print("False Guess Bias:")
if guess_bias < 0.5:
    print("fungi")
    print("by")
    print(0.5 - guess_bias)
elif guess_bias > 0.5:
    print("insect")
    print("by")
    print(guess_bias - 0.5)
else:
    print("oh shit")
    print("50/50")
print("\n")
falses = df[df["guesses"] == False]['filenames']
trues = df[df["guesses"] == True]['filenames']
falses = falses.str.replace("\\", "/")
print("Accuracy:")
print(len(trues)/len(df))
print("\n")
if click.confirm('Print list of False Images?', default=True):
    print('\n')
    print("False Images")
    print("+------------+")
    print("+------------+")
    for false in falses:
        print(false)

# %%
# 'https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/'
# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
# load the model
model = VGG16()
# redefine model to output right after the first hidden layer
ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = load_img('../images/insect/xdc1v7h1uqh41.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot the output from each block
square = 8
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()

# %%
