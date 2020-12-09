# %%
print("___  ___          _      _ ")
print("|  \/  |         | |    | |")
print("| .  . | ___   __| | ___| |")
print("| |\/| |/ _ \ / _` |/ _ \ |")
print("| |  | | (_) | (_| |  __/ |")
print("\_|  |_/\___/ \__,_|\___|_|")
print("...........................\n")
print("v2020.03.04\n")


## %% Imports
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps

print("\n")
## %% Image Data Prep
print(" _____                            ______               ")
print("|_   _|                           | ___ \              ")
print("  | | _ __ ___   __ _  __ _  ___  | |_/ / __ ___ _ __  ")
print("  | || '_ ` _ \ / _` |/ _` |/ _ \ |  __/ '__/ _ \ '_ \ ")
print(" _| || | | | | | (_| | (_| |  __/ | |  | | |  __/ |_) |")
print(" \___/_| |_| |_|\__,_|\__, |\___| \_|  |_|  \___| .__/ ")
print("                       __/ |                    | |    ")
print("......................|___/.....................|_|....")
print("\n")
PATH = "../images"
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')

train_fungi_dir = os.path.join(train_dir, 'fungi')  # directory with our training fungi pictures
train_insect_dir = os.path.join(train_dir, 'insect')  # directory with our training insect pictures
validation_fungi_dir = os.path.join(validation_dir, 'fungi')  # directory with our validation fungi pictures
validation_insect_dir = os.path.join(validation_dir, 'insect')  # directory with our validation insect picture
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
num_fungi_tr = len(os.listdir(train_fungi_dir))
num_insect_tr = len(os.listdir(train_insect_dir))
print(f"Fungi Train Images:{num_fungi_tr}")
print(f"Insect Train Images:{num_insect_tr}")
print("\n")
num_fungi_val = len(os.listdir(validation_fungi_dir))
num_insect_val = len(os.listdir(validation_insect_dir))
print(f"Fungi Validation Images:{num_fungi_val}")
print(f"Insect Validation Images:{num_insect_val}")
print("\n")
total_train = num_fungi_tr + num_insect_tr
total_val = num_fungi_val + num_insect_val


print("______                              _                ")
print("| ___ \                            | |               ")
print("| |_/ /_ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___ ")
print("|  __/ _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __|")
print("| | | (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \\")
print("\_|  \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/")
print(".....................................................")
                                                     
batch_size = click.prompt('Enter batch_size (64):', type=int)
epochs = click.prompt('Enter epochs (1000):', type=int)
img_h = click.prompt('Enter img_h (500):', type=int)
img_w = click.prompt('Enter img_w (500):', type=int)
Conv2D_1 =  click.prompt('Enter Conv2D_1 (16, 32, 64, 128):', type=int)
DropOut_1 = click.prompt('Enter DropOut_1 (decimal pct):', type=int)
Conv2D_2 = click.prompt('Enter Conv2D_2 (16, 32, 64, 128):', type=int)
Conv2D_3 = click.prompt('Enter Conv2D_3 (16, 32, 64, 128):', type=int)
DropOut_2 = click.prompt('Enter DropOut_2 (decimal pct):', type=int)


print(f"Total Train:")
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(img_h, img_w),
                                                           class_mode='binary')
print(f"Total Validation:")
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(img_h, img_w),
                                                              class_mode='binary',
                                                              )
print("\n")
print("\n")
## %% Build Model

model = Sequential([
    Conv2D(Conv2D_1, 3, padding='same', activation='relu', 
        input_shape=(img_h, img_w ,3)),
    MaxPooling2D(),
    Dropout(DropOut_1),
    Conv2D(Conv2D_2, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(Conv2D_3, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(DropOut_2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation= 'sigmoid')
])
model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])

print("___  ___          _      _  ______       _ _     _ ")
print("|  \/  |         | |    | | | ___ \     (_) |   | |")
print("| .  . | ___   __| | ___| | | |_/ /_   _ _| | __| |")
print("| |\/| |/ _ \ / _` |/ _ \ | | ___ \ | | | | |/ _` |")
print("| |  | | (_) | (_| |  __/ | | |_/ / |_| | | | (_| |")
print("\_|  |_/\___/ \__,_|\___|_| \____/ \__,_|_|_|\__,_|")
print("...................................................")
print("\n")
model.summary()
print("\n")
print("\n")

## %% Fit Model
print("___  ___          _      _  ______ _ _   _   _             ")
print("|  \/  |         | |    | | |  ___(_) | | | (_)            ")
print("| .  . | ___   __| | ___| | | |_   _| |_| |_ _ _ __   __ _ ")
print("| |\/| |/ _ \ / _` |/ _ \ | |  _| | | __| __| | '_ \ / _` |")
print("| |  | | (_) | (_| |  __/ | | |   | | |_| |_| | | | | (_| |")
print("\_|  |_/\___/ \__,_|\___|_| \_|   |_|\__|\__|_|_| |_|\__, |")
print("                                                      __/ |")
print(".....................................................|___/ ")
print("\n")
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    verbose= 1
)

# # # %% Model History

# # import matplotlib.pyplot as plt

# # acc = history.history['accuracy']
# # val_acc = history.history['val_accuracy']

# # loss=history.history['loss']
# # val_loss=history.history['val_loss']

# # epochs_range = range(epochs)

# # plt.figure(figsize=(8, 8))
# # plt.subplot(1, 2, 1)
# # plt.plot(epochs_range, acc, label='Training Accuracy')
# # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# # plt.legend(loc='lower right')
# # plt.title('Training and Validation Accuracy')

# # plt.subplot(1, 2, 2)
# # plt.plot(epochs_range, loss, label='Training Loss')
# # plt.plot(epochs_range, val_loss, label='Validation Loss')
# # plt.legend(loc='upper right')
# # plt.title('Training and Validation Loss')
# # plt.show()

# # %% Save Model
num = click.prompt('Enter Model Number:', type=int)
model.save("../models/test_model/{num}/")


# # %% 
