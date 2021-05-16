import os
import zipfile
import random
import tensorflow as tf
import wget
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import pickle

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.system('export TF_CUDNN_RESET_RND_GEN_STATE=1')
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
'''
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
path = "C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/"
wget.download(url, out=path)
'''
path = "C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/"

'''
zip_ref = zipfile.ZipFile(os.path.join(path, "cats-and-dogs.zip"), 'r')
zip_ref.extractall(os.path.join(path,"cats-and-dogs"))
zip_ref.close()
'''

dataset = os.path.join(path,"cats-and-dogs")

print(len(os.listdir(os.path.join(dataset,"PetImages/Cat/"))))
print(len(os.listdir(os.path.join(dataset,"PetImages/Dog/"))))

# Creating training and test directories
to_create = [
    'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs',
    'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/training',
    'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/testing',
    'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/training/cats',
    'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/training/dogs',
    'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/testing/cats',
    'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/testing/dogs'
]

'''
for directory in to_create:
	try:
		os.mkdir(directory)
		print(directory, 'created')
	except:
		print(directory, 'failed')
'''

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    all_files = []
    
    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + file_name

        if os.path.getsize(file_path):
            all_files.append(file_name)
        else:
            print('{} is zero length, so ignoring'.format(file_name))
    
    n_files = len(all_files)
    split_point = int(n_files * SPLIT_SIZE)
    
    shuffled = random.sample(all_files, n_files)
    
    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]
    
    for file_name in train_set:
        copyfile(SOURCE + file_name, TRAINING + file_name)
        
    for file_name in test_set:
        copyfile(SOURCE + file_name, TESTING + file_name)

CAT_SOURCE_DIR = "C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-and-dogs/PetImages/Cat/"
TRAINING_CATS_DIR = 'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/training/cats/'
TESTING_CATS_DIR = 'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/testing/cats/'
DOG_SOURCE_DIR = "C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-and-dogs/PetImages/Dog/"
TRAINING_DOGS_DIR = 'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/training/dogs/'
TESTING_DOGS_DIR = 'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/testing/dogs/'

split_size = .9
'''
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
'''

print(len(os.listdir('C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/training/cats/')))
print(len(os.listdir('C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/training/dogs/')))
print(len(os.listdir('C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/testing/cats/')))
print(len(os.listdir('C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/testing/dogs/')))


# ------ Define a Keras model to classify CATS vs DOGS ---------
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(32, (3,3), input_shape=(150,150,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

model.summary()

TRAINING_DIR = 'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/training/'
train_datagen = ImageDataGenerator(
	rescale=1/255,
	rotation_range=40,
	width_shift_range=.2,
	height_shift_range=.2,
	shear_range=.2,
	zoom_range=.2,
	horizontal_flip=True,
	fill_mode='nearest'
	)

train_generator = train_datagen.flow_from_directory(
	TRAINING_DIR,
	batch_size=64,
	class_mode='binary',
	target_size=(150,150)
	)	

VALIDATION_DIR = 'C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/cats-v-dogs/testing/'
validation_datagen = ImageDataGenerator(
	rescale=1/255,
	rotation_range=40,
	width_shift_range=.2,
	height_shift_range=.2,
	shear_range=.2,
	zoom_range=.2,
	horizontal_flip=True,
	fill_mode='nearest'
	)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	batch_size=64,
	class_mode='binary',
	target_size=(150,150)
	)

# --------------Now train the model----------
history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=2,
            verbose=1)

# Save the model for future use
model.save('C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/model.h5')

'''
# PLOT LOSS AND ACCURACY
# %matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')

# Load the saved model
# loaded_model = tf.keras.models.load_model(dataset)
'''
