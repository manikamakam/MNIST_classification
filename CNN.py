import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential, load_model
from keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from keras.layers import Dense, Dropout, Flatten # core layers

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.datasets import mnist
from keras.utils.vis_utils import plot_model



def summarize_diagnostics(history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss=history.history['loss']
	val_loss=history.history['val_loss']

	epochs_range = range(epochs)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	# plt.show()
	# filename = sys.argv[0].split('/')[-1]
	plt.savefig('graphs.png')
	plt.close()

def define_model():

	model=Sequential()

	#model.add(Lambda(standardize,input_shape=(28,28,1)))    
	model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
	model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
	model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())    
	model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
	    
	model.add(MaxPooling2D(pool_size=(2,2)))
	    
	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dense(512,activation="relu"))
	    
	model.add(Dense(10,activation="softmax"))
	    
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

	return model


# fig, ax = plt.subplots(2,1, figsize=(18, 10))
# ax[0].plot(history.history['loss'], color='b', label="Training loss")
# ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
# legend = ax[0].legend(loc='best', shadow=True)

# ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
# ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
# legend = ax[1].legend(loc='best', shadow=True)


#load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() #everytime loading data won't be so easy :)

# print(X_train.shape)
# print(X_test.shape)

# fig = plt.figure()
# for i in range(9):
#   plt.subplot(3,3,i+1)
#   plt.tight_layout()
#   plt.imshow(X_train[i], cmap='gray', interpolation='none')
#   plt.title("Digit: {}".format(y_train[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()

#Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

print(X_train.shape)
print(X_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"Size of training labels: {y_train.shape}")
print(f"Size of testing labels: {y_test.shape}")

epochs = 50
batch_size = 64

# With data augmentation to prevent overfitting
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
# datagen2.fit(X_test)
train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)
test_gen = datagen.flow(X_test, y_test, batch_size=batch_size)

# print(len(train_gen))
# print(len(test_gen))

model = define_model()
# # print(X_train.shape[0] // batch_size)

history = model.fit_generator(train_gen, 
                              epochs = epochs, 
                              steps_per_epoch = X_train.shape[0] // batch_size,
                              validation_data = test_gen,
                              validation_steps = X_test.shape[0] // batch_size)

model.save('model.h5')
# model = load_model('model.h5')

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

_, acc = model.evaluate_generator(test_gen, steps=X_test.shape[0] // batch_size, verbose=0)
print('Testing Accuracy: %.3f' % (acc * 100.0))

_, train_acc = model.evaluate_generator(train_gen, steps=X_train.shape[0] // batch_size, verbose=0)
print('Training Accuracy: %.3f' % (train_acc * 100.0))


summarize_diagnostics(history)

fig = plt.figure(figsize=(10, 10)) # Set Figure

y_pred = model.predict(X_test) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

Y_pred = np.argmax(y_pred, 1) # Decode Predicted labels
Y_test = np.argmax(y_test, 1) # Decode labels

mat = confusion_matrix(Y_test, Y_pred) # Confusion matrix

# Plot Confusion matrix
sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
plt.xlabel('Predicted Values')
plt.ylabel('True Values');
# plt.show();
plt.savefig('confusion_matrix.png')
plt.close()

X_test__ = X_test.reshape(X_test.shape[0], 28, 28)

fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test__[i], cmap='binary')
    ax.set(title = f"Real Number is {y_test[i].argmax()}\nPredicted Number is {y_pred[i].argmax()}");

plt.savefig('results.png')
plt.close()