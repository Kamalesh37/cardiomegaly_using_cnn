import tensorflow as tf
from keras import models,layers
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator instance for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Loading train and test images from your device directories
train_set = train_datagen.flow_from_directory('C:/data/train/train',
                                              target_size=(224, 224),
                                              batch_size=16,
                                              class_mode='binary')  # Use 'binary' for binary classification

test_set = test_datagen.flow_from_directory('C:/data/test/test',
                                            target_size=(224, 224),
                                            batch_size=16,
                                            class_mode='binary')

# You can now use `train_set` and `test_set` in model training


model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_set, epochs=30, validation_data=test_set)

model.save('C:/model/my_model.h5')
