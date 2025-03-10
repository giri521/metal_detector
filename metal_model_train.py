# metal_model_train.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model

# Path
data_dir = "dataset"
img_size = 150

# Image augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(data_dir,
                                    target_size=(img_size, img_size),
                                    batch_size=32,
                                    class_mode='categorical',
                                    subset='training')

val = datagen.flow_from_directory(data_dir,
                                  target_size=(img_size, img_size),
                                  batch_size=32,
                                  class_mode='categorical',
                                  subset='validation')

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=10)

# Save model
model.save("metal_detector_model.h5")
