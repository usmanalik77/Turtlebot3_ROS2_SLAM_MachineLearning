
### SAMPLE FOR TRAINING A DATA

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

current_dir = os.get()

data_path = os.path.join(current_dir, ".npz")

(training_images, training_labels), _ = tf.keras.datasets.load_data(path=data_path)

def reshape_and_normalize(images):
    

    images = images.reshape(-1, 28, 28, 1)
    
    images = images/ 255.0
    
    return images

(training_images, _), _ = tf.keras.datasets.load_data(path=data_path)


training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end (self, epoch, logs=[]):
        if logs.get('accuracy') is not None and logs.get('accuracy') >= 0.8:
            print("\nReached 80% accuracy so cancelling training!")
            
            self.model.stop_training = True

callbacks = myCallback()

def convolutional_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
      
    ])
    

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        
    return model


model = convolutional_model()

model_params = model.count_params()

assert model_params < 1000000, (
    f'Your model has {model_params:,} params. For successful grading, please keep it ' 
    f'under 1,000,000 by reducing the number of units in your Conv2D and/or Dense layers.'
)

callbacks = myCallback()

history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])


print(f"Your model was trained for {len(history.epoch)} epochs")