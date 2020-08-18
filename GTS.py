#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
from numpy.random import seed
seed(501)
from tensorflow import set_random_seed
set_random_seed(501)
from keras import models
from keras import layers
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import regularizers


# In[2]:


b_size = 30
train_datagen2 = ImageDataGenerator(rescale = 1./255)
validation_datagen2 = ImageDataGenerator(rescale = 1./255)
test_datagen2 = ImageDataGenerator(rescale = 1./255)

train_generator2 = train_datagen2.flow_from_directory("C:/Data Analytics/Big_Data/Assignments/project/German traffic sign/GTS/train/", 
                                                    target_size=(100,100),
                                                    color_mode= "rgb",
                                                    batch_size = b_size, 
                                                    shuffle = True,
                                                    class_mode = 'categorical')

validation_generator2 = validation_datagen2.flow_from_directory("C:/Data Analytics/Big_Data/Assignments/project/German traffic sign/GTS/val/", 
                                                              target_size=(100,100),
                                                              color_mode= "rgb",
                                                              batch_size = b_size, 
                                                              shuffle = True,
                                                              class_mode = 'categorical')

test_generator2 = test_datagen2.flow_from_directory("C:/Data Analytics/Big_Data/Assignments/project/German traffic sign/GTS/test/", 
                                                  target_size=(100,100),
                                                  color_mode= "rgb",
                                                  batch_size = b_size, 
                                                  shuffle = False,
                                                  class_mode = 'categorical')


# In[3]:


from keras.layers import BatchNormalization

network = models.Sequential()

network.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (100,100,3),kernel_regularizer=regularizers.l2(0.01)))
network.add(layers.BatchNormalization())
network.add(layers.MaxPooling2D((2,2)))

network.add(layers.Conv2D(64,(3,3), activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))
network.add(layers.BatchNormalization())
network.add(layers.MaxPooling2D((2,2)))

network.add(layers.Conv2D(128,(3,3), activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))
network.add(layers.BatchNormalization())
network.add(layers.MaxPooling2D((2,2)))

network.add(layers.Flatten())


network.add(layers.Dense(512, activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))

network.add(layers.Dense(6, activation = 'softmax'))


# In[4]:


network.summary()


# In[ ]:


from keras.optimizers import Adam
optimizer  = Adam(lr=1e-4, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

network.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['acc'])


# In[ ]:


history = network.fit_generator(
            train_generator2,
            steps_per_epoch = np.ceil(7920/b_size),
            epochs = 4,
            validation_data = validation_generator2,
            validation_steps = np.ceil(2640/b_size))


# In[ ]:


network.save('C:/Data Analytics/Big_Data/Assignments/modelA.h5')


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label ='Train acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Taining and Validation Accuracy')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label = 'Train loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title("Training and Validation loss")
plt.legend()
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report


y_p = network.predict_generator(test_generator2, np.ceil(700/b_size))
y_pred = np.argmax(y_p, axis = 1)

confusion_mat = confusion_matrix(test_generator2.classes, y_pred)
print(confusion_mat)

score, acc = network.evaluate_generator(test_generator2, np.ceil(700/b_size))
print("Test set score after augmentation: ", score)
print("Test set accuracy after augmentation: ", acc)

target_names = ['Rose', 'Daisy','Dandelion','Sunflower', 'Tulip']
print(classification_report(test_generator2.classes, y_pred, target_names=target_names))


# In[ ]:




