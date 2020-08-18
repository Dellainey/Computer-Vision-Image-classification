{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dellainey\\Anaconda3\\lib\\site-packages\\h5py\\__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\n",
      "  from ._conv import register_converters as _register_converters\n",
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from numpy.random import seed\n",
    "seed(501)\n",
    "from tensorflow import set_random_seed\n",
    "set_random_seed(501)\n",
    "from keras import models\n",
    "from keras import layers\n",
    "from keras.layers import Convolution2D\n",
    "from keras.layers import MaxPooling2D\n",
    "from keras.layers import Flatten\n",
    "from keras.layers import Dense\n",
    "from keras.preprocessing import image\n",
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "from keras.utils import to_categorical\n",
    "from keras import regularizers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 7920 images belonging to 6 classes.\n",
      "Found 2640 images belonging to 6 classes.\n",
      "Found 2640 images belonging to 6 classes.\n"
     ]
    }
   ],
   "source": [
    "b_size = 30\n",
    "train_datagen2 = ImageDataGenerator(rescale = 1./255)\n",
    "validation_datagen2 = ImageDataGenerator(rescale = 1./255)\n",
    "test_datagen2 = ImageDataGenerator(rescale = 1./255)\n",
    "\n",
    "train_generator2 = train_datagen2.flow_from_directory(\"C:/Data Analytics/Big_Data/Assignments/project/German traffic sign/GTS/train/\", \n",
    "                                                    target_size=(100,100),\n",
    "                                                    color_mode= \"rgb\",\n",
    "                                                    batch_size = b_size, \n",
    "                                                    shuffle = True,\n",
    "                                                    class_mode = 'categorical')\n",
    "\n",
    "validation_generator2 = validation_datagen2.flow_from_directory(\"C:/Data Analytics/Big_Data/Assignments/project/German traffic sign/GTS/val/\", \n",
    "                                                              target_size=(100,100),\n",
    "                                                              color_mode= \"rgb\",\n",
    "                                                              batch_size = b_size, \n",
    "                                                              shuffle = True,\n",
    "                                                              class_mode = 'categorical')\n",
    "\n",
    "test_generator2 = test_datagen2.flow_from_directory(\"C:/Data Analytics/Big_Data/Assignments/project/German traffic sign/GTS/test/\", \n",
    "                                                  target_size=(100,100),\n",
    "                                                  color_mode= \"rgb\",\n",
    "                                                  batch_size = b_size, \n",
    "                                                  shuffle = False,\n",
    "                                                  class_mode = 'categorical')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from keras.layers import BatchNormalization\n",
    "\n",
    "network = models.Sequential()\n",
    "\n",
    "network.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (100,100,3),kernel_regularizer=regularizers.l2(0.01)))\n",
    "network.add(layers.BatchNormalization())\n",
    "network.add(layers.MaxPooling2D((2,2)))\n",
    "\n",
    "#network.add(layers.Conv2D(32, (3,3), activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))\n",
    "#network.add(layers.BatchNormalization())\n",
    "#network.add(layers.MaxPooling2D((2,2)))\n",
    "\n",
    "network.add(layers.Conv2D(64,(3,3), activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))\n",
    "network.add(layers.BatchNormalization())\n",
    "network.add(layers.MaxPooling2D((2,2)))\n",
    "\n",
    "#network.add(layers.Conv2D(64,(3,3), activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))\n",
    "#network.add(layers.BatchNormalization())\n",
    "#network.add(layers.MaxPooling2D((2,2)))\n",
    "\n",
    "network.add(layers.Conv2D(128,(3,3), activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))\n",
    "network.add(layers.BatchNormalization())\n",
    "network.add(layers.MaxPooling2D((2,2)))\n",
    "\n",
    "#network.add(layers.Conv2D(128,(3,3), activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))\n",
    "#network.add(layers.BatchNormalization())\n",
    "#network.add(layers.MaxPooling2D((2,2)))\n",
    "\n",
    "#network.add(layers.Conv2D(256,(3,3), activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))\n",
    "#network.add(layers.BatchNormalization())\n",
    "#network.add(layers.MaxPooling2D((2,2)))\n",
    "\n",
    "network.add(layers.Flatten())\n",
    "\n",
    "#network.add(layers.Dropout(0.5))\n",
    "\n",
    "network.add(layers.Dense(512, activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))\n",
    "\n",
    "network.add(layers.Dense(6, activation = 'softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 98, 98, 32)        896       \n",
      "_________________________________________________________________\n",
      "batch_normalization_1 (Batch (None, 98, 98, 32)        128       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 49, 49, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 47, 47, 64)        18496     \n",
      "_________________________________________________________________\n",
      "batch_normalization_2 (Batch (None, 47, 47, 64)        256       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 23, 23, 64)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_3 (Conv2D)            (None, 21, 21, 128)       73856     \n",
      "_________________________________________________________________\n",
      "batch_normalization_3 (Batch (None, 21, 21, 128)       512       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_3 (MaxPooling2 (None, 10, 10, 128)       0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 12800)             0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 512)               6554112   \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 6)                 3078      \n",
      "=================================================================\n",
      "Total params: 6,651,334\n",
      "Trainable params: 6,650,886\n",
      "Non-trainable params: 448\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "network.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from keras.optimizers import Adam\n",
    "optimizer  = Adam(lr=1e-4, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)\n",
    "\n",
    "network.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['acc'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/4\n",
      "189/264 [====================>.........] - ETA: 10:50 - loss: 10.8466 - acc: 0.8965"
     ]
    }
   ],
   "source": [
    "history = network.fit_generator(\n",
    "            train_generator2,\n",
    "            steps_per_epoch = np.ceil(7920/b_size),\n",
    "            epochs = 4,\n",
    "            validation_data = validation_generator2,\n",
    "            validation_steps = np.ceil(2640/b_size))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "network.save('C:/Data Analytics/Big_Data/Assignments/modelA.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "acc = history.history['acc']\n",
    "val_acc = history.history['val_acc']\n",
    "\n",
    "epochs = range(1, len(acc)+1)\n",
    "\n",
    "plt.plot(epochs, acc, 'bo', label ='Train acc')\n",
    "plt.plot(epochs, val_acc, 'b', label = 'Validation acc')\n",
    "plt.title('Taining and Validation Accuracy')\n",
    "plt.legend()\n",
    "plt.figure()\n",
    "\n",
    "loss = history.history['loss']\n",
    "val_loss = history.history['val_loss']\n",
    "plt.plot(epochs, loss, 'bo', label = 'Train loss')\n",
    "plt.plot(epochs, val_loss, 'b', label = 'Validation loss')\n",
    "plt.title(\"Training and Validation loss\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "\n",
    "y_p = network.predict_generator(test_generator2, np.ceil(700/b_size))\n",
    "y_pred = np.argmax(y_p, axis = 1)\n",
    "\n",
    "confusion_mat = confusion_matrix(test_generator2.classes, y_pred)\n",
    "print(confusion_mat)\n",
    "\n",
    "score, acc = network.evaluate_generator(test_generator2, np.ceil(700/b_size))\n",
    "print(\"Test set score after augmentation: \", score)\n",
    "print(\"Test set accuracy after augmentation: \", acc)\n",
    "\n",
    "target_names = ['Rose', 'Daisy','Dandelion','Sunflower', 'Tulip']\n",
    "print(classification_report(test_generator2.classes, y_pred, target_names=target_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
