# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:09:00 2018

@author: 919
"""

import glob
import librosa
import numpy as np

from sklearn.preprocessing import LabelBinarizer

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import keras.optimizers

root = "New folder\\"
rootWav = "speechWords\\"

# cnt = 0
mfccList = []
labels = []
maxLength = 32  ## 1808
for i in glob.glob(rootWav + "*"):
    for item in glob.glob(i + "\\*"):
        # cnt += 1
        #print(item)
        # playsound.playsound(item,True)
        wave, sr = librosa.load(item,mono = True, sr = None)
        mfcc = librosa.feature.mfcc(wave, sr = 16000)
        padWidth = maxLength - mfcc.shape[1]
        mfcc = np.pad(mfcc,pad_width = ((0, 0), (0, padWidth)), mode = "constant")
        mfccList.append(mfcc)
        item = str(item).split('\\')
        item = item[1]
        #print(item)
        labels.append(item)
        #if mfcc.shape[1] > maxLength :
        #    maxLength = mfcc.shape[1]

print(maxLength)

import pickle
labels2 = []
for label in labels:
    if label not in labels2:
        labels2.append(label)
pickle.dump(labels2 , open("labels_classes.pkl","wb"))
#labels2 = to_categorical(labels, num_classes=30)
encoder = LabelBinarizer()
transfomed_labels = encoder.fit_transform(labels)

## Convert to np array type
mfccList = np.array(mfccList)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mfccList, transfomed_labels, test_size = 0.15, random_state = 0)

## RESHAPE MFCCs FOR CONVNETS:
X_train = X_train.reshape(X_train.shape[0], 20, 32, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 32, 1) 


def define_model1():
    # feature extractor model
	inputs1 = Input(shape=(20, 32, 1))
	c1 = Conv2D(64, (3, 3), padding='same', activation = 'relu')(inputs1)
	m1 = MaxPooling2D(pool_size = (2, 2))(c1)
	c2 = Conv2D(64, (3, 3), padding='same', activation = 'relu')(m1)
	#m2 = MaxPooling2D(pool_size = (2, 2))(c2)
	c3 = Conv2D(64, (3, 3), padding='same', activation = 'relu')(c2)
	m3 = MaxPooling2D(pool_size = (2, 2))(c3)
	f = Flatten()(m3)
	#fe1 = Dropout(0.5)(inputs1)
	fe1 = Dense(256, activation='relu')(f)
	#fe2 = Dense(128, activation='relu')(fe1)
	
	outputs = Dense(30, activation='softmax')(fe1)
	# tie it together [image, seq] [word]
	model = Model(inputs=inputs1, outputs=outputs)
	adamOpt = keras.optimizers.Adam(lr=1e-3, clipvalue=5., beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer=adamOpt, metrics=['accuracy'])
	# summarize model
	print(model.summary())
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model


model = define_model1()
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))

predicted = model.predict(X_test)
 
 
#np.save('mfccList.npy', mfccList)
#np.save('transformedLabels.npy', transfomed_labels)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 