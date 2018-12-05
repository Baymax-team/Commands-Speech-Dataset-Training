# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:34:24 2018

@author: 919
"""

from keras.models import load_model
import librosa
import numpy as np
import pickle

classes = list(pickle.load(open('labels_classes.pkl', 'rb')))
wavSound = '01bb6a2a_nohash_3.wav'
wavSound1 = '2e0d80f7_nohash_1.wav'
wavS3 = 'record0.wav'
wavS4 = 'record4.wav'
model = load_model('weights_2/model-ep016-loss0.077-val_loss0.929.h5')
maxLength = 32

wave, sr = librosa.load(wavS3,mono = True, sr = None)
mfcc = librosa.feature.mfcc(wave, sr = 16000)
padWidth = maxLength - mfcc.shape[1]
mfcc = np.pad(mfcc,pad_width = ((0, 0), (0, padWidth)), mode = "constant")

mfcc1 = np.reshape(np.array(mfcc), (1, 20, 32, 1))

pred = model.predict(mfcc1)
pred = int(pred.argmax(axis=-1))
print(classes[pred])