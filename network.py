from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
import numpy as np
from keras.constraints import maxnorm
with open("content.bin","rb") as f:
   content=np.load(f)
X_train=content.astype(np.float32)
with open("result.bin","rb") as f:
    result=np.load(f)
Y_train=np.expand_dims(result, axis=1).astype(np.float32)
model = Sequential()
model.add(Convolution2D(7, 4, 4, border_mode='valid', input_shape=(3, 134,70),W_constraint =maxnorm(2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(3, 4, 4,W_constraint =maxnorm(2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256,W_constraint =maxnorm(1)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1,W_constraint =maxnorm(1)))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer="rmsprop")
model.fit(X_train, Y_train, batch_size=128, nb_epoch=100,verbose=1)
model.save_weights("hehe0.h5",overwrite=True)
json_string=model.to_json()
open('model.json', 'w').write(json_string)