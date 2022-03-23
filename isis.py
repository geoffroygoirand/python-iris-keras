
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam


model = Sequential()

model.add(Dense(15,input_shape=(4,),activation='tanh'))
model.add(Dense(10,activation='tanh'))
model.add(Dense(6,activation='tanh'))
model.add(Dense(3,activation='softmax'))

model.compile(Adam(lr=0.06),'categorical_crossentropy',metrics=['accuracy'])

model.summary()
