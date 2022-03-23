
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam

iris = datasets.load_iris()
X = iris.data
Y = iris.target

model = Sequential()

model.add(Dense(15,input_shape=(4,),activation='tanh'))
model.add(Dense(10,activation='tanh'))
model.add(Dense(6,activation='tanh'))
model.add(Dense(3,activation='softmax'))

model.fit(X, Y, epochs=5)

history = model.compile(Adam(lr=0.06),'categorical_crossentropy',metrics=['accuracy'])

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.summary()
