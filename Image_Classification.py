import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 
 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
 
 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
 
 
model.fit(train_images, train_labels, epochs=10)
 
 
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
 

def show(i):
  plt.imshow(test_images[i],cmap=plt.cm.binary)
  
def predict(i):
  n=np.argmax(predictions[i])
  percentage=100*np.max(predictions[i])
  predicted_label=class_names[n]
  print("("+str(n)+")",predicted_label,"with",percentage,"% Probablity")
