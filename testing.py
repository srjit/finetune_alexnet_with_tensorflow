import os
import numpy as np
import cv2

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)


current_dir = os.getcwd()
image_dir = os.path.join(current_dir, 'images')

img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]


images = []
images = [cv2.imread(image) for image in img_files]




from alexnet import AlexNet
from caffe_classes import class_names
import tensorflow as tf

#input layer image dimensions
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
# dropout rate
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, 1000, [])
score = model.fc8
softmax = tf.nn.softmax(score)




# Now let's run it
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)

    for i, image in enumerate(images):
        
        # Convert image to float32 and resize to (227x227)
        img = cv2.resize(image.astype(np.float32), (227,227))
        
        # Subtract the ImageNet mean
        img -= imagenet_mean
        
        # Reshape as needed to feed into model
        img = img.reshape((1,227,227,3))

        probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})

        # class names with highest probability
        class_name = class_names[np.argmax(probs)]

        print(class_name)


print("Original files", img_files)        
