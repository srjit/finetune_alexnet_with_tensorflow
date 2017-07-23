import os

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


train_image_path = "/home/sree/code/finetune_alexnet_with_tensorflow/train"

label_keys = {
    'cat' : 0,
    'dog' : 1
    }

img_files = [os.path.join(train_image_path, f) for f in os.listdir(train_image_path) if f.endswith('.jpg')]


def cat_or_dog(inp):
    output = inp
    if ('cat') in inp:
        output += " " + str(label_keys['cat'])
    elif ('dog') in inp:
        output += " " + str(label_keys['dog'])
        
    else:
        output += " " + str(2)
    return output



tagged_array = [cat_or_dog(filename) for filename in img_files]

train = tagged_array[0:17501]
test = tagged_array[17501:21071]
validation = tagged_array[21701:-1]

with open("train.txt", "w") as f:
    for line in train:
        f.write(line + "\n")

with open("test.txt", "w") as f:
    for line in test:
        f.write(line + "\n")

with open("validation.txt", "w") as f:
    for line in validation:
        f.write(line + "\n")
        
        
