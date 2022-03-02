from train_RBM import *
from read_images import *
from RBM import *
from generer_image import *
import matplotlib.pyplot as plt


EPOCHS = 1300
BATCH_SIZE = 20
EPSILON = 0.1
num_classes = 36
q = 200

NB_IMAGES = 6
ITER_GIBBS = 1000

classes = [i for i in range(num_classes)]
#classes = [10, 11, 12] 
data, p, image_shape = lire_alpha_digit(classes)
print(image_shape)
RBM = RBM(p, q)
RBM, accuracies = train_RBM(RBM, EPOCHS, EPSILON, BATCH_SIZE, data)
generer_image(RBM, ITER_GIBBS , NB_IMAGES, p, q, image_shape)

