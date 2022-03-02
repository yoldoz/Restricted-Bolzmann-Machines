from train_RBM import *
from read_images import *
from RBM import *
from generer_image import *
import matplotlib.pyplot as plt

EPOCHS = 300
BATCH_SIZE = 5
EPSILON = 0.1
num_classes = 10
q = 100

NB_IMAGES = 20
ITER_GIBBS = 1000

#classes = [i for i in range(num_classes)]
classes = [2, 3]
data, p, image_shape = lire_alpha_digit(classes,dataset_name="MNIST")
RBM = RBM(p, q)
RBM, accuracies = train_RBM(RBM, EPOCHS, EPSILON, BATCH_SIZE, data)

generer_image(RBM, ITER_GIBBS , NB_IMAGES, p, q, image_shape)


fig = plt.figure()
fig.canvas = plt.FigureCanvasBase(fig) 
plt.plot(accuracies)
plt.xlabel('epochs', fontsize=16)
plt.ylabel('accuracy', fontsize=16)
fig.savefig('mnist.png')
