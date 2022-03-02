from train_RBM import *
from read_images import *
from RBM import *
from generer_image import *
import numpy as np
import matplotlib.pyplot as plt



EPOCHS = 1000
BATCH_SIZES = [1, 5, 10]
EPSILONS = [0.001, 0.01, 0.1]
qs = [50, 100, 150]
num_classes = 3
classes = np.random.choice(range(36), num_classes)
for batch_size in BATCH_SIZES:
    for epsilon in EPSILONS:
        for q in qs:
            data, p, image_shape = lire_alpha_digit(classes)
            rbm = RBM(p, q)
            rbm, accuracies = train_RBM(rbm, EPOCHS, epsilon, batch_size, data)
            fig = plt.figure()
            fig.canvas = plt.FigureCanvasBase(fig) 
            plt.plot(accuracies)
            fig.suptitle('batch_size={} learning_rate={} q={}'.format(batch_size, epsilon, q), fontsize=16)
            plt.xlabel('epochs', fontsize=16)
            plt.ylabel('accuracy', fontsize=16)
            fig.savefig('code\hyperparameters_test/batch_size={} learning_rate={} q={}.jpg'.format(batch_size, epsilon, q))



