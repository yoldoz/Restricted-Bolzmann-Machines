import numpy as np
from matplotlib.pyplot import imshow, show, figure



def generer_image(RBM, iter_gibbs, nb_images, p, q, image_shape):
    for i in range(1, nb_images):
        v = (np.random.random(p) < 0.5) * 1
        for j in range(iter_gibbs):
            h = (np.random.random(q) < RBM.entree_sortie_RBM(v)) * 1
            v = (np.random.random(p) < RBM.sortie_entree_RBM(h)) * 1
        v = v.reshape(image_shape)
        v = 1 - v
        figure(figsize = (1,1))
        imshow(v, cmap='Greys', )
    show()
