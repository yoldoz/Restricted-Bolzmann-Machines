import scipy.io
import numpy as np



def lire_alpha_digit(vecteur, dataset_name = None):
    if dataset_name is None:
        mat = scipy.io.loadmat(r"code\binaryalphadigs.mat")
        image_shape = mat['dat'][0,:][0].shape
        n = int(len(vecteur)*39)
        p = int(image_shape[0]*image_shape[1])
        data = np.zeros((n,p))
        i = 0
        for v in vecteur:
            data_class = mat['dat'][v,:]
            for l in data_class:
                data[i,:]= np.ravel(l)
                i+=1 
    elif dataset_name == 'MNIST':
        mat = scipy.io.loadmat(r'code\mnist_all.mat')
        image_shape = (28,28)
        classes = {i : "train"+str(i) for i in vecteur}
        n = 0
        for im_class in classes.values():
            n += mat[im_class].shape[0]
        p = image_shape[0]*image_shape[1]
        data = np.zeros((n,p))
        i = 0
        for v in vecteur:
            data_class = classes[v]
            for image in mat[data_class]:
                data[i,:] = (image>127)*1
                i+=1   
    return data, p, image_shape