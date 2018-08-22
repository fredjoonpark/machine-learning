import numpy as np
from sklearn.cluster import KMeans

class ImageQuantizer:

    def __init__(self, b):
        self.b = b # number of bits per pixel

    def quantize(self, img):
        """
        Quantizes an image into 2^b clusters

        Parameters
        ----------
        img : a (H,W,3) numpy array

        Returns
        -------
        quantized_img : a (H,W) numpy array containing cluster indices

        Stores
        ------
        colours : a (2^b, 3) numpy array, each row is a colour
        """
        H, W, _ = img.shape
        model = KMeans(n_clusters=2**self.b, n_init=3)

        raw_x=[]
        pix = img.reshape((-1,3))
        for i in range(H):
            for j in range(W):
                raw_x.append(img[i,j,:])
        x = np.array(raw_x)
        model.fit(x)
        raw_quantized_img = model.predict(x)
        
        quantized_img = raw_quantized_img.reshape((H,W))

        self.colours = model.cluster_centers_

        return quantized_img


    def dequantize(self, quantized_img):
        H, W = quantized_img.shape
        quantized_flat = quantized_img.flatten() 

        img = np.zeros((H*W,3), dtype='uint8')
        for i in range(H*W):
            img[i] = self.colours[quantized_flat[i]]
        img = img.reshape((H,W,3))
        
        return img
