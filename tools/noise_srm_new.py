import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def SRM_Trans(img):
    kernal = np.array([[[0, 0, 0, 0, 0], [0, -1 / 4, 2 / 4, -1 / 4, 0], [0, 2 / 4, -4 / 4, 2 / 4, 0],
                        [0, -1 / 4, 2 / 4, -1 / 4, 0], [0, 0, 0, 0, 0]],
                       [[-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12], [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
                        [-2 / 12, 8 / 12, -12 / 12, 8 / 12, -2 / 12], [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
                        [-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12]],
                       [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1 / 2, -2 / 2, 1 / 2, 0], [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]])

    dst_r = (cv2.filter2D(img[:, :, 0], -1, kernal[0, :, :]) + cv2.filter2D(img[:, :, 1], -1,
                                                                            kernal[0, :, :]) + cv2.filter2D(
        img[:, :, 2], -1, kernal[0, :, :])) / 3
    dst_g = (cv2.filter2D(img[:, :, 0], -1, kernal[1, :, :]) + cv2.filter2D(img[:, :, 1], -1,
                                                                            kernal[1, :, :]) + cv2.filter2D(
        img[:, :, 2], -1, kernal[1, :, :])) / 3
    dst_b = (cv2.filter2D(img[:, :, 0], -1, kernal[2, :, :]) + cv2.filter2D(img[:, :, 1], -1,
                                                                            kernal[2, :, :]) + cv2.filter2D(
        img[:, :, 2], -1, kernal[2, :, :])) / 3
    dst = cv2.merge([dst_r, dst_g, dst_b])

    ress = np.array(dst, dtype=float)
    return ress

if __name__ == '__main__':
    img = Image.open('5041a57c9128aaed13a60cc43084de70.jpg').convert('RGB')
    img = np.asarray(img)

    kernal = np.array([[[0,0,0,0,0],[0,-1/4,2/4,-1/4,0],[0,2/4,-4/4,2/4,0],[0,-1/4,2/4,-1/4,0],[0,0,0,0,0]],
    [[-1/12,2/12,-2/12,2/12,-1/12],[2/12,-6/12,8/12,-6/12,2/12],[-2/12,8/12,-12/12,8/12,-2/12],[2/12,-6/12,8/12,-6/12,2/12],[-1/12,2/12,-2/12,2/12,-1/12]],
    [[0,0,0,0,0],[0,0,0,0,0],[0,1/2,-2/2,1/2,0],[0,0,0,0,0],[0,0,0,0,0]]])


    dst_r = (cv2.filter2D(img[:,:,0], -1, kernal[0,:,:]) + cv2.filter2D(img[:,:,1], -1, kernal[0,:,:]) + cv2.filter2D(img[:,:,2], -1, kernal[0,:,:]))/3
    dst_g = (cv2.filter2D(img[:,:,0], -1, kernal[1,:,:]) + cv2.filter2D(img[:,:,1], -1, kernal[1,:,:]) + cv2.filter2D(img[:,:,2], -1, kernal[1,:,:]))/3
    dst_b = (cv2.filter2D(img[:,:,0], -1, kernal[2,:,:]) + cv2.filter2D(img[:,:,1], -1, kernal[2,:,:]) + cv2.filter2D(img[:,:,2], -1, kernal[2,:,:]))/3
    dst = cv2.merge([dst_r, dst_g, dst_b])

    titles = ['srcImg','convImg']
    imgs = [img, dst]

    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(imgs[i])
        plt.title(titles[i])
    plt.savefig('output_plot2.png', dpi=300, bbox_inches='tight')
    plt.show()