import numpy as np
from skimage import io, img_as_ubyte
import sys


def mse_metric(im1, im2):
    if im1.shape != im2.shape:
        print('error: image size not required')
        return -1
    return np.mean(np.square(im1-im2))
#    mse = 0.0
#    e = im1 - im2
#    coeff = 1.0 / (e.shape[0]*e.shape[1])
#    for i in e:
#        mse += coeff * np.dot(i, i)
#    return (mse, np.mean(np.square(im1-im2)), coeff*np.sum(np.square(im1-im2)))

def psnr_metric(im1, im2):
    mse = mse_metric(im1, im2)
    return 10.0 * np.log10((255 * 255) / mse)

def my_cov(a, b):
    return np.mean((a - np.mean(a))*(b - np.mean(b)))

def ssim_metric(im1, im2):
    mu_1 = np.mean(im1)
    mu_2 = np.mean(im2)
    sigma_1 = np.var(im1, ddof=1)
    sigma_2 = np.var(im2, ddof=1)
    #sigma_1_2 = np.cov(np.concatenate((im1.reshape(1, -1), im2.reshape(1, -1)), axis=0))[0, 1]
    sigma_1_2 = my_cov(im1, im2)
    c1 = 0.01 * 0.01 * 255 * 255
    c2 = 0.03 * 0.03 * 255 * 255
    l = (2 * mu_1 * mu_2 + c1) / (mu_1 * mu_1 + mu_2 * mu_2 + c1)
    k = (2 * sigma_1_2 + c2) / (sigma_1 + sigma_2 + c2)
    
    return l*k

def calc_metrics(metrics_type, input_file_1, input_file_2):
    file_1 = io.imread(input_file_1)
    file_2 = io.imread(input_file_2)
    #transform to [0, 255]
    file_1 = img_as_ubyte(file_1).astype(np.int64)
    file_2 = img_as_ubyte(file_2).astype(np.int64)
    #check image size, same size required 
    if file_1.shape[0] != file_2.shape[0] or file_1.shape[1] != file_2.shape[1] :
        print('error: imege size not required!')
        return -1
    #transform to one color component
    if len(file_1.shape) > 2:
        file_1 = file_1[:,:,0]
    if len(file_2.shape) > 2:
        file_2 = file_2[:,:,0]
    if metrics_type == 'mse':
        return mse_metric(file_1, file_2)
    if metrics_type == 'psnr':
        return psnr_metric(file_1, file_2)
    if metrics_type == 'ssim':
        return ssim_metric(file_1, file_2)
    return -1

def to_i(i, a, b):
    if i < a:
        return a
    elif i > b:
        return b
    return i

def prepare_image(image, rad):
    #print('rad', rad)
    t1 = []
    t1.extend([[image[0]]] * rad)
    t1.append(image)
    t1.extend([[image[-1]]] * rad)
    image = np.concatenate(tuple(t1), axis=0)
    t2 = []
    t2.extend([image[:,0].reshape(-1, 1)] * rad)
    t2.append(image)
    t2.extend([image[:,-1].reshape(-1, 1)] * rad)
    image = np.concatenate(tuple(t2), axis=1)
    #print(image)
    return image

def median_filter(input_file, output_file, rad):
    image = io.imread(input_file)
    image = img_as_ubyte(image).astype(np.int64)
    if len(image.shape) > 2:
        image = image[:,:,0]
    out_image = np.zeros((image.shape[0], image.shape[1]))
    image = prepare_image(image, rad)
    #io.imsave('./trash.bmp', image.astype(np.uint8))

    for i in range(0, out_image.shape[0]):
        for j in range(0, out_image.shape[1]):
            k, l = i + rad, j + rad
            out_image[i, j] = np.median(image[k-rad:k+rad+1 , l-rad:l+rad+1])
    io.imsave(output_file, out_image.astype(np.uint8))
    return 0

def gauss_conv(window, rad, sigma2):
    pixel = 0
    for i in range(-rad, rad+1):
        for j in range(-rad, rad+1):
            exp = -1.0*(i*i + j*j)/(2.0*sigma2)
            pixel += window[rad+i,rad+j] * np.exp(exp) / (2.0 * np.pi * sigma2)
    return pixel

def gauss_filter(input_file, output_file, sigma_d):
    image = io.imread(input_file)
    image = img_as_ubyte(image).astype(np.int64)
    if len(image.shape) > 2:
        image = image[:,:,0]
    out_image = np.zeros((image.shape[0], image.shape[1]))
    rad = int(np.ceil(3.0*sigma_d))
    image = prepare_image(image, rad)
    sigma2 = -1.0 / (2.0 * sigma_d * sigma_d)
    one_pisigma2 = 1.0 / (2.0 * np.pi * sigma_d * sigma_d)
    for i in range(0, out_image.shape[0]):   #k, l = i + rad, j + rad
        for j in range(0, out_image.shape[1]):   #out_image[i, j] = gauss_conv(image[k-rad:k+rad+1, l-rad:l+rad+1], rad, sigma2)
            pixel = 0.0
            for li in range(-rad, rad+1):
                for lj in range(-rad, rad+1):    #exp = -1.0*(li*li + lj*lj)/sigma2
                    pixel += float(image[i+rad  +li , j+rad  +lj]) * float(np.exp(float(li*li + lj*lj)*sigma2))
            out_image[i, j] = pixel * one_pisigma2
    io.imsave(output_file, out_image.astype(np.uint8))
    return 0

def bilat_J(mass, s_d, s_r, i, j, rad):
    up = 0.0
    down = 0.0
    for k in range(i - rad, i + rad+1):
        for l in range(j - rad,j + rad+1):
            d_1 = ((i - k)*(i - k) + (j - l)*(j - l))/(2.0 * s_d * s_d)
            div = float(mass[i, j] - mass[k, l])
            d_2 = (div*div)/(2.0 * s_r * s_r)
            w = np.exp(-1.0*(d_1 + d_2))
            up += float(mass[k, l]) * w
            down += w
    if down == 0:
        return 0
    return up/down
    

def bilateral_filter(input_file, output_file, sigma_d, sigma_r):
    image = io.imread(input_file)
    image = img_as_ubyte(image).astype(np.int64)
    if len(image.shape) > 2:
        image = image[:,:,0]
    out_image = np.zeros((image.shape[0], image.shape[1]))
    rad = int(np.ceil(3.0*sigma_d))
    image = prepare_image(image, rad)
    sigma_d_22 = -1.0 / (2.0 * sigma_d * sigma_d)
    sigma_r_22 = -1.0 / (2.0 * sigma_r * sigma_r)
    for i in range(0, out_image.shape[0]):
        for j in range(0, out_image.shape[1]):
            #out_image[i, j] = bilat_J(image, sigma_d, sigma_r, i+rad, j+rad, rad)
            up, down = 0.0, 0.0
            local_color = float(image[i + rad, j + rad])
            for k in range(i, i + rad + rad + 1):
                for l in range(j, j + rad + rad + 1):
                    tmp_color = float(image[k, l])
                    d_1 = ((i + rad - k)*(i + rad - k) + (j + rad - l)*(j + rad - l)) * sigma_d_22
                    div = local_color - tmp_color
                    d_2 = div * div * sigma_r_22
                    w = float(np.exp(float(d_1 + d_2)))
                    up += w * tmp_color
                    down += w
            out_image[i, j] = up/down
    io.imsave(output_file, out_image.astype(np.uint8))
    return 0

def parse_sys_argv():
    if sys.argv[1] in {'mse', 'psnr', 'ssim'}:
        print(calc_metrics(sys.argv[1], sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == 'median':
        median_filter(sys.argv[3], sys.argv[4], int(sys.argv[2]))
    elif sys.argv[1] == 'gauss':
        gauss_filter(sys.argv[3], sys.argv[4], float(sys.argv[2]))
    elif sys.argv[1] == 'bilateral':
        bilateral_filter(sys.argv[4], sys.argv[5], float(sys.argv[2]), float(sys.argv[3]))
    else:
        print('command not found')


if __name__ == "__main__":
    parse_sys_argv()
