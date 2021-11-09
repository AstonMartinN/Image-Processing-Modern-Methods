import numpy as np
from skimage import io, img_as_ubyte
import sys
from numpy.linalg import eig

def mse_metric(im1, im2):
    if im1.shape != im2.shape:
        print('error: image size not required')
        return -1
    return np.mean(np.square(im1-im2))
    
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

def prepare_image(image, rad):
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
    return image

########################################
########### MAIN #######################
########################################

def calc_window_grad(window, s4, rad, matrixs):
    grad_x = np.sum(window * matrixs[0])
    grad_y = np.sum(window * matrixs[1])
    divcoeff = 1.0 / (6.28318530718*s4)
    grad_x *= divcoeff
    grad_y *= divcoeff
    result = (grad_x*grad_x + grad_y*grad_y)**0.5
    return result

def calc_grad(input_file, output_file, sigma):
    image = io.imread(input_file)
    image = img_as_ubyte(image).astype(np.int64)
    if len(image.shape) > 2:
        image = image[:,:,0]
    out_image = np.zeros(image.shape)
    rad = int(np.ceil(3.0*sigma))
    image = prepare_image(image, rad)
    s = sigma*sigma
    xmat = np.array([[-i*np.exp(-(i*i + j*j)/(2.0*s)) for i in range(-rad, rad+1)] for j in range(-rad, rad+1)])
    ymat = np.array([[-j*np.exp(-(i*i + j*j)/(2.0*s)) for i in range(-rad, rad+1)] for j in range(-rad, rad+1)])
    for i in range(0, out_image.shape[0]):
        for j in range(0, out_image.shape[1]):
            out_image[i, j] = calc_window_grad(image[i:i+rad+rad+1 , j:j+rad+rad+1], s*s, rad, (xmat, ymat))
    out_image = out_image*(255/np.max(out_image))
    io.imsave(output_file, out_image.astype(np.uint8))
    return 0

def calc_window_grad_with_tg(window, s4, rad, matrixs):
    grad_x = np.sum(window * matrixs[0])
    grad_y = np.sum(window * matrixs[1])
    divcoeff = 1.0 / (6.28318530718*s4)
    grad_x *= divcoeff
    grad_y *= divcoeff
    result = (grad_x*grad_x + grad_y*grad_y)**0.5
    tg = 0
    if grad_x != 0: #x/y ! вдоль линии угол
        tg = grad_y / grad_x
    else:
        tg = grad_y / 0.00000000001
    return (result, tg)

def nonmax_process(out_image, grad_image, tg_mass):
    for i in range(0, out_image.shape[0]):
        for j in range(0, out_image.shape[1]):
            tg = tg_mass[i, j]
            color_1, color_2 = 0.0, 0.0
            if tg > 0.0:
                if tg > 1.0:
                    color_1 = (1.0 - 1.0/tg) * grad_image[1+(i + 1), 1+(j)] + (1.0/tg) * grad_image[1+(i + 1),1+(j + 1)]
                    color_2 = (1.0 - 1.0/tg) * grad_image[1+(i - 1), 1+(j)] + (1.0/tg) * grad_image[1+(i - 1),1+(j - 1)]
                else:
                    color_1 = (1.0 - tg) * grad_image[1+(i),1+(j+1)] + tg * grad_image[1+(i + 1),1+(j + 1)]
                    color_2 = (1.0 - tg) * grad_image[1+(i),1+(j-1)] + tg * grad_image[1+(i - 1),1+(j - 1)]
            else:
                tg *= -1.0
                if tg > 1.0:
                    color_1 = (1.0 - 1.0/tg) * grad_image[1+(i + 1),1+(j)] + (1.0/tg) * grad_image[1+(i + 1),1+(j - 1)]
                    color_2 = (1.0 - 1.0/tg) * grad_image[1+(i - 1),1+(j)] + (1.0/tg) * grad_image[1+(i - 1),1+(j + 1)]
                else:
                    color_1 = (1.0 - tg) * grad_image[1+i,1+ j - 1] + tg * grad_image[1+(i + 1),1+(j - 1)]
                    color_2 = (1.0 - tg) * grad_image[1+i,1+ j + 1] + tg * grad_image[1+(i - 1),1+(j + 1)]
            if grad_image[1+i, 1+j] >= max(color_1, color_2):
                out_image[i, j] = grad_image[1+i, 1+j]

def non_max(input_file, output_file, sigma):
    image = io.imread(input_file)
    image = img_as_ubyte(image).astype(np.int64)
    if len(image.shape) > 2:
        image = image[:,:,0]
    input_image_size = image.shape
    grad_image = np.zeros(input_image_size, dtype=np.float64)
    tg_mass = np.zeros(input_image_size, dtype=np.float64)
    out_image = np.zeros(input_image_size, dtype=np.float64)
    #change main image
    rad = int(np.ceil(3.0*sigma))
    image = prepare_image(image, rad)

    s = sigma*sigma
    xmat = np.array([[-i*np.exp(-(i*i + j*j)/(2.0*s)) for i in range(-rad, rad+1)] for j in range(-rad, rad+1)])
    ymat = np.array([[-j*np.exp(-(i*i + j*j)/(2.0*s)) for i in range(-rad, rad+1)] for j in range(-rad, rad+1)])
    for i in range(0, grad_image.shape[0]):
        for j in range(0, grad_image.shape[1]):
            grad_image[i, j], tg_mass[i, j] = calc_window_grad_with_tg(image[i:i+rad+rad+1 , j:j+rad+rad+1], s*s, rad, (xmat, ymat))
    #grad_image = grad_image*(255/np.max(grad_image))
    grad_image = prepare_image(grad_image, 1)
    nonmax_process(out_image, grad_image, tg_mass)

    out_image = out_image*(255/np.max(out_image))    
    io.imsave(output_file, out_image.astype(np.uint8))
    return 0

def search_ones(out_image):
    while 1 in out_image:
        count = 0
        for i in range(0, out_image.shape[0]):
            if 1 in out_image[i]:
                for j in range(0, out_image.shape[1]):
                    if out_image[i, j] == 1:
                        k, l, d, f = -1, 2, -1, 2
                        if i == 0:
                            k = 0
                        if j == 0:
                            d = 0
                        if i == out_image.shape[0] - 1:
                            l = 1
                        if j == out_image.shape[1] - 1:
                            f = 1
                        out_image[i, j] = 2
                        if 255 in out_image[i+k:i+l, j+d:j+f]:
                            out_image[i, j] = 255
                            count += 1
                        elif 1 not in out_image[i+k:i+l, j+d:j+f]:
                            out_image[i, j] = 0
                            count += 1
                        else:
                            out_image[i, j] = 1
        if count == 0:
            out_image[out_image == 1] = 0
            break

def canny_alg(input_file, output_file, sigma, thr_high, thr_low):
    image = io.imread(input_file)
    image = img_as_ubyte(image).astype(np.int64)
    if len(image.shape) > 2:
        image = image[:,:,0]
    input_image_size = image.shape
    grad_image = np.zeros(input_image_size, dtype=np.float)
    tg_mass = np.zeros(input_image_size, dtype=np.float)
    out_image = np.zeros(input_image_size, dtype=np.float)

    #change main image
    rad = int(np.ceil(3.0*sigma))
    image = prepare_image(image, rad)
    s = sigma*sigma
    xmat = np.array([[-i*np.exp(-(i*i + j*j)/(2.0*s)) for i in range(-rad, rad+1)] for j in range(-rad, rad+1)])
    ymat = np.array([[-j*np.exp(-(i*i + j*j)/(2.0*s)) for i in range(-rad, rad+1)] for j in range(-rad, rad+1)])
    for i in range(0, grad_image.shape[0]):
        for j in range(0, grad_image.shape[1]):
            grad_image[i, j], tg_mass[i, j] = calc_window_grad_with_tg(image[i:i+rad+rad+1 , j:j+rad+rad+1], s*s, rad, (xmat, ymat))
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #grad_image = grad_image*(255/np.max(grad_image))

    grad_image = prepare_image(grad_image, 1)
    nonmax_process(out_image, grad_image, tg_mass)

    out_image = out_image.astype(np.uint8)
    max_grad = np.max(grad_image)
    hight_level, low_level = thr_high*max_grad, thr_low*max_grad
    for i in range(0, out_image.shape[0]):
        for j in range(0, out_image.shape[1]):
            if out_image[i, j] > 0:
                if grad_image[i+1, j+1] > hight_level:
                    out_image[i, j] = 255
                elif grad_image[i+1, j+1] < hight_level and grad_image[i+1, j+1] > low_level:
                    out_image[i, j] = 1
                else:
                    out_image[i, j] = 0

    search_ones(out_image)
    io.imsave(output_file, out_image)
    return 0


#########################
#########################
#########################
#########################


def parse_sys_argv():
    if sys.argv[1] in {'mse', 'psnr', 'ssim'}:
        print(calc_metrics(sys.argv[1], sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == 'grad':
        calc_grad(sys.argv[3], sys.argv[4], float(sys.argv[2]))
    elif sys.argv[1] == 'nonmax':
        non_max(sys.argv[3], sys.argv[4], float(sys.argv[2]))
    elif sys.argv[1] == 'canny':
        canny_alg(sys.argv[5], sys.argv[6], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    else:
        print('command not found')


if __name__ == "__main__":
    parse_sys_argv()



