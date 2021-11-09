import numpy as np
from skimage import io
import sys

def mirror_h(input_image):
    new_image = np.zeros(input_image.shape, input_image.dtype)
    for i in range(0, new_image.shape[0]):
        new_image[input_image.shape[0] - 1 - i] = input_image[i]
    return new_image

def mirror_v(input_image):
    new_image = np.zeros(input_image.shape, input_image.dtype)
    for i in range(0, new_image.shape[0]):
        new_image[i] = input_image[i][::-1]
    return new_image

def mirror_d(input_image):
    new_size = None
    if len(input_image.shape) == 2:
        new_size = (input_image.shape[1], input_image.shape[0])
    elif len(input_image.shape) == 3:
        new_size = (input_image.shape[1], input_image.shape[0], input_image.shape[2])
    else : 
        print("ERROR")
        return 
    new_image = np.zeros(new_size, input_image.dtype)
    for i in range(0, new_image.shape[0]):
        for j in range(0, new_image.shape[1]):
            new_image[i][j] = input_image[j][i]
    return new_image

def mirror_cd(input_image):
    new_size = None
    if len(input_image.shape) == 2:
        new_size = (input_image.shape[1], input_image.shape[0])
    elif len(input_image.shape) == 3:
        new_size = (input_image.shape[1], input_image.shape[0], input_image.shape[2])
    else : 
        print("ERROR")
        return 
    new_image = np.zeros(new_size, input_image.dtype)
    for i in range(0, new_image.shape[0]):
        for j in range(0, new_image.shape[1]):
            new_image[i][j] = input_image[input_image.shape[0] - 1 -  j][input_image.shape[1] - 1 - i]
    return new_image

def mirror_image(m_type, input_file, output_file):
    input_image = io.imread(input_file)
    mirror_image = None
    if m_type == 'h':
        mirror_image = mirror_h(input_image)
    elif m_type == 'v':
        mirror_image = mirror_v(input_image)
    elif m_type == 'd':
        mirror_image = mirror_d(input_image)
    elif m_type == 'cd':
        mirror_image = mirror_cd(input_image)
    else:
        return
    io.imsave(output_file, mirror_image)
    return mirror_image

def rotate_cw_90(input_image):
    new_size = None
    if len(input_image.shape) == 2:
        new_size = (input_image.shape[1], input_image.shape[0])
    elif len(input_image.shape) == 3:
        new_size = (input_image.shape[1], input_image.shape[0], input_image.shape[2])
    else : 
        print("ERROR")
        return 
    new_image = np.zeros(new_size, input_image.dtype)
    for i in range(0, new_image.shape[0]):
        for j in range(0, new_image.shape[1]):
            new_image[i][j] = input_image[input_image.shape[0] - 1 -  j][i]
    return new_image

def rotate_cw_270(input_image):
    new_size = None
    if len(input_image.shape) == 2:
        new_size = (input_image.shape[1], input_image.shape[0])
    elif len(input_image.shape) == 3:
        new_size = (input_image.shape[1], input_image.shape[0], input_image.shape[2])
    else : 
        print("ERROR")
        return 
    new_image = np.zeros(new_size, input_image.dtype)
    for i in range(0, new_image.shape[0]):
        for j in range(0, new_image.shape[1]):
            new_image[i][j] = input_image[j][input_image.shape[1] - 1 - i]
    return new_image

def rotate_cw_180(input_image):
    new_image = np.zeros(input_image.shape, input_image.dtype)
    for i in range(new_image.shape[0] - 1, -1, -1):
        new_image[input_image.shape[0] - 1 - i] = input_image[i][::-1]
    return new_image

def rotate_image(r_type, angle, input_file, output_file):
    input_image = io.imread(input_file)
    rotate_image = None
    if r_type == 'ccw' :
        angle = -1 * angle
    angle = angle % 360
    if angle == 0 :
        rotate_image = input_image
    elif angle == 90 :
        rotate_image = rotate_cw_90(input_image)
    elif angle == 180 :
        rotate_image = rotate_cw_180(input_image)
    elif angle == 270 :
        rotate_image = rotate_cw_270(input_image)
    io.imsave(output_file, rotate_image)
    return rotate_image


def extract_image(left_x, top_y, width, height, input_file, output_file):
    input_image = io.imread(input_file)
    new_size = None
    if len(input_image.shape) == 2:
        new_size = (height, width)
    elif len(input_image.shape) == 3:
        new_size = (height, width, input_image.shape[2])
    else : 
        print("ERROR")
        return
    extract_image = np.zeros(new_size, dtype=input_image.dtype)
    if not (left_x + width - 1 < 0 or top_y + height - 1 < 0 or left_x > input_image.shape[1] or top_y > input_image.shape[0]):
        for i in range(max(top_y, 0), min(top_y + height, input_image.shape[0])):
            for j in range(max(left_x, 0), min(left_x + width, input_image.shape[1])):
                extract_image[i - top_y][j - left_x] = input_image[i][j]
    io.imsave(output_file, extract_image)
    return extract_image

def auto_rotate(input_file, output_file):
    input_image = io.imread(input_file)
    size_y, size_x = input_image.shape[0], input_image.shape[1]
    # lt rt
    # lb rb
    lt = input_image[0 : size_y//2,0 : size_x//2]
    lb = input_image[(size_y + 1)//2 : ,0 : size_x//2]
    rt = input_image[0 : size_y//2, ((size_x + 1)//2) : ]
    rb = input_image[(size_y + 1)//2 : ,((size_x + 1)//2) : ]
    rt = rt.mean()
    lb = lb.mean()
    lt = lt.mean()
    rb = rb.mean()
    div_x = (rt + rb) - (lt + lb)
    div_y = (lt + rt) - (lb + rb)
    output_image = None
    if abs(div_x) > abs(div_y) :
        if div_x > 0:
            output_image = rotate_cw_270(input_image)
        else:
            output_image = rotate_cw_90(input_image)
    else:
        if div_y < 0:
            output_image = rotate_cw_180(input_image)
        else:
            output_image = input_image
    io.imsave(output_file, output_image)
    return 


if __name__ == "__main__":
    if sys.argv[1] == 'mirror':
        mirror_image(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'extract':
        extract_image(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7])
    elif sys.argv[1] == 'rotate':
        rotate_image(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    elif sys.argv[1] == 'autorotate':
        auto_rotate(sys.argv[2], sys.argv[3])
    else:
        print('command not found')

