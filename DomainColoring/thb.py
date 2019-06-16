import os
from multiprocessing import Pool, cpu_count
from time import time

from PIL import Image

SIZE = (100, 100)
saveDirST = 'thumbsST'
saveDirMT = 'thumbsMT'
sd = saveDirMT


def get_image_paths(folder):
    return (os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')))


def create_thumbnail(filename):
    im = Image.open(filename)
    im.thumbnail(SIZE, Image.ANTIALIAS)
    base, fname = os.path.split(filename)
    save_path = os.path.join(base, sd, fname)
    im.save(save_path)


if __name__ == '__main__':
    folder = os.path.abspath('/Users/asd/Pictures/selection')

    os.mkdir(os.path.join(folder, saveDirST))
    os.mkdir(os.path.join(folder, saveDirMT))

    images = get_image_paths(folder)
    sd = saveDirMT
    st = time()
    with Pool(cpu_count()) as pool:
        pool.map(create_thumbnail, images)
        pool.close()
        pool.join()
    lMT = time() - st

    images = get_image_paths(folder)
    sd = saveDirST
    st = time()
    for image in images:
        create_thumbnail(image)
    lST = time() - st

    print('mt:', lMT, 'st:', lST)
