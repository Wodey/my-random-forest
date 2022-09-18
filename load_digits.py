import numpy as np
from PIL import Image
from os import listdir

def _load_digit(name):
    res = []
    for i in listdir(f"digits_dataset/{name}"):
        image = Image.open(f"digits_dataset/{name}/{i}")
        resized_image = image.resize((100, 100))
        image_array = np.asarray(resized_image)
        reshaped_image = image_array.reshape(1, image_array.size)
        if reshaped_image.shape == (1, 30000):
            res.append(reshaped_image.tolist()[0])
    return np.array(res)

def load_digits():
    res = _load_digit('0')
    res_target = np.full((res.shape[0], 1), 0)
    for i in {'1', '2', '3', '4', '5', '6', '7', '8', '9'}:
        print(f"current_number - {i}")
        current = _load_digit(i)
        res = np.r_[res, current]
        current_target = np.full((current.shape[0], 1), int(i))
        res_target = np.r_[res_target, current_target]

    p = np.random.permutation(len(res))
    X = res[p]
    y = res_target[p]
    return X, y
