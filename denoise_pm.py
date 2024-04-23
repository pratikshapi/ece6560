import numpy as np

def F1(dt, K):
    return np.exp(-1 * (np.power(dt, 2)) / (np.power(K, 2)))

def F2(dt, K):
    func = 1 / (1 + ((dt/K)**2))
    return func

def anisodiff_f1(img, steps, K, del_t=0.25):
    upgrade_img = np.zeros(img.shape, dtype=img.dtype)
    for t in range(steps):
        dn = img[:-2, 1:-1] - img[1:-1, 1:-1]
        ds = img[2:, 1:-1] - img[1:-1, 1:-1]
        de = img[1:-1, 2:] - img[1:-1, 1:-1]
        dw = img[1:-1, :-2] - img[1:-1, 1:-1]
        upgrade_img[1:-1, 1:-1] = img[1:-1, 1:-1] + del_t * (
            F1(dn, K) * dn + F1(ds, K) * ds + F1(de, K) * de + F1(dw, K) * dw)
        img = upgrade_img
    return img

def anisodiff_f2(img, steps, K, del_t=0.25):
    upgrade_img = np.zeros(img.shape, dtype=img.dtype)
    for t in range(steps):
        dn = img[:-2, 1:-1] - img[1:-1, 1:-1]
        ds = img[2:, 1:-1] - img[1:-1, 1:-1]
        de = img[1:-1, 2:] - img[1:-1, 1:-1]
        dw = img[1:-1, :-2] - img[1:-1, 1:-1]
        upgrade_img[1:-1, 1:-1] = img[1:-1, 1:-1] + del_t * (
            F2(dn, K) * dn + F2(ds, K) * ds + F2(de, K) * de + F2(dw, K) * dw)
        img = upgrade_img
    return img

