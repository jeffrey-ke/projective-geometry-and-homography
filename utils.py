from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


def normalize(v):
    return v / max(np.sqrt(v[0]**2 + v[1]**2 + v[2]**2), np.finfo(np.float32).eps)


def MyWarp(img, H):
    h, w = img.shape[:2]
    pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float64).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = (pts.min(axis=0).ravel() - 0.5).astype(int)
    [xmax, ymax] = (pts.max(axis=0).ravel() + 0.5).astype(int)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    # Ht = np.eye(3)

    result = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin))
    return Ht, result


def cosine(u, v):
    return (u[0] * v[0] + u[1] * v[1]) / (np.sqrt(u[0]**2 + u[1]**2) * np.sqrt(v[0]**2 + v[1]**2))


def annotate(impath):
    im = Image.open(impath)
    im = np.array(im)
    h,w = im.shape[:-1]
    print(im.shape[:-1])
    print(f"height: {h}, width: {w}")
    clicks = []

    def click(event):
        nonlocal clicks
        if len(clicks) == 16:
            plt.close()
            return
        x, y = event.xdata, event.ydata
        print(x,y)
        clicks.append([x, y])


    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.imshow(im)

    _ = fig.canvas.mpl_connect('button_press_event', click)
    plt.show()

    return clicks
