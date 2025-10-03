from dataclasses import dataclass
from pathlib import Path

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


def annotate(impath, num_points=16):
    im = Image.open(impath)
    im = np.array(im)
    h,w = im.shape[:-1]
    print(im.shape[:-1])
    print(f"height: {h}, width: {w}")
    clicks = []

    def click(event):
        nonlocal clicks
        if len(clicks) == num_points:
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

@dataclass
class ImageAnno:
    name: str
    img: np.ndarray
    raw_points: np.ndarray
    normalized_points: np.ndarray
    raw_lines: np.ndarray
    normalized_lines: np.ndarray
    normalize_T: np.ndarray
    inv_normalize_T: np.ndarray

def anno_to_anno_transform(raw_points_src, raw_points_tar):
    # get the of raw_points_src to match raw_points_tar
    # center raw_points_src around the centroid of raw_points_tar
    centroid_tar = np.mean(raw_points_tar[:, :-1], axis=0).reshape(2,1)
    scale_tar = np.max(
        np.linalg.norm(raw_points_tar[:, :-1], axis=-1)
    )
    # this reminds me of depthanything's scale invariant loss
    centroid_src = np.mean(raw_points_src[:, :-1], axis=0).reshape(2,1)
    scale_src = np.max(
        np.linalg.norm(raw_points_src[:, :-1], axis=-1)
    )
    src2norm = np.diag([1/scale_src, 1/scale_src, 1])
    src2norm[:2, -1:] = -1/scale_src * centroid_src

    norm2tar = np.diag([scale_tar, scale_tar, 1])
    norm2tar[:2, -1:] = centroid_tar

    return norm2tar @ src2norm

def add_lines(img, points, seed):
    points = points / points[..., -1:] #TODO: put this in your journal: a good example of broadcasting rules: shape 16x3 / 16x1
    points = (
        points.reshape(points.shape[0]//4, 4, 3)
        [..., :-1]
        .astype(int)
    )
    rng = np.random.default_rng(seed=seed)
    for (x11,y11),(x12,y12),(x21,y21),(x22,y22) in points:
        color = rng.uniform(0, 255, (3,)).astype(int).tolist()
        cv2.line(img, (x11,y11), (x12,y12), color, 21)
        cv2.line(img, (x21,y21), (x22,y22), color, 21)
    return img

def write_results(img_orig, img_warped, points_orig, points_warped, output_path, name, tag, skip_orig=False):
    if not skip_orig:
        cv2.imwrite(
            str(
                (Path(output_path) / f"{name}_unrectified_train_{tag}").with_suffix(".jpg")
            ),
            add_lines(img_orig, points_orig[:8], seed=10)
        )
        cv2.imwrite(
            str(
                (Path(output_path) / f"{name}_unrectified_eval_{tag}").with_suffix(".jpg")
            ),
            add_lines(img_orig, points_orig[8:], seed=10)
        )

    cv2.imwrite(
        str(
            (Path(output_path) / f"{name}_rectified_train_{tag}").with_suffix(".jpg")
        ),
        add_lines(img_warped, points_warped[:8], seed=10)
    )
    cv2.imwrite(
        str(
            (Path(output_path) / f"{name}_rectified_eval_{tag}").with_suffix(".jpg")
        ),
        add_lines(img_warped, points_warped[8:], seed=10)
    )
