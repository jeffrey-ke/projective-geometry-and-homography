#!/usr/bin/env python

from dataclasses import astuple, dataclass
import pdb
import os
from pathlib import Path

import numpy as np
import cv2
import tyro

import utils

@dataclass
class PointAnno:
    img_name: str
    img_normal: np.ndarray
    img_perspective: np.ndarray
    points_normal: np.ndarray
    points_perspective: np.ndarray

def load_point_anno(path, normalize=True):
    obj = np.load(Path(path) / "annotation" / "q3_annotation.npy", allow_pickle=True).item()
    for img_name, points_persp in obj.items():
        img_id = img_name.split("-")[0]
        img_normal = cv2.imread(str(Path(path) / "q3" / f"{img_id}-normal.png"))
        h, w = img_normal.shape[:-1]
        points_normal = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1],])
        points_persp = np.concatenate((points_persp, np.ones_like(points_persp[..., :1])), axis=-1)

        if normalize:
            points_normal = np.array(
                    list(
                        map(
                            utils.normalize,
                            points_normal
                        )
                    )
            )
            points_persp = np.array(
                list(
                        map(
                            utils.normalize,
                            points_persp
                        )
                )
            )

        yield PointAnno(
            img_name,
            img_normal,
            cv2.imread(str(Path(path) / "q3" / f"{img_id}-perspective.png")),
            points_normal,
            points_persp,
        )

def create_A_i(point_normal, point_persp):
    x,y,w = point_persp
    point_normal = point_normal.reshape(1,3)
    A_i = np.zeros((2,9))
    A_i[0, 3:6] = -w * point_normal
    A_i[0, 6:9] = y * point_normal
    A_i[1, 0:3] = w * point_normal
    A_i[1, 6:9] = -x * point_normal
    return A_i

def main(data_path: str = "data", output_path: str = "output"):
    img_annos = load_point_anno(data_path, )
    for img_name, image_normal, image_pers, points_normal, points_persp in map(astuple, img_annos):
        A = np.concatenate(
            (*[create_A_i(p_n, p_p) for p_n, p_p in zip(points_normal, points_persp)],),
            axis=0
        )
        _, E, Vh = np.linalg.svd(A)
        h = Vh[-1]
        H = np.array([
            h[0:3],
            h[3:6],
            h[6:9]
        ])
        warped_img = cv2.warpPerspective(image_normal, H, image_pers.shape[:-1][::-1])
        warped_img[warped_img == 0] = image_pers[warped_img == 0]
        warped_points = points_normal @ H
        cv2.imwrite(
            os.path.join(output_path, f"q3_{img_name}_warped.png"),
            warped_img
        )

if __name__ == "__main__":
    tyro.cli(main)
