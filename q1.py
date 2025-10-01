#!/usr/bin/env python

from pathlib import Path
from dataclasses import dataclass, astuple

import numpy as np
import cv2
import pdb
import tyro

import utils


def load_image():
    pass


    # returns H
def affine_rect(lines):
    normalized = np.array(list(map(utils.normalize, lines))) # assume 8 lines, p2
    inf_point1 = np.cross(normalized[0], normalized[1])
    inf_point2 = np.cross(normalized[2], normalized[3])
    image_infinity = utils.normalize(
        np.cross(inf_point1, inf_point2)
    )
    H = np.eye(3)
    H[-1,:] = image_infinity
    return H

@dataclass
class ImageAnno:
    name: str
    img: np.ndarray
    points: np.ndarray
    lines: np.ndarray

def load_annotated(path, q):
    with open(f'{path}/annotation/{q}_annotation.npy','rb') as f:
        q1_annotation = np.load(f, allow_pickle=True).item()
    img_dir = Path(path) / "q1"
    for img_name, points in q1_annotation.items():
        num_points = points.shape[0]
        points = np.concatenate((points, np.ones_like(points[:,-1:])), axis=-1)
        assert(num_points % 2 == 0)
        points_reshaped = points.reshape(num_points // 2, 2, 3)
        lines = np.cross(
                points_reshaped[:, 0, :],
                points_reshaped[:, 1, :]
        )
        img = cv2.imread((img_dir / img_name).with_suffix(".jpg"))
        yield ImageAnno(
            img_name,
            img,
            points,
            lines
        )

def main(data_path: str = "data", output_path: str = "output") :
    imgs_annos = load_annotated(data_path, "q1")
    Path(output_path).mkdir(exist_ok=True)
    for img_name, img, _, lines in map(astuple, imgs_annos):
        eval_lines = lines[4:]

        before_eval_angles = (utils.cosine(*eval_lines[0:2]), utils.cosine(*eval_lines[2:]))
        H = affine_rect(lines)
        H_inv = np.linalg.inv(H)
        warped = utils.MyWarp(img, H)
        warped_lines = eval_lines @ H_inv
        cv2.imwrite(
            (Path(output_path) / img_name).with_suffix(".jpg"),
            warped
        )
        after_eval_angles = (utils.cosine(*warped_lines[0:2]), utils.cosine(*warped_lines[2:]))
        with open(Path(output_path) / "q1out.txt", "w") as f:
            f.write(f"Angle 1 Before: {before_eval_angles[0]}\n")
            f.write(f"Angle 2 Before: {before_eval_angles[1]}\n")
            f.write(f"Angle 1 After: {after_eval_angles[0]}\n")
            f.write(f"Angle 2 After: {after_eval_angles[1]}\n")

        


if __name__ == "__main__":
    tyro.cli(main)
