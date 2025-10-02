#!/usr/bin/env python
from dataclasses import astuple
from pathlib import Path
import pdb

import numpy as np
import cv2
import tyro

from q1 import load_annotated, affine_rect 
import utils

def metric_from_affine(lines):
    lines = np.array(list(map(utils.normalize, lines)))
    affine_lines_paired = lines.reshape(lines.shape[0]//2, 2, 3)[:4, ...] # shape 4, 2, 3
    def create_row(perp_lines):
        l, m = perp_lines
        row = np.array([ l[0] * m[0], l[0] * m[1] + l[1] * m[0], l[1] * m[1] ])
        return row

    A = np.stack(
        (*[create_row(perp_lines) for perp_lines in affine_lines_paired],), 
        axis=0
    )
    _, E, Vh = np.linalg.svd(A)
    s = Vh[-1]
    dual_conic_img = np.zeros((3,3))
    dual_conic_img[:-1, :-1] = np.array([
        [s[0], s[1]],
        [s[1], s[2]]
    ])
    U, E, Uh = np.linalg.svd(dual_conic_img)
    E[-1] = 1
    H_m = np.diag(np.sqrt(1 / E)) @ Uh
    return H_m

    
def main(data_path: str = "data", output_path: str = "output/q2"):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    imgs_annos_aff = load_annotated(data_path, "q1")
    imgs_annos_met = load_annotated(data_path, "q2")
    
    for parallel, perp in zip(imgs_annos_aff, imgs_annos_met):

        H_a = parallel.inv_normalize_T @ affine_rect(parallel.normalized_lines) @ parallel.normalize_T
        Ht, affine_img = utils.MyWarp(parallel.img, H_a)
        H_a_inv = np.linalg.inv(H_a)

        # a2m = utils.anno_to_anno_transform(parallel.raw_points, perp.raw_points)
        a2m = np.eye(3)
        m2a = np.eye(3)
        # m2a = np.linalg.inv(a2m)

        perpendicular_lines_unrect = perp.raw_lines
        perpendicular_lines_affine = perp.raw_lines @ (a2m @ H_a_inv @ m2a)

        perpendicular_points_unrect = perp.raw_points
        perpendicular_points_affine = perp.raw_points @ (Ht @ a2m @ H_a @ m2a).T

        utils.write_results(perp.img, affine_img, perpendicular_points_unrect, perpendicular_points_affine, output_path, perp.name, tag="affine")

        H_m = metric_from_affine(perpendicular_lines_affine)
        H_m_inv = np.linalg.inv(H_m)

        Ht, metric_img = utils.MyWarp(perp.img, H_m @ a2m @ H_a) # wonder what would happen if I did MyWarp(img, H_m @ H_a)
        perpendicular_lines_metric = perpendicular_lines_affine @ (H_m_inv)
        perpendicular_points_metric = perpendicular_points_affine @ (H_m_inv).T

        utils.write_results(perp.img, metric_img, perpendicular_points_unrect, perpendicular_points_metric, output_path, perp.name, tag="metric", skip_orig=True)

        unrect_angles = (utils.cosine(*perpendicular_lines_unrect[4:6, :]), utils.cosine(*perpendicular_lines_unrect[6:8, :]))
        metric_angles = (utils.cosine(*perpendicular_lines_metric[4:6, :]), utils.cosine(*perpendicular_lines_metric[6:8, :]))

        with open(Path(output_path) / f"{perp.name}_q2out.txt", "w") as f:
            # f.write(f"Affine angles: {affine_angles[0]}, {affine_angles[1]}\n")
            f.write(f"Metric angles: {metric_angles[0]}, {metric_angles[1]}\n")


if __name__ == "__main__":
    tyro.cli(main)

