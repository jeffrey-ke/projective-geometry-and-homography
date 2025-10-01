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

def annotate_img(img, lines):
    return img
    
def main(data_path: str = "data", output_path: str = "output"):
    imgs_annos_aff = load_annotated(data_path, "q1")
    imgs_annos_met = load_annotated(data_path, "q2")
    
    for aff_anno, met_anno in zip(imgs_annos_aff, imgs_annos_met):
        img_name, img, _, lines_aff = astuple(aff_anno)
        H_a = affine_rect(lines_aff)
        affine_img = utils.MyWarp(img, H_a)
        H_a_inv = np.linalg.inv(H_a)
        # H_a_inv /= H_a_inv[2,2]

        affine_lines = lines_aff @ H_a_inv
        cv2.imwrite(
            str(
                (Path(output_path) / f"{img_name}_affine").with_suffix(".jpg")
            ),
            annotate_img(affine_img, affine_lines)
        )
        affine_angles = (utils.cosine(*affine_lines[4:6, :]), utils.cosine(*affine_lines[6:8, :]))

        *_, lines_met = astuple(met_anno)
        H_m = metric_from_affine(lines_met)
        H_m_inv = np.linalg.inv(H_m)
        # H_m_inv /= H_m_inv[2,2]
        metric_img = utils.MyWarp(img, H_m @ H_a) # wonder what would happen if I did MyWarp(img, H_m @ H_a)
        metric_lines = lines_met @ (H_m_inv)
        cv2.imwrite(
            str(
                    (Path(output_path) / f"{img_name}_metric").with_suffix(".jpg")
            ),
            annotate_img(metric_img, metric_lines)
        )
        metric_angles = (utils.cosine(*metric_lines[4:6, :]), utils.cosine(*metric_lines[6:8, :]))

        with open(Path(output_path) / f"{img_name}_q2out.txt", "w") as f:
            f.write(f"Affine angles: {affine_angles[0]}, {affine_angles[1]}\n")
            f.write(f"Metric angles: {metric_angles[0]}, {metric_angles[1]}\n")


if __name__ == "__main__":
    tyro.cli(main)

