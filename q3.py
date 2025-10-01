from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np
import cv2
import tyro

@dataclass
class PointAnno:
    img_name: str
    img_normal: np.ndarray
    img_perspective: np.ndarray
    points_normal: np.ndarray
    points_perspective: np.ndarray

def load_point_anno(path):
    obj = np.load(Path(path) / "annotation" / "q3_annotation.npy", allow_pickle=True).item()
    for img_name, points_persp in obj.items():
        img_id = img_name.split("-")[0]
        img_normal = cv2.imread(str(Path(path) / "q3" / f"{img_id}-normal.png"))
        w, h = img_normal.shape[:-1]
        points_normal = np.array([
            [0,0],
            [0, h],
            [w,h],
            [w,0],
        ])

        yield PointAnno(
            img_name,
            img_normal,
            cv2.imread(str(Path(path) / "q3" / f"{img_id}-perspective.png")),
            points_normal,
            points_persp,
        )

def main(data_path: str, output_path: str):
    img_annos = load_point_anno(data_path)
    for img_name, image_normal, image_pers, points_normal, points_persp in map(astuple, img_annos):
        pass



if __name__ == "__main__":
    tyro.cli(main)
