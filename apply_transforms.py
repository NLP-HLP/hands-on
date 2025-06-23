import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import skimage


def main(gaze_path):
    gaze_path = Path(gaze_path)
    gaze = pd.read_csv(gaze_path)
    stimuli = list(gaze["stimulus"].unique())

    transforms_path = gaze_path.with_suffix(".transforms.json")
    with open(transforms_path) as f:
        transforms = json.load(f)

    gaze_corrected = pd.DataFrame()
    for stimulus in stimuli:
        stimulus_gaze = gaze[gaze["stimulus"] == stimulus]
        src_points, dst_points = transforms[stimulus]
        if src_points is not None and dst_points is not None:
            transform = skimage.transform.ThinPlateSplineTransform()
            src = np.array(src_points)
            dst = np.array(dst_points)
            transform.estimate(src, dst)
            stimulus_gaze[["pixel_x", "pixel_y"]] = transform(
                stimulus_gaze[["pixel_x", "pixel_y"]]
            )
        gaze_corrected = pd.concat([gaze_corrected, stimulus_gaze])

    gaze_corrected.to_csv(gaze_path.with_suffix(".corrected.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gaze_path",
        type=Path,
        help="Path to the gaze data CSV file.",
    )
    args = parser.parse_args()
    main(args.gaze_path)
