# import the necessary packages

import argparse

import os
import glob


def pythonファイル実行時の引数をバリデーションする():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input video")
    ap.add_argument("-o", "--output", required=True, help="path to output video")
    ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.5,
        help="minimum probability to filter weak detections",
    )
    ap.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help="threshold when applyong non-maxima suppression",
    )
    args = vars(ap.parse_args())
    return args


def アウトプットフォルダを綺麗にする():
    files = glob.glob("output/*.png")
    for f in files:
        os.remove(f)


def main():
    """
    STEP_1
    main.py 41行目まで

    1-1 python main_refactor_STEP1.pyを実行
    1-2 python main.pyの後 下記コメントアウト(53,54)行目を外しpython main_refactor_STEP1.pyを実行 同様のエラー
    1-3 python main_refactor_STEP1.py--input input/highway.mp4 --output output/highway.avi --yolo yolo-coco
    """
    アウトプットフォルダを綺麗にする()
    arg_res = pythonファイル実行時の引数をバリデーションする()
    print(arg_res, type(arg_res), arg_res["yolo"])


if __name__ == "__main__":
    main()
