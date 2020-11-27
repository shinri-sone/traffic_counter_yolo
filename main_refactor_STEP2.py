# import the necessary packages

import argparse

import os
import glob

# Step1から新しくimportしたパッケージ
import cv2


def VIDEOCAPTUREの初期設定(input):
    vs = cv2.VideoCapture(input)
    writer = None
    (W, H) = (None, None)

    frameIndex = 0
    return vs, writer, (W, H), frameIndex


def COCOクラスのラベルを読み込む(args):
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    return LABELS


def YOLOのweightと設定のファイルパスからdarknetでYOLOを使う(args):
    """
    使う→determine the output layer
    """
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
    # darknetでmodelを読み込む
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln_old = net.getLayerNames()
    # 参考
    # https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#ac1840896b8643f91532e98c660627fb9
    ln_new = [ln_old[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return ln_new
    # わかりやすく
    # print(ln_old)
    # print(net.getUnconnectedOutLayers())
    # for i in net.getUnconnectedOutLayers():
    #    print(ln_old[i[0] - 1])


def pythonファイル実行時の引数をバリデーションする():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input video")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output video")
    ap.add_argument("-y", "--yolo", required=True,
                    help="base path to YOLO directory")
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
    STEP_2
    main.py 78行目まで

    2-1 どんなラベルを読み込むか?
    2-2 57,58行目の色の配列をPythonインタラクティブシェルで確認
    2-3 YOLO..の関数をコメントアウトし最後のレイヤーを決定する
    2-4 VIDEOCAPTUREの初期設定
    """
    アウトプットフォルダを綺麗にする()
    args = pythonファイル実行時の引数をバリデーションする()
    COCOクラスのラベルを読み込む(args)

    final_layer=YOLOのweightと設定のファイルパスからdarknetでYOLOを使う(args)
    print(final_layer)
    # 2-4 VIDEOCAPTUREの初期設定
    vs, writer, (W, H), frameIndex = VIDEOCAPTUREの初期設定(args["input"])


if __name__ == "__main__":
    main()
