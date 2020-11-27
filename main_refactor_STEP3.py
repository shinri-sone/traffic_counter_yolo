# import the necessary packages

import argparse

import os
import glob
import cv2

# Step2から新しくimportしたパッケージ
import imutils
import time

def YOLOで物体検出をする(frame,ln,net):

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()




def フレーム毎の処理(frame,W,HG):
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    print(H,W)



def フレーム毎の読み込み(video_stream,width,height,yolo_layer,yolo_detactor):
    # try to determine the total number of frames in the video file
    try:
        prop = (
            cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        )
        total = int(video_stream.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    while True:
        # read the next frame from the file
        (grabbed, frame) = video_stream.read()
        #print(grabbed)
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        #3-2
        フレーム毎の処理(frame,width,height)

        # 3-3 
        YOLOで物体検出をする(frame,yolo_layer,yolo_detactor)


def VIDEOCAPTUREの初期設定(input):
    print(input)
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
    return ln_new,net
    # わかりやすく
    # print(ln_old)
    # print(net.getUnconnectedOutLayers())
    # for i in net.getUnconnectedOutLayers():
    #     print(ln_old[i[0] - 1])


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
    STEP_3
    main.py 117行目まで

    3-1 フレーム毎の読み込み
    3-2 フレーム毎の処理
    3-3 2-3をコメントアウトし,YOLOで物体検出をする
    """
    アウトプットフォルダを綺麗にする()
    args = pythonファイル実行時の引数をバリデーションする()
    # 2-2 print(COCOクラスのラベルを読み込む(args))

    # 2-3 YOLOのweightと設定のファイルパスからdarknetでYOLOを使う(args)
    layer,detactor= YOLOのweightと設定のファイルパスからdarknetでYOLOを使う(args)

    # 2-4 VIDEOCAPTUREの初期設定
    video_stream, writer, (W, H), frameIndex = VIDEOCAPTUREの初期設定(args["input"])
    # 3-1
    フレーム毎の読み込み(video_stream,W,H,layer,detactor)



if __name__ == "__main__":
    main()
