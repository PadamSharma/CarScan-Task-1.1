import os

os.system("git clone https://github.com/PushpakBhoge512/Assignment.git")

import cv2 as cv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import numpy as np
import seaborn as sns

from matplotlib.pyplot import imshow
from matplotlib import cm
import random

#  UTILITY FUNCTIONS


def get_data_dirs(img_dir, data_dir):
    img_dirs = img_dirs = sorted(glob(img_dir))
    data_dirs = sorted(glob(data_dir))
    return img_dirs, data_dirs


def get_bbox_coord(pts):
    mx_coord, mn_coord = [], []
    mx_coord.append(max(pts, key=lambda i: i[0])[0])
    mx_coord.append(max(pts, key=lambda i: i[1])[1])

    mn_coord.append(min(pts, key=lambda i: i[0])[0])
    mn_coord.append(min(pts, key=lambda i: i[1])[1])
    return mx_coord, mn_coord


def get_poly_or_bbox(IMG_DIR, DATA_DIR, alpha, get_bbox=False):
    img_dirs, data_dirs = get_data_dirs(IMG_DIR, DATA_DIR)
    for img, json in zip(img_dirs, data_dirs):
        image = cv.imread(img)
        overlay = image.copy()
        jsn = pd.read_json(json)
        jsn = jsn.loc[jsn["type"] == "polygonlabels"]
        i = 0
        mx = len(jsn)
        pts_ = []
        name = []
        clr_plt = [random.randint(0, 256) for i in range(mx)]
        # print(len(jsn))
        while i < mx:
            pts = jsn.iloc[i].value["points"]

            # print(len(pts))
            x_o = jsn.iloc[i].original_width
            y_o = jsn.iloc[i].original_height
            # print(x_o, y_o)
            for j in range(len(pts)):
                pts[j][0] *= x_o / 100.0
                pts[j][1] *= y_o / 100.0
            name.append(jsn.iloc[i].value["polygonlabels"])
            i += 1

            pts_.append(pts)
        for c, pts in enumerate(pts_):
            pts = np.array(pts)
            pts = np.int32(pts)

            pts = pts.reshape((-1, 1, 2))
            image = cv.fillPoly(image, [pts], color=np.array(cm.jet(clr_plt[c])) * 255)

        # Following line overlays transparent rectangle over the image
        if not get_bbox:
            image = cv.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        for c, pts in enumerate(pts_):
            pts = np.array(pts)
            pts = np.int32(pts)

            mx_coord, mn_coord = get_bbox_coord(pts)
            pts = pts.reshape((-1, 1, 2))

            if get_bbox:
                image = cv.rectangle(
                    image,
                    tuple(mn_coord),
                    tuple(mx_coord),
                    color=np.array(cm.jet(clr_plt[c])) * 255,
                    thickness=2,
                )
                image = cv.polylines(
                    image,
                    [pts],
                    color=np.array(cm.jet(clr_plt[c])) * 255,
                    thickness=1,
                    isClosed=False,
                )
                image = cv.polylines(
                    image,
                    [pts],
                    color=np.array(cm.jet(clr_plt[c])) * 255,
                    thickness=1,
                    isClosed=True,
                )
            else:
                image = cv.polylines(
                    image,
                    [pts],
                    color=np.array(cm.jet(clr_plt[c])) * 255,
                    thickness=2,
                    isClosed=False,
                )
                image = cv.polylines(
                    image,
                    [pts],
                    color=np.array(cm.jet(clr_plt[c])) * 255,
                    thickness=2,
                    isClosed=True,
                )

        if get_bbox:
            image = cv.addWeighted(overlay, 0.5, image, 0.5, 0)

            for c, (pts, img_text) in enumerate(zip(pts_, name)):

                pts = np.array(pts)
                pts = np.int32(pts)
                mx_coord, mn_coord = get_bbox_coord(pts)

                image = cv.putText(
                    image,
                    img_text[0],
                    color=np.array(cm.jet(clr_plt[c])) * 255,
                    org=tuple(mn_coord),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.55,
                    thickness=2,
                )

        # show the image, provide window name first
        cv2_imshow(image)
        # add wait key. window waits until user presses a key
        # cv.waitKey(0)
        # and finally destroy/close all open windows
        cv.destroyAllWindows()

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
