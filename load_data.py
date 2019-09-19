from utils import mkdir, scan_folder, scan_file, cropImg, drawBoxes
import os
import numpy as np
import cv2
import random
import time


def validate(annotation_path = "annotation.txt"):
    # annotation_path
    invalid = []
    with open(annotation_path, 'rt') as f:
        for count_line, line in enumerate(f):
            words = line.split()

            img_path = words[0]

            isExists = os.path.exists(img_path)

            if not isExists:
                print(img_path, "invalid")
                invalid.append(img_path)

    return invalid

def sort_data(label_path = "label/", dataset_path = "dataset_sorted_aug/"):

    # Existing Paths
    # label_path

    # New Paths
    # dataset_path
    mkdir(dataset_path)

    # Scan Files
    class_amount, txt_list = scan_file(label_path, "txt")
    txt_list.sort()
    print(txt_list)

    count_sample = np.zeros(class_amount)
    count_img = np.zeros(class_amount)
    invalid_path = []
    record = []

    for count, txt_filename in enumerate(txt_list):
        label = ""
        for i in range(class_amount):
            if i != count:
                label += "0 "
            else:
                label += "1 "

        subfolder_path = dataset_path + str(count)
        mkdir(subfolder_path)

        deviation = [[1, 1], [1, -1], [-1, -1], [1, -1]]
        txt_path = label_path + txt_filename
        MAX_X = 640
        MAX_Y = 480

        with open(txt_path, 'rt') as f:
            print(txt_filename)
            for count_line, line in enumerate(f):
                words = line.split()

                img_path = words[0]
                # print("address:", img_path)

                isExists = os.path.exists(img_path)


                if isExists:
                    im = cv2.imread(img_path)
                    h = im.shape[0]
                    w = im.shape[1]
                    random.seed(time)
                    direction = random.choice(deviation)
                    dx = random.randrange(1) * direction[0]
                    dy = random.randrange(1) * direction[1]
                    x1 = int(float(words[1]) * w)
                    y1 = int(float(words[2]) * h)
                    x2 = int(float(words[3]) * w)
                    y2 = int(float(words[4]) * h)
                    bw = x2 - x1
                    bh = y2 - y1
                    _x1 = int(float(words[1]) * w - 0.1 * bw + dx)
                    _y1 = int(float(words[2]) * h - 0.1 * bh + dy)
                    _x2 = int(float(words[3]) * w + 0.1 * bw + dx)
                    _y2 = int(float(words[4]) * h + 0.1 * bh + dy)
                    if _x1<0:
                        _x1 = 0
                    elif _x1>MAX_X:
                        _x1 = MAX_X
                    if _x2<0:
                        _x2 = 0
                    elif _x2>MAX_X:
                        _x2 = MAX_X
                    if _y1<0:
                        _y1 = 0
                    elif _y1>MAX_Y:
                        _y1 = MAX_Y
                    if _y1<0:
                        _y1 = 0
                    elif _y1>MAX_Y:
                        _y1 = MAX_Y
                    cropped_img = cropImg(im, _x1, _y1, _x2, _y2)
                    resized_img = cv2.resize(cropped_img, (28, 28))
                    # print(resized_img.shape)
                    grayFrame = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                    flipped = cv2.flip(grayFrame, 1)
                    # print(grayFrame.shape)
                    # cv2.imshow("gray", grayFrame)
                    # cv2.waitKey(0)
                    # cv2.imshow("flip", flipped)
                    # cv2.waitKey(0)
                    new_img_path_1 = "{}/{}.png".format(subfolder_path, 2*count_line)
                    new_img_path_2 = "{}/{}.png".format(subfolder_path, 2*count_line+1)
                    cv2.imwrite(new_img_path_1, grayFrame)
                    cv2.imwrite(new_img_path_2, flipped)
                    count_img[count] += 2
                    single_record_1 = "{} {}\n".format(new_img_path_1, count)
                    single_record_2 = "{} {}\n".format(new_img_path_2, count)
                    print(single_record_1)
                    print(single_record_2)

                    isExists_new = os.path.exists(new_img_path_1)
                    if isExists_new:
                        record.append(single_record_1)
                        count_sample[count] += 1
                    else:
                        print("new_img_path_1 invalid")

                    isExists_new = os.path.exists(new_img_path_2)
                    if isExists_new:
                        record.append(single_record_2)
                        count_sample[count] += 1
                    else:
                        print("new_img_path_2 invalid")

                else:
                    invalid_path.append(img_path)

    annotation_path = "annotation.txt"
    invalid_annotation_path = "invalid_path.txt"

    with open(annotation_path, 'w') as f:
        for single_record in record:
            f.write(single_record)
    with open(invalid_annotation_path, 'w') as f:
        for img_invalid_path in invalid_path:
            f.write(img_invalid_path)

    print(count_sample)
    print(count_img)

sort_data()

