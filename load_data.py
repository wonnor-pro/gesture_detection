from utils import mkdir, scan_folder, scan_file, cropImg, drawBoxes
import os
import numpy as np
import cv2


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

def sort_data(label_path = "label/", dataset_path = "dataset_sorted/"):

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

        txt_path = label_path + txt_filename

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
                    x1 = int(float(words[1]) * w)
                    y1 = int(float(words[2]) * h)
                    x2 = int(float(words[3]) * w)
                    y2 = int(float(words[4]) * h)
                    cropped_img = cropImg(im, x1, y1, x2, y2)
                    resized_img = cv2.resize(cropped_img, (28, 28))
                    # print(resized_img.shape)
                    grayFrame = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                    # print(grayFrame.shape)
                    # cv2.imshow("gray", grayFrame)
                    # cv2.waitKey(0)
                    new_img_path = "{}/{}.png".format(subfolder_path, count_line)
                    cv2.imwrite(new_img_path, grayFrame)
                    count_img[count] += 1
                    single_record = "{} {}\n".format(new_img_path, count)
                    print(single_record)

                    isExists_new = os.path.exists(new_img_path)
                    if isExists_new:
                        record.append(single_record)
                        count_sample[count] += 1
                    else:
                        print("new_img_path invalid")

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

