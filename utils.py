import os
import fnmatch
import cv2
import numpy as np

def mkdir(path):
    path = path.strip()
    path= path.rstrip("/")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' created successfully')
        return True
    else:
        print(path + ' already exist')
        return False

def scan_folder(base_path = ''):
    '''
    This function will scan the folder in the given directory and return the folder
    name list.

    :param basepath: string, should end with '/';
    :return: folder_list: list of folder names;
    '''
    with os.scandir(base_path) as entries:
        folder_list = []
        for entry in entries:
            if entry.is_dir():
                folder_list.append(entry.name)
    return folder_list

def scan_file(file_dir = '', file_postfix = 'jpg'):
    '''
    This function will scan the file in the given directory and return the number
    and file name list for files satisfying the postfix condition.
    :param file_dir: string, should end with '/';
    :param file_type: string, no need for '.';
    :return: file_count: list of file names whose postfix satisfies the condition;
    '''
    file_count = 0
    file_list = []
    for f_name in os.listdir(file_dir):
        if fnmatch.fnmatch(f_name, ('*.' + file_postfix)):
            file_count += 1
            file_list.append(f_name)
    return file_count, file_list

def drawBoxes(im, boxes, color=(0,255,0)):
    x1 = boxes[0]
    y1 = boxes[1]
    x2 = boxes[2]
    y2 = boxes[3]
    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    return im

def cropImg(im, x1, y1, x2, y2):
    img = im.copy()
    box = [x1, y1, x2, y2]
    drawBoxes(img, box)
    # cv2.imshow("original", img)
    # cv2.waitKey(0)
    tmp = im[y1:y2, x1:x2].copy()
    cv2.imshow("result", tmp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return tmp

def test():
    return None

test()
