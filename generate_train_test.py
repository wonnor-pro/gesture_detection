import random
import numpy as np
from utils import mkdir, scan_folder, scan_file, cropImg, drawBoxes
import os

def get_train_test_list(dataset_path = "dataset_sorted_aug/", testdata_amount = 500):
    # dataset_path = "dataset_sorted/"
    # testdata_amount = 500

    subfolder_list = scan_folder(dataset_path)
    sample_amount = np.zeros(len(subfolder_list))
    subfolder_list.sort()

    for count, subfolder in enumerate(subfolder_list):
        subfolder_path = dataset_path + subfolder
        file_amount, _ = scan_file(subfolder_path, 'png')
        sample_amount[count] = file_amount
        print(subfolder, file_amount)

    print(sample_amount)

    indexList = []

    for count, subfolder in enumerate(subfolder_list):
        random.seed(count)
        img_amount = sample_amount[count]
        totalList = np.arange(img_amount)
        testList = random.sample(range(0,int(img_amount)),testdata_amount)
        assert testdata_amount < img_amount, "test data should be less than total data"
        testList.sort()
        testList = np.array(testList)
        trainList = np.setdiff1d(totalList, testList)

        index = np.array([testList, trainList])
        indexList.append(index)

    indexList = np.array(indexList)
    print(indexList.shape)

    for i in indexList:
        print(len(i[0]), len(i[1]), len(i[0])+len(i[1]))
    print(sample_amount)

    train_files = []
    train_labels = []
    test_files = []
    test_labels = []

    for subfolder in range(len(indexList)):
        subfolder_path = dataset_path + str(subfolder)
        for img in indexList[subfolder][0]:
            img_path = "{}/{}.png".format(subfolder_path, int(img))
            isExists = os.path.exists(img_path)
            if isExists:
                test_files.append(img_path)
                test_labels.append(subfolder)
            else:
                raise RuntimeError("Wrong test index list")

    for subfolder in range(len(indexList)):
        subfolder_path = dataset_path + str(subfolder)
        for img in indexList[subfolder][1]:
            img_path = "{}/{}.png".format(subfolder_path, int(img))
            isExists = os.path.exists(img_path)
            if isExists:
                train_files.append(img_path)
                train_labels.append(subfolder)
            else:
                raise RuntimeError("Wrong train index list")


    return train_files, train_labels, test_files, test_labels




def write_txt(subfolder_list, indexList, dataset_path):

    train_txt_path = "train.txt"
    test_txt_path = "test.txt"

    write_amount_train = np.zeros(len(subfolder_list))
    write_amount_test = np.zeros(len(subfolder_list))

    with open(test_txt_path, 'w') as f:
        for subfolder in range(len(indexList)):
            subfolder_path = dataset_path + str(subfolder)
            for img in indexList[subfolder][0]:
                img_path = "{}/{}.png".format(subfolder_path, int(img))
                single_record = "{} {}\n".format(img_path, subfolder)
                isExists = os.path.exists(img_path)
                if isExists:
                    f.write(single_record)
                    write_amount_test[subfolder] += 1
                else:
                    print("{} invalid".format(single_record))

    with open(train_txt_path, 'w') as f:
        for subfolder in range(len(indexList)):
            subfolder_path = dataset_path + str(subfolder)
            for img in indexList[subfolder][1]:
                img_path = "{}/{}.png".format(subfolder_path, int(img))
                single_record = "{} {}\n".format(img_path, subfolder)
                isExists = os.path.exists(img_path)
                if isExists:
                    f.write(single_record)
                    write_amount_train[subfolder] += 1
                else:
                    print("{} invalid".format(single_record))

