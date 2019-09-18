from utils import mkdir, scan_folder, scan_file, cropImg, drawBoxes

dataset_path = "label/"
_, subfolder_list = scan_file(dataset_path, 'txt')
subfolder_list.sort()


print(subfolder_list)