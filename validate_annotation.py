import os

annotation_path = "test.txt"
invalid = []
with open(annotation_path, 'rt') as f:
    for count_line, line in enumerate(f):
        words = line.split()

        img_path = words[0]

        isExists = os.path.exists(img_path)

        if not isExists:
            print(img_path, "invalid")
            invalid.append(img_path)

print(invalid)


