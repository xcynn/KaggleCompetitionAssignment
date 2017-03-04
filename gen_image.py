from PIL import Image
import numpy
import csv
from collections import Counter
import pprint

# Counte occurance of letters
with open("train.csv", "r") as file:
    data = csv.reader(file, delimiter=",")
    counter = Counter()
    count = 0
    for line in data:
        if count == 0:
            count += 1
            continue

        counter.update(line[1])

        count += 1
        if count > 10000:
            pprint.pprint(dict(counter))
            break

def to_image(line, file_name):
    print line
    image_array = [int(i) for i in line[4:]]
    print 'len(image_array) = ', len(image_array)
    array = numpy.array(image_array, dtype=numpy.uint8).reshape(16, 8) * 255
    print array
    img = Image.fromarray(array, mode='L')
    print img
    img.save(file_name)

# Generate image from data array
with open("train.csv", "r") as file:
    data = csv.reader(file, delimiter=",")
    count = 0
    # skip first line
    for line in data:
        if count == 0:
            count += 1
            continue

        to_image(line, 'train_images/%06d-%s-%s.bmp' % (int(line[0]), line[1], line[3]))

        count += 1
        if count > 100:
            break


# Generate image from data array
with open("test.csv", "r") as file:
    data = csv.reader(file, delimiter=",")
    count = 0
    # skip first line
    for line in data:
        if count == 0:
            count += 1
            continue

        to_image(line, 'test_images/%06d-%s-%s.bmp' % (int(line[0]), line[1], line[3]))

        count += 1
        if count > 100:
            break