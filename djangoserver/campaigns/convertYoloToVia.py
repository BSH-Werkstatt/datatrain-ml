import os
import sys
import json
from PIL import Image

if len(sys.argv) < 1:
    print("Please define the directory where the text files are.")
    exit(1)

directory = sys.argv[1]

filepaths = []

for root, directories, files in os.walk(directory):
    for file in files:
        if '.txt' in file:
            filepaths.append(os.path.join(root, file))

via = {}
for filepath in filepaths:
    f = open(filepath, "r")
    line = f.readline()
    values = line.strip().split(" ")

    img_name = os.path.splitext(os.path.basename(filepath))[0] + ".JPEG"
    img_path = filepath[:-4] + ".JPEG"
    img_size = os.stat(img_path).st_size
    object_class = "Washing Machine" if values[0] == 0 else "Control Panel"

    im = Image.open(img_path)
    width, height = im.size

    x1 = round((float(values[1]) - float(values[3])/2) * width)
    if x1 < 0:
        x1 = 0

    x2 = round((float(values[1]) + float(values[3])/2) * width)
    if x2 > width - 1:
        x2 = width - 1

    y1 = round((float(values[2]) - float(values[4])/2) * height)
    if y1 < 0:
        y1 = 0

    y2 = round((float(values[2]) + float(values[4])/2) * height)
    if y2 > width - 1:
        y2 = width - 1

    via[img_name+str(img_size)] = {
        "size": img_size,
        "filename": img_name,
        "source": "washingMachines",
        "regions": [
            {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [
                        x1,
                        x2,
                        x2,
                        x1
                    ],
                    "all_points_y": [
                        y1,
                        y1,
                        y2,
                        y2
                    ]
                },
                "region_attributes": {
                    "name": object_class
                }
            }
        ],
        "file_attributes": {}
    }

    f.close()

filename = os.path.join(directory, "via_region_data.json")

if filename:
    with open(filename, 'w') as f:
        json.dump(via, f, indent=4)
        f.close()