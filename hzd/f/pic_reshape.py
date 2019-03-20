import os

from PIL import Image

file_set = []
width = 224
height = 224
root_dir = "D:\\carData\\carData"
target_dir = "D:\\processedData"

unsolved_list = os.listdir(root_dir)
while len(unsolved_list) > 0:
    cur_name = unsolved_list.pop()
    unsolved_path = os.path.join(root_dir, cur_name)
    if os.path.isfile(unsolved_path):
        out_path = os.path.join(target_dir, cur_name)
        lastIndex = out_path.rfind("\\")
        parent_path = out_path[0:lastIndex]
        if not os.path.exists(parent_path):
            os.makedirs(parent_path, exist_ok=True)
        img = Image.open(unsolved_path)
        try:
            new_img = img.resize((width, height), Image.BILINEAR)
            new_img.save(out_path)
        except Exception as e:
            print("Error on resolve file {} info : {}".format(unsolved_path, e))
    else:
        inside_unsolved_list = os.listdir(unsolved_path)
        for inside_unsolved_path in inside_unsolved_list:
            relative_path = cur_name + os.sep + inside_unsolved_path
            unsolved_list.append(relative_path)
