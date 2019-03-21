# -*- coding: utf-8 -*-
import os, shutil

file_set = []
width = 224
height = 224
root_dir = "D:\\processedData"
target_dir = "D:\\classifiedData"

unsolved_list = os.listdir(root_dir)
brand_type_name = {}
while len(unsolved_list) > 0:
    cur_name = unsolved_list.pop()
    unsolved_path = os.path.join(root_dir, cur_name)
    if os.path.isfile(unsolved_path):
        continue
    else:
        inside_unsolved_list = os.listdir(unsolved_path)
        if len(inside_unsolved_list) > 0:
            tmp_path = unsolved_path + os.sep + inside_unsolved_list[0]
            if os.path.isfile(tmp_path):
                type_name = os.path.dirname(tmp_path)
                type_name = type_name[type_name.rfind(os.sep) + 1:]
                if not type_name.find('（') == -1:
                    type_name = type_name[0:type_name.find('（')]
                if not type_name.find('(') == -1:
                    type_name = type_name[0:type_name.find('(')]
                if type_name not in brand_type_name:
                    brand_type_name[type_name] = unsolved_path.replace(root_dir, target_dir)
                    os.makedirs(brand_type_name[type_name], exist_ok=True)
                    for file in inside_unsolved_list:
                        file_path = unsolved_path + os.sep + file
                        if os.path.isfile(file_path):
                            shutil.copy(file_path, brand_type_name[type_name] + os.sep + file)
                        else:
                            for name in os.listdir(file_path):
                                inside_unsolved_list.append(unsolved_path + os.sep + file + os.sep + name)
                # shutil.copytree(unsolved_path, brand_type_name[type_name])
                # shutil.rmtree(unsolved_path)
            else:
                for inside_unsolved_path in inside_unsolved_list:
                    relative_path = cur_name + os.sep + inside_unsolved_path
                    unsolved_list.append(relative_path)
        else:
            continue
