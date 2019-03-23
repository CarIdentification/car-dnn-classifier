# -*- coding: utf-8 -*-
import os

file_set = []

root_dir = "D:\\trainData\\trainData"
unsolved_list = os.listdir(root_dir)

brand_count = {}


def travel_dir(cur_dir):
    dir_list = os.listdir(cur_dir)
    cur_dir_file_count = 0
    for element in dir_list:
        element_path = cur_dir + os.sep + element
        # 是文件就计数
        if os.path.isfile(element_path):
            cur_dir_file_count = cur_dir_file_count + 1
        else:
            travel_dir(element_path)
    if cur_dir_file_count != 0:
        brand_name = cur_dir[len(root_dir) + 1:]
        # print("%s = %d" % (brand_name, cur_dir_file_count))
        brand_count[brand_name] = cur_dir_file_count
        if len(brand_count) % 50 == 0:
            print(len(brand_count))


travel_dir(root_dir)
print('traveled total:%d' % (len(brand_count)))
sorted_list = sorted(brand_count.items(), key=lambda d: d[1], reverse=True)
print('sorted')
for key, values in sorted_list:
    print(key, values)
