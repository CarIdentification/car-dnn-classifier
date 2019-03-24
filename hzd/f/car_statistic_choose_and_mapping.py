# -*- coding: utf-8 -*-
import os
import shutil

file_set = []

root_dir = "D:\\trainData\\trainData"
unsolved_list = os.listdir(root_dir)

brand_count = {}


def output_data_and_mapping(items, each_num, target_path):
    print('训练集输出')
    mapper = {}
    cursor = 0
    for i_key, i_value in items:
        mapper[cursor] = i_key
        source_date_path = root_dir + os.sep + i_key
        target_data_path = target_path + os.sep + str(cursor)
        copied_num = 0
        if not os.path.exists(target_data_path):
            os.makedirs(target_data_path)
        for file in os.listdir(source_date_path):
            file_path = source_date_path + os.sep + file
            shutil.copy(file_path, target_data_path + os.sep + file)
            copied_num += 1
            if copied_num == each_num:
                break
        cursor += 1
        if cursor % 30 == 0:
            print('已输出:%d' % cursor)
    print('数据输出完成 写出映射关系')
    out_mapper_file = open(target_path + os.sep + 'mapper', 'w')
    for t_key in mapper:
        out_mapper_file.write('%s:%s\n' % (t_key, mapper[t_key]))
    out_mapper_file.close()


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
while True:
    num_str = input('切分的最少的训练集张数')
    num = int(num_str)
    found_num = 0

    for key, value in sorted_list:
        if value >= num:
            found_num = found_num + 1
        else:
            break
    print('训练集张数至少有 %d 张的车型数为： %d' % (num, found_num))
    will_output_data = input('是否导出数据及生成映射关系? y/n')
    if will_output_data.lower() == 'y':
        output_path = input('输出到的路径')

        output_data_and_mapping(sorted_list[:found_num], num, output_path)
        print('done')
        break
    else:
        print('重新选择')
