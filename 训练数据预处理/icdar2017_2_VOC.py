import os
import glob
from PIL import Image
import cv2

# 先创建下面的文件夹，把icdar2017rctw_train_v1.2/train/下的所有图片复制到target_img_dir文件夹下
base_dir = "ICDAR2017/VOC2007"

target_img_dir = base_dir + "/" + "JPEGImages/"
target_ann_dir = base_dir + "/" + "Annotations/"
target_set_dir = base_dir + "/" + "ImageSets/"

# 把该路径下的所有.txt文件转转成xml存到target_ann_dir文件夹下
train_txt_dir = "icdar2017rctw_train_v1.2/train/"

img_list = []

for file_name in os.listdir(target_img_dir):
    if file_name.split('.')[-1] == 'jpg':
        img_list.append(file_name)

for idx in range(len(img_list)):
    img_name = target_img_dir + img_list[idx]
    gt_name = train_txt_dir + 'image_' + img_list[idx].split('.')[0].split('_')[1] + '.txt'
    print(img_list)

    gt_obj = open(gt_name, 'rb')

    gt_txt = gt_obj.read()

    gt_split = gt_txt.decode().split('\n')
    # print(str(gt_split))
    img = cv2.imread(img_name)

    im = Image.open(img_name)
    imgwidth, imgheight = im.size

    # write in xml file
    xml_file = open(
        (target_ann_dir + img_list[idx].split('.')[0] + '.xml'), 'w')
    xml_file.write('<?xml version="1.0" ?>\n')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>text</folder>\n')
    xml_file.write('    <filename>' + img_list[idx] + '</filename>\n')
    xml_file.write('    <source>\n')
    xml_file.write('        <database>icdar2017</database>\n')
    xml_file.write('        <annotation>text</annotation>\n')
    xml_file.write('        <flickrid>000000</flickrid>\n')
    xml_file.write('    </source>\n')
    xml_file.write('    <owner>\n')
    xml_file.write('        <width>zhl</width>\n')
    xml_file.write('    </owner>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(imgwidth) + '</width>\n')
    xml_file.write('        <height>' + str(imgheight) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    f = False
    for gt_line in gt_split:
        gt_ind = gt_line.split(',')
        if gt_ind == ['']:
            continue
        xmin = min(gt_ind[0], gt_ind[6])
        ymin = min(gt_ind[1], gt_ind[3])
        xmax = max(gt_ind[2], gt_ind[4])
        ymax = max(gt_ind[5], gt_ind[7])

        xml_file.write('    <object>\n')
        xml_file.write('        <name>text</name>\n')
        xml_file.write('        <pose>none</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>' + str(gt_ind[8]) + '</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(xmin) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(ymin) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(xmax) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(ymax) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')

img_lists = glob.glob(target_ann_dir + '/*.xml')
img_names = []
for item in img_lists:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    img_names.append(temp1)

train_fd = open(target_set_dir + "/Main/trainval.txt", 'w')
for item in img_names:
    train_fd.write(str(item) + '\n')
