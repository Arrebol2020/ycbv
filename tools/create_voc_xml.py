import os
import json
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_json(json_path):
    with open(json_path, 'r') as f:
        photo_info = json.load(f)

    return photo_info


def make_voc_dir():
    """
    创建voc的相关目录
    :return:
    """
    os.makedirs(os.path.join(BASE_DIR, "..", "fasterrcnn", "data", "VOCdevkit2007", 'VOC2007', 'Annotations'))
    os.makedirs(os.path.join(BASE_DIR, "..", "fasterrcnn", "data", "VOCdevkit2007", 'VOC2007', 'ImageSets'))
    os.makedirs(os.path.join(BASE_DIR, "..", "fasterrcnn", "data", "VOCdevkit2007", 'VOC2007', 'ImageSets/Main'))
    os.makedirs(os.path.join(BASE_DIR, "..", "fasterrcnn", "data", "VOCdevkit2007", 'VOC2007', 'Images'))


def create_xml(image_name, bboxes, objs, channel=3,
               xml_root="../VOC2007/Annotations", width=640, height=480):
    """
    创建xml文件
    :param image_name: 图像文件名称
    :param bboxes: 该图像中bounding box集合
    :param objs: 该图像中的类别集合
    :param channel: 该图像的通道数
    :param xml_root: 保存xml的根目录
    :param width: 图像的宽度
    :param height: 图像的高度
    :return:
    """
    from lxml.etree import Element, SubElement, tostring

    random.seed(2021)

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'Images'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    for i, bbox in enumerate(bboxes):
        if -1 not in bbox and 0 not in bbox:
            # 左上角的点坐标、w、h
            x, y, w, h = bbox
            left, top, right, bottom = x, y, x + w, y + h
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = str(objs[i])
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = '%s' % left
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = '%s' % top
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = '%s' % right
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = '%s' % bottom

    xml = tostring(node_root, pretty_print=True)

    save_xml = os.path.join(xml_root, image_name.replace(image_name[image_name.find(".")+1:], 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)


def create_xmls(data_path, is_for_train=True):
    """
    为data path下的所有图像文件创建xml
    :param data_path: 数据路径
    :param is_for_train: 使用是用于训练
    :return:
    """
    for i, file in enumerate(os.listdir(data_path)):  # 遍历data_path下的所有子文件夹
        # 拼接成完整的路径
        rgb_path = os.path.join(os.path.join(data_path, file), 'rgb')
        scene_gt_info_path = os.path.join(os.path.join(data_path, file), "scene_gt_info.json")
        scene_gt_path = os.path.join(os.path.join(data_path, file), "scene_gt.json")

        photos_path = os.listdir(rgb_path)  # 读取rgb文件夹下的图片
        bboxes_info = parse_json(scene_gt_info_path)
        objs_info = parse_json(scene_gt_path)

        if is_for_train:
            for j in range(0, len(photos_path), 10):
                index = j + random.randint(0, 9)  # 在10张中随机选择一张

                if index < len(photos_path):
                    print("当前：", file, photos_path[index])
                    photo_path = os.path.join(rgb_path, photos_path[index])  # 图片的路径

                    bboxes = bboxes_info[str(int(photos_path[index][:-4]))]  # 获得当前图像中被识别物体的位置
                    objs = objs_info[str(int(photos_path[index][:-4]))]  # 获取图像中被识别物体的id

                    bounding_boxes = []
                    objects = []
                    object_ids = ""
                    for k, bbox in enumerate(bboxes):
                        bounding_boxes.append(bbox['bbox_visib'])
                        objects.append(objs[k]['obj_id'])
                        object_ids += str(objs[k]['obj_id']) + "_"

                    new_photo_path = os.path.join(annotations_path, "..", "Images",
                                                  object_ids + 'tr_' + file + '_' + photos_path[index])  # 新的图片路径
                    shutil.copy(photo_path, new_photo_path)  # 复制图片

                    create_xml(object_ids + 'tr_' + file + '_' + photos_path[index],
                               bounding_boxes, objects, xml_root=annotations_path)
        else:
            for j in range(len(photos_path)):
                print("当前：", file, photos_path[j])
                photo_path = os.path.join(rgb_path, photos_path[j])  # 图片的路径

                # +1 是因为json中的字典的key是从1开始的
                bboxes = bboxes_info[str(int(photos_path[j][:-4]))]  # 获得当前图像中被识别物体的位置
                objs = objs_info[str(int(photos_path[j][:-4]))]  # 获取图像中被识别物体的id

                bounding_boxes = []
                objects = []
                object_ids = ""
                for k, bbox in enumerate(bboxes):
                    bounding_boxes.append(bbox['bbox_visib'])
                    objects.append(objs[k]['obj_id'])
                    object_ids += str(objs[k]['obj_id']) + "_"

                new_photo_path = os.path.join(annotations_path, "..", "Images",
                                              object_ids + 'te_' + file + '_' + photos_path[j])  # 新的图片路径
                shutil.copy(photo_path, new_photo_path)

                create_xml(object_ids + 'te_' + file + '_' + photos_path[j],
                           bounding_boxes, objects, xml_root=annotations_path)


if __name__ == "__main__":
    make_voc_dir()  # 创建相关文件夹
    # 相关的路径
    annotations_path = os.path.join(BASE_DIR, "..", "fasterrcnn", "data", "VOCdevkit2007", "VOC2007", "Annotations")
    path_train_real = os.path.join(BASE_DIR, "..", "data", "train_real")
    path_test19 = os.path.join(BASE_DIR, "..", "data", "ycbv_test_bop19")

    create_xmls(data_path=path_test19, is_for_train=False)  # 创建测试集的xml
    create_xmls(data_path=path_train_real, is_for_train=True)  # 创建训练集的xml

