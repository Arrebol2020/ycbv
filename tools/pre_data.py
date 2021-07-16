import json
import os
import random
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

"""
获得train_real，获得子文件夹1
获得子文件夹1中rgb文件夹下的图片数量
    每隔10个图片随机获得一张
    读取子文件夹1中scene_gt_info.json 获得类别位置
    对每个类别进行裁剪
    读取子文件夹1中scene_gt.json，类别id
    保存到新的文件夹汇总
"""

# https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md


def parse_json(json_path):
    with open(json_path, 'r') as f:
        photo_info = json.load(f)

    return photo_info


if __name__ == "__main__":

    random.seed(2021)

    path_train_real = os.path.join(BASE_DIR, "..", "data", "train_real")
    path_train = os.path.join(path_train_real, "train")

    for i, file in enumerate(os.listdir(path_train_real)):  # 遍历train_real下的所有子文件夹
        if "train" not in file and "test" not in file:  # 将train和test文件夹排除
            # 拼接成完整的路径
            rgb_path = os.path.join(os.path.join(path_train_real, file), 'rgb')
            scene_gt_info_path = os.path.join(os.path.join(path_train_real, file), "scene_gt_info.json")
            scene_gt_path = os.path.join(os.path.join(path_train_real, file), "scene_gt.json")

            photos_path = os.listdir(rgb_path)  # 读取rgb文件夹下的图片
            bboxes_info = parse_json(scene_gt_info_path)
            objs_info = parse_json(scene_gt_path)

            for j in range(0, len(photos_path), 10):
                index = j + random.randint(0, 9)  # 在10张中随机选择一张

                if index < len(photos_path):
                    photo_path = os.path.join(rgb_path, photos_path[index])  # 图片的路径
                    # +1 是因为json中的字典的key是从1开始的
                    bboxes = bboxes_info[str(index + 1)]  # 获得当前图像中被识别物体的位置
                    objs = objs_info[str(index + 1)]  # 获取挡墙图像中被识别物体的id

                    for k, bbox in enumerate(bboxes):
                        img = cv2.imread(photo_path)  # 读取图片
                        bbox = bbox['bbox_visib']  # 获取bounding box，左上角的点坐标、w、h
                        if -1 not in bbox and 0 not in bbox:
                            img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]  # 裁剪图片
                            obj = objs[k]['obj_id']  # 裁剪区域图片的id

                            path_obj = os.path.join(path_train, str(obj))

                            print(photo_path, bbox, obj, bbox.count(-1))

                            if not os.path.exists(path_obj):  # 若该分类没有文件夹，则创建
                                os.makedirs(path_obj)

                            cv2.imwrite(os.path.join(path_obj, file + '_' + photos_path[index]), img)  # 保存图片


