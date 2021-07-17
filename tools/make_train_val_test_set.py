import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_txt(val_rate=0.1):
    random.seed(2021)

    xml_path = os.path.join(BASE_DIR, "..", "fasterrcnn", "data", "VOCdevkit2007", "VOC2007", "Annotations")
    train_txt_path = os.path.join(BASE_DIR, "..", "fasterrcnn", "data", "VOCdevkit2007", "VOC2007",
                                  "ImageSets", "Main", "train.txt")
    val_txt_path = os.path.join(BASE_DIR, "..", "fasterrcnn", "data", "VOCdevkit2007", "VOC2007",
                                  "ImageSets", "Main", "val.txt")
    test_txt_path = os.path.join(BASE_DIR, "..", "fasterrcnn", "data", "VOCdevkit2007", "VOC2007",
                                 "ImageSets", "Main", "test.txt")

    xmls = os.listdir(xml_path)
    random.shuffle(xmls)  # 打乱数据

    trainval_xmls = []
    test_xmls = []

    for xml in xmls:
        if "tr" in xml:
            trainval_xmls.append(xml)
        elif "te" in xml:
            test_xmls.append(xml)

    val_data = trainval_xmls[0: int(len(trainval_xmls) * val_rate)]
    train_data = trainval_xmls[int(len(trainval_xmls) * val_rate):]

    with open(train_txt_path, 'w') as f:
        for data in train_data:
            f.write(data[:-4] + '\n')

    with open(val_txt_path, 'w') as f:
        for data in val_data:
            f.write(data[:-4] + '\n')

    with open(test_txt_path, 'w') as f:
        for data in test_xmls:
            f.write(data[:-4] + '\n')


if __name__ == "__main__":
    create_txt(0.1)
