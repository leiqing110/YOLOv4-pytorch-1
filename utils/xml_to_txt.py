# -*- coding: utf-8 -*-
# @Author  : argus
# @File    : make_train_val_test_set.py
# @Software: PyCharm

import os
import random


def _main():
    xmlfilepath = "G:/dataset/VisDrone2019-DET/Annotations_XML"
    total_xml = os.listdir(xmlfilepath)

    num = len(total_xml)
    list = range(num)

    ftrainval = open(
        "G:/dataset/VisDrone2019-DET/ImageSetsV4/trainval.txt",
        "w",
    )

    for i in list:
        name = total_xml[i][:-4] + "\n"
        ftrainval.write(name)

    ftrainval.close()


if __name__ == "__main__":
    _main()
