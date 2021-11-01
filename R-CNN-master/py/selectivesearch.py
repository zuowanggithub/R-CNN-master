# -*- coding: utf-8 -*-

"""
@author: zj
@file:   selectivesearch.py
@time:   2020-02-25
"""

import sys
import cv2


def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()   # 创建选择性搜索分割对象
    return gs


def config(gs, img, strategy='q'):
    gs.setBaseImage(img)   # 设置输入图像，我们将运行分割

    if (strategy == 's'):
        gs.switchToSingleStrategy()
    elif (strategy == 'f'):  # 快速但低召回选择性搜索方法
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):  # 高召回但慢选择性搜索方法
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()  # 运行选择性搜索分割输入图像
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects


if __name__ == '__main__':
    """
    选择性搜索算法操作
    """
    gs = get_selective_search()

    #img = cv2.imread('./data/lena.jpg', cv2.IMREAD_COLOR)
    img = cv2.imread(r'../imgs/000012.jpg', cv2.IMREAD_COLOR)
    config(gs, img, strategy='q')

    rects = get_rects(gs)
    print(rects)  # 候选区域建议框
    print('Total Number of Region Proposals: {}'.format(len(rects)))

