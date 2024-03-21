import time
import shutil
import numpy as np
import cv2
import os
import selectivesearch
from utils.util import check_dir
from utils.util import parse_car_csv
from utils.util import parse_xml
from utils.util import compute_ious

"""
1. 이미지에 Selective search 알고리즘을 적용하여 region proposals를 추출. 그리고 해당 이미지에 대한 xml 파일을 읽어들여 ground truth box를 파악.

2. region proposals와 ground truth box를 비교하여 IoU 값을 도출하고 0.5 이상인 sample은 positive_list, 0.5 미만인 sample은 negative_list에 저장.

3. 그리고 이미지별 region proposal에 대한 위치를positive/negative 여부에 따라 서로다른 csv 파일에 저장. 
   예를 들어 1111.jpg 파일에서 positive sample에 해당하는 bounding box의 좌표는 1111_1.csv 파일에, 
   negative sample에 해당하는 bounding box는 1111_0.csv 파일에 저장. 
   
Selective Search 과정이 GPU가 아닌 CPU 동작이여서 시간이 엄청 걸린다.
"""

def parse_annotation_jpeg(annotation_path, jpeg_path, gs):

    img = cv2.imread(jpeg_path)

    selectivesearch.config(gs, img, strategy='q') # 이미지에 Selective search 알고리즘을 적용하여 region proposals를 추출
    rects = selectivesearch.get_rects(gs) # region proposals : 약 2000개의 boundary boxes
    bndboxs = parse_xml(annotation_path) # ground truth boxes : 실제 이미지의 정확한 boundary box

    # get size of the biggest bounding box(region proposals)
    maximum_bndbox_size = 0
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    iou_list = compute_ious(rects, bndboxs) #  region proposals와 ground truth box를 비교하여 IoU 값을 도출

    positive_list = list()
    negative_list = list()

    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]

        # When fine-tuning the pre-trained CNN model
        # positive : iou >= 0.5
        # negative : iou < 0.5
        # Only the bounding box with iou greater than 0.5 is saved
        # 0.5 이상인 sample은 positive_list, 0.5 미만인 sample은 negative_list에 저장
        if iou_score >= 0.5:
            positive_list.append(rects[i])

        # negative : iou < 0.5 And if it is more than 20% of the largest bounding box
        if 0 < iou_list[i] < 0.5 and rect_size > maximum_bndbox_size / 5.0:
            negative_list.append(rects[i])
        else:
            pass

    return positive_list, negative_list


if __name__ == '__main__':
    car_root_dir = './my_voc2007/voc_car'
    finetune_root_dir = './my_voc2007/finetune_car'
    check_dir(finetune_root_dir)

    gs = selectivesearch.get_selective_search()

    for name in ['train', 'val']:
        src_root_dir = os.path.join(car_root_dir, name)
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')

        # AlexNet의 fine tuning을 위한 데이터셋을 젖아하기 위한 path 지정
        dst_root_dir = os.path.join(finetune_root_dir, name)
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
        check_dir(dst_root_dir)
        check_dir(dst_annotation_dir)
        check_dir(dst_jpeg_dir)

        total_num_positive = 0
        total_num_negative = 0

        samples = parse_car_csv(src_root_dir) # csv 데이터를 문자열로 samples에 저장

        src_csv_path = os.path.join(src_root_dir, 'car.csv')
        dst_csv_path = os.path.join(dst_root_dir, 'car.csv')

        shutil.copyfile(src_csv_path, dst_csv_path)

        for sample_name in samples:
            since = time.time()

            src_annotation_path = os.path.join(src_annotation_dir, sample_name + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + '.jpg')

            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)


            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample_name + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation_dir, sample_name + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + '.jpg')

            shutil.copyfile(src_jpeg_path, dst_jpeg_path)

            """
            이미지별 region proposal에 대한 위치를positive/negative 여부에 따라 서로다른 csv 파일에 저장
            1111.jpg 파일에서 positive sample에 해당하는 bounding box의 좌표는 1111_1.csv 파일에,
            negative sample에 해당하는 bounding box는 1111_0.csv 파일에 저장
            """
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')

            time_elapsed = time.time() - since
            print('parse {}.png in {:.0f}m {:.0f}s'.format(sample_name, time_elapsed // 60, time_elapsed % 60))
        print('%s positive num: %d' % (name, total_num_positive))
        print('%s negative num: %d' % (name, total_num_negative))
    print('done')