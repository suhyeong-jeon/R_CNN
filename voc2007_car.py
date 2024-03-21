import os
import shutil
import numpy as np

voc2007_train_path = './VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/car_train.txt'
voc2007_val_path = './VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/car_val.txt'

voc2007_images_path = './VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
voc2007_annotations_path = './VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations'

my_voc2007_train_origin_path = './my_voc2007/voc_car/train'
my_voc2007_val_origin_path = './my_voc2007/voc_car/val'
my_voc2007_images_train_path = './my_voc2007/voc_car/train/JPEGImages'
my_voc2007_images_val_path = './my_voc2007//voc_car/val/JPEGImages'
my_voc2007_annotations_train_path = './my_voc2007/voc_car/train/Annotations'
my_voc2007_annotations_val_path = './my_voc2007/voc_car/val/Annotations'

car_train_array = []
car_val_array = []

os.makedirs(my_voc2007_images_train_path, exist_ok = True)
os.makedirs(my_voc2007_images_val_path, exist_ok = True)
os.makedirs(my_voc2007_annotations_train_path, exist_ok = True)
os.makedirs(my_voc2007_annotations_val_path, exist_ok = True)

def read_txt(voc2007_annotations, my_data_train_origin, my_data_val_origin, train = True):
    with open(voc2007_annotations, 'r') as f:
        for line in f:
            f_data = line.strip().split(' ')
            f_data = ' '.join(f_data).split()
            if train:
                car_train_array.append(f_data)
            else:
                car_val_array.append(f_data)


def copy_files(car_array, voc2007_image_path, voc2007_annotation_path, original_path, my_data_image_train, my_data_image_val, my_data_annotation_train, my_data_annotation_val, train = True):

    print(f"--- Data sorting started... ---")

    data = []

    for array in car_array:
        if array[1] == '1':
            image_name = array[0]
            image_path = f'{voc2007_image_path}/{image_name}.jpg'

            data.append(image_name)

            annotation_path = f'{voc2007_annotation_path}/{image_name}.xml'

            if train:
                shutil.copyfile(image_path, f"{my_data_image_train}/{image_name}.jpg")
                shutil.copyfile(annotation_path, f"{my_data_annotation_train}/{image_name}.xml")


            else:
                shutil.copyfile(image_path, f"{my_data_image_val}/{image_name}.jpg")
                shutil.copyfile(annotation_path, f"{my_data_annotation_val}/{image_name}.xml")

    csv_path = os.path.join(original_path, 'car.csv')
    np.savetxt(csv_path, np.array(data), fmt='%s')




    print("--- Data sorting finished... ---")


if __name__ == '__main__':
    read_txt(voc2007_train_path, my_voc2007_train_origin_path, my_voc2007_val_origin_path, train = True)
    read_txt(voc2007_val_path, my_voc2007_train_origin_path, my_voc2007_val_origin_path, train = False)

    copy_files(car_train_array, voc2007_images_path, voc2007_annotations_path, my_voc2007_train_origin_path,my_voc2007_images_train_path, my_voc2007_images_val_path,
               my_voc2007_annotations_train_path, my_voc2007_annotations_val_path, train = True)

    copy_files(car_val_array, voc2007_images_path, voc2007_annotations_path, my_voc2007_val_origin_path, my_voc2007_images_train_path, my_voc2007_images_val_path,
               my_voc2007_annotations_train_path, my_voc2007_annotations_val_path, train = False)
