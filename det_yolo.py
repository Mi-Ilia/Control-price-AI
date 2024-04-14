'''Локализация названия и цены с последующим распознаванием текста'''

from ultralytics import YOLO
import cv2
import easyocr
import os
import shutil
import sys

global PATH_model_det
global PATH_img_crop

def detect_yolo(img):
    # загрузка модели
    model = YOLO(PATH_model_det)
    class_names = model.names

    h, w, _ = img.shape
    results = model.predict(img, conf=0.3, show=False)

    boxes = results[0].boxes.xyxy.tolist()
    clss = results[0].boxes.cls.tolist()

    # Iterate through the bounding boxes
    for box, cls in zip(boxes, clss):
        x1, y1, x2, y2 = box

        crop_object = img[int(y1):int(y2), int(x1):int(x2)]

        cv2.imwrite(PATH_img_crop + '//' + str(int(cls)) + '.jpg', crop_object)

def ocr(file_path): # распознавание текста
    reader = easyocr.Reader(['ru'])
    res = reader.readtext(file_path, detail =0) #detail: если поставить 1, то выведется вероятность и координаты текста на картинке
    result = ' '.join(res).strip()
    return result

def prog(PATH_img):

    # создание папки
    if not os.path.exists(PATH_img_crop):
        os.makedirs(PATH_img_crop)
        #print(f'Папка {PATH_img_crop} создана')
    else:
        r=0
        #print(f'Папка {PATH_img_crop} уже существует')

    img = cv2.imread(PATH_img)
    detect_yolo(img)


    text_price_rub = ocr(PATH_img_crop + '//' + '0.jpg') # цена (рубли)
    text_price_cop = ocr(PATH_img_crop + '//' + '3.jpg') # цена (копейки)
    text_name = ocr(PATH_img_crop + '//' + '1.jpg') # название

    text = f'{text_name} {text_price_rub} {text_price_cop}'

    print(f'{text}')

    # удаление папки
    shutil.rmtree(PATH_img_crop)

if __name__ == "__main__":
    PATH_img_crop = 'image_crop'
    filename = 'image.jpg'
    PATH_model_det = 'yolo_8n_price.pt'

    if not filename.isalpha():
        filename = str(filename)

    prog(filename)
