from paddleocr import PaddleOCR
import cv2 as cv
import os
from PIL import Image



def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory
# 选择你要识别的图片ds路径
img_path = r"Images"

for file in os.listdir(img_path):

    FILE_PATH = os.path.join(img_path,file)
    result = ocr.ocr(FILE_PATH, cls=True)
    line = result[0]

    # 显示结果

    image = cv.imread(FILE_PATH)
    rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    boxes = [rect[0] for rect in line]
    txts = [rect[1][0] for rect in line]
    scores = [rect[1][1] for rect in line]

    for txt,box in zip(txts,boxes) :
        cv.rectangle(image,(int(box[0][0]),int(box[0][1])),(int(box[2][0]),int(box[2][1])),(0,255,0),2)
        cv.putText(image,txt,(int(box[0][0]),int(box[0][1])),cv.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)




    cv.namedWindow('finalImg', cv.WINDOW_NORMAL)
    cv.imshow("finalImg",image)

    while True:
        if cv.waitKey(0) & 0xFF == ord('q'):
            break
    