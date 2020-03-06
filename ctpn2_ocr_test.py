from ctpn2.ctpn2 import CTPN_model2
from PIL import Image
import cv2
import time
from IPython.display import display
from ocr.ocr import CrnnOcr
import ocr.model_net as mn

ctpnmodel = CTPN_model2('ctpn2/checkpoints_mlt/')

crnn = CrnnOcr('ocr/model/model_acc97.pth')

im = cv2.imread('test/hjzm.jpg')[:, :, ::-1]
img, (rh, rw) = ctpnmodel.resize_image(im)

boxes = ctpnmodel.predict(img)

textall = ''
start = time.time()
for rec in boxes:
    xmin = min(rec[0], rec[6])
    xmax = max(rec[2], rec[4])
    ymin = min(rec[1], rec[3])
    ymax = max(rec[7], rec[5])
    partImg = img[ymin:ymax, xmin:xmax]
    image = Image.fromarray(partImg).convert('L')
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    transformer = mn.resizeNormalize((w, 32))
    image = transformer(image).cpu()

    text = crnn.predict(image)
    if len(text) > 2:
        print(text)
        textall = textall + text
        display(Image.fromarray(partImg))
print(time.time() - start)
print(textall)
