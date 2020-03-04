# 这里基于tensorflow，pytorch实现对自然场景的文字检测及端到端的OCR中文文字识别

## 注意参考了https://github.com/xiaofengShi/CHINESE-OCR

## 还在更新
ctnp_ocr_test.ipynb文件是运行的Demo，运行前需要下载模型，并放入指定文件夹下（可查看CHINESE-OCR-API\img）。

chinese_ocr_api\ctpn目录下：
python train_ctpn.py可开始训练CTPN

chinese_ocr_api\lib\datasets\pascal_voc.py
下需要确认训练数据的路径self._devkit_path = './data/VOCdevkit2007'

训练后在chinese_ocr_api下会产生几个文件夹，其中data/下有一个cache/voc_2007_trainval_gt_roidb.pkl文件，如果换新数据重新训练需要先删除该缓存文件不然会报（KeyError: 'max_overlaps'）的错。

数据集预处理
下载icdar2017rctw_train_v1.2数据集，可转为训练需要的数据格式，在放入指定文件夹下，就可以扩充训练数据。如果有特殊要求，可以用labelImg自己标一些数据。

