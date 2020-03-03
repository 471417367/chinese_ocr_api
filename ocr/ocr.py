import torch
import torch.utils.data
from torch.autograd import Variable
import ocr.model_net as mn
import ocr.labels_ocr


class CrnnOcr:
    def __init__(self, path):
        alphabet = ocr.labels_ocr.alphabet
        self.converter = mn.strLabelConverter(alphabet)
        self.model = mn.CRNN(32, 1, len(alphabet) + 1, 256).cpu()
        # path = 'ocr/model/model_acc97.pth'
        self.model.eval()
        self.model.load_state_dict(torch.load(path))

    def predict(self, image):
        image = image.view(1, *image.size())
        image = Variable(image)
        preds = self.model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        if len(sim_pred) > 0:
            if sim_pred[0] == u'-':
                sim_pred = sim_pred[1:]

        return sim_pred
