import shutil
from pathlib import Path
from fastai.data.all import *
from fastai.vision.all import *

from models.yolo import Model
from utils.config import hyp
from utils.loss import *
from utils.utils import custom_splitter, EvaluatorCallback

def get_data_source(path, one_batch_training):
    if one_batch_training:
        images, lbl_bbox = images[:32], lbl_bbox[:32] # one_batch
        Path(path + '/one_batch_train_sample').mkdir(parents=True, exist_ok=True)
        for img in images:
            shutil.copy2(Path(path + '/images/' + img), Path(path + '/one_batch_train_sample/'))
        data_source = Path(path + '/one_batch_train_sample/')
    else:
        data_source = Path(path + '/images')
        images, lbl_bbox = get_annotations(path + '/labels/data_mini.json')

    return data_source, images, lbl_bbox

def create_dataloaders(path, img_size, bs=2, device='cuda', one_batch_training=False):
    data_source, images, lbl_bbox = get_data_source(path, one_batch_training)
    img2bbox = dict(zip(images, lbl_bbox))

    datablock = DataBlock(blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
                          get_items=get_image_files,
                          splitter=custom_splitter(train_pct=0.95),
                          get_y=[lambda o: img2bbox[o.name][0], lambda o: img2bbox[o.name][1]], 
                          item_tfms=Resize(img_size),
                          batch_tfms=Normalize.from_stats(*imagenet_stats),
                          n_inp=1)


    dls = datablock.dataloaders(data_source, bs=bs, device=torch.device(device), shuffle_train=False)
    dsets = datablock.datasets(data_source)

    return dls, dsets

def train(path, img_size, bs=2, one_batch_training=False, path_to_model='./models/yolov5s.yaml'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dls, dsets = create_dataloaders(path, img_size, bs, device, one_batch_training)
    n_classes = len(dls.vocab)

    model = Model(cfg=path_to_model, nc=n_classes)
    if 'cuda' == device:
        model.cuda()
        
    hyp['cls'] *= n_classes / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = n_classes  # attach number of classes to model
    model.hyp = hyp
    model.gr = 1.0
    learner = Learner(dls, model, loss_func=partial(compute_loss, model=model), cbs=[EvaluatorCallback()])

    return learner

if __name__ == "__main__":
    print("In main ")
    train('/home/dev/Downloads/data/bdd_tiny', 608)
