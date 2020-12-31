import shutil
from pathlib import Path
from fastai.data.all import *
from fastai.vision.all import *

def train(path, img_size):
    coco_source = Path(path + '/images')
    images, lbl_bbox = get_annotations(path + '/labels/data_mini.json')

    # After get_annotations bboxes have [x1, y1, x2, y2] or ltrb format
    one_batch_training = False

    if one_batch_training:
        images, lbl_bbox = images[:32], lbl_bbox[:32] # one_batch
        Path("/content/coco_sample/one_batch_train_sample").mkdir(parents=True, exist_ok=True)
        for img in images:
            shutil.copy2(Path('/content/coco_sample/train_sample/' + img), Path('/content/coco_sample/one_batch_train_sample/'))
        coco_source = Path('/content/coco_sample/one_batch_train_sample/')

    img2bbox = dict(zip(images, lbl_bbox))


    def custom_splitter(train_pct):
        def fn(name_list):
            train_idx, valid_idx = RandomSplitter(valid_pct=1.0-train_pct)(name_list)
            np.random.shuffle(train_idx)
            train_len = int(len(train_idx) * train_pct)
            return train_idx[0:train_len], valid_idx
        return fn

    coco = DataBlock(blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
                    get_items=get_image_files,
                    splitter=custom_splitter(train_pct=0.95),
                    get_y=[lambda o: img2bbox[o.name][0], lambda o: img2bbox[o.name][1]], 
                    item_tfms=Resize(img_size),
                    batch_tfms=Normalize.from_stats(*imagenet_stats),
                    n_inp=1)


    dls = coco.dataloaders(coco_source, bs=2, device=torch.device('cuda'), shuffle_train=False)
    dsets = coco.datasets(coco_source)

    n_classes = len(dls.vocab)

    inference = False

    model = Model(cfg='/content/yolov5s.yaml', nc=n_classes)
    if use_cuda:
        model.cuda()


    class EvaluatorCallback(Callback):
        def after_pred(self):
            if inference:
                self.learn.yb = tuple()
                return
            labels = []
            for i, l in enumerate(self.yb[1]):
                l = l.unsqueeze(-1)
                l = torch.cat([l, l], dim=1)
                l[:, 0] = i  # add target image index for build_targets()
                labels.append(l)

            labels = torch.cat(labels, dim=0).view(self.yb[0].shape[0], self.yb[0].shape[1], 2)

            res = torch.cat([labels, cast(self.yb[0], Tensor)], dim=2) # bboxes + categories
            res = torch.cat([res[i, :(res[:, :, 1:].sum(dim=2) != 0).sum(dim=1)[i]] for i in range(len(res))]) # remove zero entries (added while padding in collate_fn)
            res[:, 2:] = (res[:, 2:] + 1) / 2 # rescale bboxes from [-1, 1] to [0, 1]
            res[..., 2:] = xyxy2xywh(res[..., 2:])
            self.learn.yb = [res]
        
    hyp['cls'] *= n_classes / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = n_classes  # attach number of classes to model
    model.hyp = hyp
    model.gr = 1.0
    learner = Learner(dls, model, loss_func=partial(compute_loss, model=model), cbs=[EvaluatorCallback()])

    return learner

if __name__ == "__main__":
    print("In main ")
    train('/home/dev/Downloads/data/bdd_tiny', 608)
