import gc
import json
import math
import pickle
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from os import PathLike, cpu_count
from pathlib import Path
from typing import Any, List, Optional, OrderedDict, Tuple

import albumentations as A
import imageio
import pandas as pd
import timm
import torch
import torch.nn as nn
import yaml
from adabelief_pytorch import AdaBelief
from albumentations.pytorch import ToTensorV2
from kornia.geometry import transform
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from timm.models.layers import Conv2dSame
from timm.models.nfnet import ScaledStdConv2dSame
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from const.flip import id_flip
from const.label_names import id_to_label, label_names
from models.loss import FocalLoss
from models.optim import SAM, Lookahead
from utils import AverageMeter, CustomLogger, make_result_dir, seed_everything, tqdm_kwargs

DATA_DIR = Path("./data")
DATA_NAME = "crop512_9"
N_CLASSES = len(label_names)


@dataclass
class Config:
    # experiment
    exp_num: str = "001"
    ver_num: int = None
    result_dir_root: PathLike = Path("results/exp")
    result_dir: PathLike = None
    seed: int = 867243624
    debug: bool = False

    # network
    model_name: str = "tf_efficientnet_b3_ns"
    checkpoint_path: PathLike = None
    len_sequence: int = 5
    pretrained: bool = True

    # criterion
    criterion: str = "focal"  # ce, focal

    # training
    num_folds: int = 5
    fold: int = 1
    earlystop_limit = 10
    epochs: int = 100
    finetune: bool = True
    finetune_step1_epochs: int = 2
    finetune_step2_epochs: int = 4

    # optimizer / scheduler
    optimizer_name: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 0.01
    scheduler: Any = ReduceLROnPlateau

    # Sharpness-Aware Minimization for Efficiently Improving Generalization [2020]
    sam: bool = False  # no simultaneous with LA, it have higher priority

    # Lookahead Optimizer: k steps forward, 1 step back [2019]
    look_ahead: bool = True
    look_ahead_k: int = 5
    look_ahead_alpha: float = 0.5

    # TODO
    # swa: bool = False
    # swa_start: int = 2
    # swa_lr: float = 1e-5

    # dataoader
    batch_size: int = 50
    num_workers: int = None

    # dataset
    # crop_padding: int = 50
    # input_size: Tuple[int, int] = (512, 512)
    # size_ratio_limit: float = 1.2
    cleared_image: bool = False
    in_memory: bool = False

    def __post_init__(self):
        if self.debug:
            self.epochs = 1
            self.finetune = False

        if self.num_workers is None:
            self.num_workers = self.batch_size * self.len_sequence // 10

    def to_yaml(self, target):
        data = yaml.load(str(self.__dict__), Loader=yaml.FullLoader)
        with open(str(target), "w") as f:
            yaml.dump(data, f)


class Net(nn.Module):
    def __init__(
        self,
        name: str = "resnet",
        pretrained: bool = True,
        n_classes: int = 1000,
        len_sequence: int = 5,
    ):
        super().__init__()

        self.backbone = timm.create_model(name, pretrained=pretrained)

        self.tuning_modules: List[nn.Module] = []

        self.freeze_step = 3

        if "nfnet" in name:
            embedding_size = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(embedding_size, n_classes)
            self.tuning_modules.append(self.backbone.head.fc)

            with torch.no_grad():
                c1: ScaledStdConv2dSame = self.backbone.stem.conv1
                w, b = c1.weight.data, c1.bias.data

                c2 = ScaledStdConv2dSame(3 * len_sequence, c1.out_channels, c1.kernel_size, c1.stride)
                c2.weight.data = torch.repeat_interleave(w, len_sequence, dim=1)
                c2.weight.bias = torch.repeat_interleave(b, len_sequence, dim=0)

                self.backbone.stem.conv1 = c2
                # self.tuning_modules.append(c2)
        else:
            embedding_size = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(embedding_size, n_classes)
            self.tuning_modules.append(self.backbone.classifier)

            with torch.no_grad():
                c1: Conv2dSame = self.backbone.conv_stem
                w = c1.weight.data

                c2 = Conv2dSame(3 * len_sequence, c1.out_channels, c1.kernel_size, c1.stride)
                c2.weight.data = torch.repeat_interleave(w, len_sequence, dim=1)

                self.backbone.conv_stem = c2
                # self.tuning_modules.append(c2)

    def forward(self, x: Tensor):
        return self.backbone(x)

    def freeze(self, step=3):
        if self.freeze_step != step:
            self.freeze_step = step

            if step == 1:
                self.backbone.requires_grad_(False)
                for m in self.tuning_modules:
                    m.requires_grad_(True)
            elif step == 2:
                self.backbone.requires_grad_(True)
                for m in self.tuning_modules:
                    m.requires_grad_(False)
            else:
                self.backbone.requires_grad_(True)


class FileLoader:
    def __init__(self, in_memory: bool, files: List[Path]) -> None:
        self.in_memory = in_memory
        self.files = files

        if self.in_memory:
            self.data = {}
            for file in tqdm(self.files, ncols=100, file=sys.stdout, desc="in-memory loading..."):
                self.data[file] = imageio.imread(file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, file: Path):
        if self.in_memory:
            return self.data[file]
        else:
            return imageio.imread(file)


class GestureDataset(Dataset):
    def __init__(self, items, fileloader: FileLoader, augmentation=False):
        super().__init__()

        self.items = items
        self.fileloader = fileloader
        self.has_label = len(self.items[0]) == 3
        self.augmentation = augmentation

        self._p = lambda: random.random() > 0.5

        t = []
        if augmentation:
            t.append(A.Affine(scale=(0.9, 1.1), translate_px=(-40, 40), rotate=(-15, 15), shear=(-10, 10)))
            t.append(A.GaussianBlur())
            t.append(A.ImageCompression())
        t.append(A.Normalize())
        t.append(ToTensorV2())
        self.transform = A.Compose(transforms=t)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        files = item[0]
        diridx = torch.tensor(item[1], dtype=torch.long)
        # ims = torch.cat([torch.load(file) for file in files])  # (L*3, H, W)
        # ims = torch.cat([self.transform(image=imageio.imread(file))["image"] for file in files])
        ims = torch.cat([self.transform(image=self.fileloader[file])["image"] for file in files])

        if self.has_label:
            label = item[2]

            # flipping
            if self.augmentation:
                ims, label = self.augment(ims, label)

            label = torch.tensor(label, dtype=torch.long)

            return ims, diridx, label
        else:
            return ims, diridx

    def fliplr(self, ims: Tensor, label: int):
        if label in id_flip:
            new_label = id_flip[label]
            new_ims = torch.flip(ims, dims=(2,))
            return new_ims, new_label
        else:
            return ims, label

    def augment(self, ims: Tensor, label: int):
        # fliplr
        if self._p():
            ims, label = self.fliplr(ims, label)

        # affine_mat = torch.zeros(2, 3, dtype=torch.float32)

        # # translation
        # if self._p():
        #     tx = random.random() * 40 - 20
        #     ty = random.random() * 40 - 20
        # else:
        #     tx, ty = 0.0, 0.0

        # # rotation
        # if self._p():
        #     theta = torch.tensor((random.random() * 20 - 10) * math.pi / 180)
        #     c = torch.cos(theta)
        #     s = torch.sin(theta)
        # else:
        #     c, s = 1.0, 0.0

        # # scale
        # if self._p():
        #     scale = 0.9 + random.random() * 0.2
        # else:
        #     scale = 1.0

        # # affine
        # affine_mat = torch.tensor(
        #     [
        #         [c * scale, -s, tx],
        #         [s, c * scale, ty],
        #     ],
        # )
        # ims = kornia.affine(ims, affine_mat, align_corners=True)

        return ims, label


def make_datasets(config: Config):
    # exp022: dataset update: output is tuple (im, diridx, label)
    print("Load datasets ...")

    # load train dataset
    files_train = sorted(list((DATA_DIR / DATA_NAME / "train").glob("*.png")))
    files_test = sorted(list((DATA_DIR / DATA_NAME / "test").glob("*.png")))
    if config.debug:
        files_train = files_train[:1000]

    fileloader = FileLoader(in_memory=config.in_memory, files=files_train + files_test)

    data = defaultdict(list)
    labels = {}
    for file in files_train:
        # train filename: {diridx:3d}_{fileidx:2d}_{label:3d}
        diridx = int(file.stem[:3])
        data[diridx].append(file)
        labels[diridx] = int(file.stem[-3:])

    items_file = []
    items_dir = []
    items_label = []
    for diridx in data:
        if len(data[diridx]) > config.len_sequence:
            for start_idx in range(len(data[diridx]) - config.len_sequence + 1):
                items_file.append(data[diridx][start_idx : start_idx + config.len_sequence])
                items_dir.append(diridx)
                items_label.append(labels[diridx])
        elif len(data[diridx]) == config.len_sequence:
            items_file.append(data[diridx])
            items_dir.append(diridx)
            items_label.append(labels[diridx])
        else:
            fake = [data[diridx][-1] for _ in range(config.len_sequence - len(data[diridx]))]
            items_file.append(data[diridx] + fake)
            items_dir.append(diridx)
            items_label.append(labels[diridx])

    label_cnt = defaultdict(list)
    for i, label in enumerate(items_label):
        label_cnt[label].append(i)

    skf = StratifiedKFold(config.num_folds, shuffle=True, random_state=config.seed)
    tidx, vidx = list(skf.split(items_file, items_label))[config.fold - 1]

    items_train = [(items_file[i], items_dir[i], items_label[i]) for i in tidx]
    items_valid = [(items_file[i], items_dir[i], items_label[i]) for i in vidx]

    ds_train = GestureDataset(items_train, fileloader=fileloader, augmentation=True)
    ds_valid = GestureDataset(items_valid, fileloader=fileloader, augmentation=False)

    dl_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    dl_train = DataLoader(ds_train, **dl_kwargs, shuffle=True)
    dl_valid = DataLoader(ds_valid, **dl_kwargs, shuffle=False)

    # load test dataset
    data = defaultdict(list)
    for file in files_test:
        # test filename: {diridx:3d}_{fileidx:2d}
        diridx = int(file.stem[:3])
        data[diridx].append(file)

    items_test = []
    for diridx in data:
        if len(data[diridx]) > config.len_sequence:
            for i in range(len(data[diridx]) - config.len_sequence):
                items_test.append((data[diridx][i : i + config.len_sequence], diridx))
        elif len(data[diridx]) == config.len_sequence:
            items_test.append((data[diridx], diridx))
        else:
            fake = [data[diridx][-1] for _ in range(config.len_sequence - len(data[diridx]))]
            items_test.append((data[diridx] + fake, diridx))

    ds_test = GestureDataset(items_test, fileloader=fileloader, augmentation=False)
    dl_test = DataLoader(ds_test, **dl_kwargs, shuffle=False)

    return dl_train, dl_valid, dl_test


class GestureTrainerOutput:
    def __init__(self) -> None:
        self._loss = AverageMeter()
        self._correct, self._total = 0, 0
        self._preds, self._targets = [], []
        self._labels = list(range(N_CLASSES))

        self._target_names = [label_names[i]["name"] for i in range(N_CLASSES)]

    @torch.no_grad()
    def update(self, loss: Tensor, preds: Tensor, labels: Tensor):
        n = preds.size(0)
        self._loss.update(loss.item(), n)

        pclass = preds if preds.dim() == 1 else preds.argmax(dim=1)
        tclass = labels if labels.dim() == 1 else labels.argmax(dim=1)

        self._correct += (pclass == tclass).sum().item()
        self._total += n

        self._preds.extend(pclass.tolist())
        self._targets.extend(tclass.tolist())

    @property
    def acc(self):
        if self._total == 0:
            return 0
        return self._correct / self._total * 100

    @property
    def loss(self):
        return self._loss()

    @property
    def f1(self):
        return f1_score(self._targets, self._preds, labels=self._labels, average="macro")

    @property
    def report(self):
        return classification_report(
            self._targets,
            self._preds,
            labels=self._labels,
            target_names=self._target_names,
        )


class GestureTrainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.log = CustomLogger(config.result_dir / "log.log")
        self.log_rpt = CustomLogger(config.result_dir / "report_train.log")
        self.log_rpv = CustomLogger(config.result_dir / "report_valid.log")

        self.epoch = 1
        self.best_loss = math.inf
        self.earlystop_cnt = 0

        self.model = Net(
            name=config.model_name,
            n_classes=N_CLASSES,
            pretrained=config.pretrained,
            len_sequence=config.len_sequence,
        ).cuda()

        if config.checkpoint_path is not None:
            self.model.load_state_dict(torch.load(config.checkpoint_path))

        self.model = nn.DataParallel(self.model)

        # criterion
        if config.criterion == "ce":
            self.criterion = nn.CrossEntropyLoss().cuda()
        elif config.criterion == "focal":
            self.criterion = FocalLoss().cuda()
        else:
            raise NotImplementedError(config.criterion)

        # optimizer
        OptimizerClass = {
            "AdamW": AdamW,
            "AdaBelief": AdaBelief,
        }[config.optimizer_name]
        if config.sam:
            self.optimizer = SAM(
                self.model.parameters(),
                base_optimizer=OptimizerClass,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = OptimizerClass(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
            if config.look_ahead:
                self.optimizer = Lookahead(
                    optimizer=self.optimizer,
                    k=config.look_ahead_k,
                    alpha=config.look_ahead_alpha,
                )
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=3, verbose=True)

        self.tdl, self.vdl, self.dl_test = make_datasets(config)

    def save(self):
        ckpt_path = self.config.result_dir / "best.pth"
        self.log.info("Save checkpoint:", ckpt_path)

        state_dict = self.model.module.state_dict()

        torch.save(state_dict, ckpt_path)

    def train_loop(self, dl: DataLoader):
        o = GestureTrainerOutput()

        tqdm_desc = f"Train [{self.epoch:02d}/{self.config.epochs:02d}]"

        with tqdm(total=len(dl.dataset), **tqdm_kwargs, desc=tqdm_desc) as t:
            for images, diridxes, labels in dl:
                images_ = images.cuda()
                labels_ = labels.cuda()
                n = images.size(0)

                self.optimizer.zero_grad()

                logits_ = self.model(images_)
                loss = self.criterion(logits_, labels_)

                o.update(loss, logits_, labels_)
                t.set_postfix_str(f"loss:{o.loss:.6f}, acc:{o.acc:.2f}")

                loss.backward()
                if self.config.sam:
                    self.optimizer.first_step(zero_grad=True)
                    logits2_ = self.model(images_)
                    loss2 = self.criterion(logits2_, labels_)
                    loss2.backward()
                    self.optimizer.second_step(zero_grad=True)
                else:
                    self.optimizer.step()

                # self.scheduler.step()

                t.update(n)

        return o

    @torch.no_grad()
    def valid_loop(self, dl: DataLoader):
        o = GestureTrainerOutput()

        tqdm_desc = f"Valid [{self.epoch:02d}/{self.config.epochs:02d}]"

        with tqdm(total=len(dl.dataset), **tqdm_kwargs, desc=tqdm_desc) as t:
            for images, diridxes, labels in dl:
                images_ = images.cuda()
                labels_ = labels.cuda()
                n = images.size(0)

                logits_ = self.model(images_)
                loss = self.criterion(logits_, labels_)

                o.update(loss, logits_, labels_)
                t.set_postfix_str(f"loss:{o.loss:.6f}, acc:{o.acc:.2f}")

                t.update(n)

        return o

    @torch.no_grad()
    def callback(self, to: GestureTrainerOutput, vo: GestureTrainerOutput):
        print()

        self.log.info(
            f"ep[{self.epoch:03d}/{self.config.epochs:03d}]",
            f"loss[{to.loss:.6f};{vo.loss:.6f}]",
            f"acc[{to.acc:.2f};{vo.acc:.2f}]",
            f"f1[{to.f1:.4f};{vo.f1:.4f}]",
        )

        _t = f"ep[{self.epoch:03d}/{self.config.epochs:03d}]"
        self.log_rpt.info("TRAIN", _t, "\n", to.report)
        self.log_rpv.info("VALID", _t, "\n", vo.report)

        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(vo.loss)

        if self.best_loss > vo.loss:
            self.best_loss = vo.loss
            self.earlystop_cnt = 0
            self.save()
        else:
            self.earlystop_cnt += 1

        self.log.flush()
        self.log_rpt.flush()
        self.log_rpv.flush()

    def fit(self):
        seed_everything(self.config.seed)

        for self.epoch in range(self.epoch, self.config.epochs + 1):
            if self.config.finetune:
                if self.epoch <= self.config.finetune_step1_epochs:
                    self.model.module.freeze(step=1)
                elif self.epoch <= self.config.finetune_step2_epochs:
                    self.model.module.freeze(step=2)
                else:
                    self.model.module.freeze(step=3)

            # Training
            self.model.train()
            to = self.train_loop(self.tdl)

            if self.earlystop_cnt >= self.config.earlystop_limit:
                self.log.info("Early Stopping")
                break
            else:
                with torch.no_grad():
                    self.model.eval()
                    vo = self.valid_loop(self.vdl)
                    self.callback(to, vo)

            self.log.flush()
            self.log_rpt.flush()
            self.log_rpv.flush()

    @torch.no_grad()
    def submit(self):
        # load best checkpoint
        checkpoint_path = self.config.result_dir / "best.pth"
        self.log.info("Load checkpoint", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        self.model.module.load_state_dict(checkpoint)

        self.model.eval()

        # exp022: update submission tta
        ret = defaultdict(list)
        with tqdm(total=len(self.dl_test.dataset), ncols=100, file=sys.stdout, desc="submission") as t:
            for images, diridxes in self.dl_test:
                logits = self.model(images.cuda()).cpu()

                for logit, diridx in zip(logits, diridxes):
                    ret[diridx.item()].append(logit)

                    t.update()

        out_sm = defaultdict(list)
        out_ms = defaultdict(list)
        for diridx, logits in ret.items():
            out_sm["Image_Path"].append(f"./test\\{diridx}")
            out_ms["Image_Path"].append(f"./test\\{diridx}")

            logits = torch.stack(logits)
            logit_sm = logits.softmax(dim=1).mean(dim=0)
            logit_ms = logits.mean(dim=0).softmax(dim=0)

            for k in range(196):
                if k in id_to_label:
                    out_sm[f"Label_{k}"].append(logit_sm[id_to_label[k]].item())
                    out_ms[f"Label_{k}"].append(logit_ms[id_to_label[k]].item())

        df_sm = pd.DataFrame(out_sm)
        df_ms = pd.DataFrame(out_ms)

        out_df_path = str(self.config.result_dir / f"exp{self.config.exp_num}_ver{self.config.ver_num}_%s.csv")
        print("Write result to", out_df_path % "_")
        df_sm.to_csv(out_df_path % "sm", index=False)
        df_ms.to_csv(out_df_path % "ms", index=False)


def main():
    for model, batch_size in [
        # ["tf_efficientnet_b3_ns", 45],
        # ["tf_efficientnet_b4_ns", 30],
        # ["tf_efficientnet_b5_ns", 20],
        # ["tf_efficientnet_b6_ns", 15],
        # ["tf_efficientnet_b7_ns", 45],
        # ["efficientnetv2_s", 50],
        ["tf_efficientnetv2_l_in21ft1k", 54],
        # ["tf_efficientnetv2_m_in21ft1k", 25],
        # ["tf_efficientnetv2_s_in21ft1k", 45],
        # ["tf_efficientnetv2_b3", 50],
        # ["tf_efficientnetv2_l", 50],
        # ["tf_efficientnetv2_m", 25],
        # ["tf_efficientnetv2_s", 45],
        # ["dm_nfnet_f0", 50],
        # ["dm_nfnet_f1", 34],
        # ["dm_nfnet_f2", 18],
    ]:
        for optimizer in ["AdaBelief"]:
            for fold in [1, 2, 3, 4, 5]:
                for seed in [4]:
                    for lr in [1e-3, 1e-4]:
                        config = Config(
                            debug=False,
                            finetune=True,
                            model_name=model,
                            batch_size=batch_size,
                            sam=True,
                            cleared_image=False,
                            pretrained=True,
                            optimizer_name=optimizer,
                            fold=fold,
                            seed=seed,
                            num_workers=16,
                            in_memory=True,
                            lr=lr,
                        )

                        config.result_dir = make_result_dir(config)
                        shutil.copy(__file__, config.result_dir / Path(__file__).name)
                        config.to_yaml(config.result_dir / "params.yaml")

                        # config.result_dir = Path("results/exp/exp022/version_002")
                        # config.ver_num = 2

                        trainer = GestureTrainer(config)
                        trainer.fit()
                        trainer.submit()

                        del trainer
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
