# code adapted from https://github.com/kekmodel/FixMatch-pytorch/tree/master
import logging
import math
import os
import shutil

import torch
from torch.optim.lr_scheduler import LambdaLR


def create_model(args):
    if args.arch == "wideresnet":
        import models.wideresnet as models

        model = models.build_wideresnet(
            depth=args.model_depth,
            widen_factor=args.model_width,
            dropout=0,
            num_classes=args.num_classes,
            n_channels=args.n_channels,
        )
    elif args.arch == "resnext":
        import models.resnext as models

        model = models.build_resnext(
            cardinality=args.model_cardinality,
            depth=args.model_depth,
            width=args.model_width,
            num_classes=args.num_classes,
        )

    elif args.arch == "vit":
        import models.vit as models

        path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth"
        model = models.vit_tiny_patch2_32(
            pretrained=True, pretrained_path=path, num_classes=args.num_classes
        )
    logging.info(
        "Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6)
    )
    return model


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def save_checkpoint(state, is_best, checkpoint, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))
