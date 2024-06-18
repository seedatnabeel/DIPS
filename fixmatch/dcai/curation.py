import math
import time

import numpy as np
import torch.nn as nn
import torch.optim as optim
from datagnosis.plugins import Plugins
from dcai.dips import DIPS_Torch
from tqdm import tqdm
from utils import AverageMeter
from utils.datahandler import DataHandler
from utils.models import create_model, get_cosine_schedule_with_warmup


def train_and_clean(args, labeled_trainloader, unshuffled_labeled_trainloader):

    end = time.time()

    model = create_model(args).to(args.device)

    no_decay = ["bias", "bn"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.wdecay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.arch == "vit":
        optimizer = optim.AdamW(
            grouped_parameters,
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.wdecay,
            amsgrad=False,
        )
    else:
        optimizer = optim.SGD(
            grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov
        )
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps
    )

    if args.use_ema:
        from models.ema import ModelEMA

        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0
    model.zero_grad()

    if args.use_small_loss:

        criterion = nn.CrossEntropyLoss()

        labeled_trainloader.dataset.return_index = True
        unshuffled_labeled_trainloader.dataset.return_index = True

        datahandler = DataHandler(
            dataloader=labeled_trainloader,
            unshuffled_dataloader=unshuffled_labeled_trainloader,
        )

        hcm = Plugins().get(
            "large_loss",
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr=args.lr,
            epochs=args.super_epochs - args.start_epoch,
            num_classes=args.num_classes,
            logging_interval=1,
        )

        hcm.fit(
            datahandler=datahandler,
            workspace="../results",
            use_caches_if_exist=False,
        )

        scores = hcm.scores
        threshold = np.percentile(scores, 99)
        indices_easy = np.where(scores < threshold)[0]
        indices_not_easy = np.where(scores >= threshold)[0]

        assert len(indices_easy) + len(indices_not_easy) == len(scores)

    elif args.use_dips:
        dips = DIPS_Torch(dataloader=unshuffled_labeled_trainloader, sparse_labels=True)

        for epoch in range(args.start_epoch, args.super_epochs):
            model.train()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

            for inputs_x, targets_x in tqdm(labeled_trainloader):

                criterion = nn.CrossEntropyLoss()
                optimizer.zero_grad()

                data_time.update(time.time() - end)
                inputs = inputs_x.to(args.device)
                targets_x = targets_x.to(args.device)

                logits_x = model(inputs)

                loss = criterion(logits_x, targets_x)

                loss.backward()

                losses.update(loss.item())
                optimizer.step()
                scheduler.step()
                if args.use_ema:
                    ema_model.update(model)

                batch_time.update(time.time() - end)
                end = time.time()

            dips.on_epoch_end(model, device=args.device, gradient=False)

        confidence = dips.confidence

        uncertainty = dips.aleatoric

        threshold_confidence = np.percentile(confidence, args.percentile_cleaning)
        indices_easy = np.where(confidence >= threshold_confidence)[0]
        indices_not_easy = np.where(confidence < threshold_confidence)[0]

        assert len(indices_easy) + len(indices_not_easy) == len(uncertainty)
    else:
        raise ValueError("Cleaning must be either Data-IQ or small loss")

    labeled_trainloader.dataset.return_index = False
    unshuffled_labeled_trainloader.dataset.return_index = False

    return indices_easy, indices_not_easy
