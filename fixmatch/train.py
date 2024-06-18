import argparse
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from dataset.dataset_getters import DATASET_GETTERS
from dcai.dips import DIPS_Torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter, accuracy
from utils.models import create_model, get_cosine_schedule_with_warmup, save_checkpoint
from utils.seed import set_seed
from utils.tensors import de_interleave, interleave

import wandb

logger = logging.getLogger(__name__)
best_acc = 0


def main():
    parser = argparse.ArgumentParser(description="PyTorch FixMatch Training")
    parser.add_argument(
        "--gpu-id", default="0", type=int, help="id(s) for CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--dataset",
        default="eurosat",
        type=str,
        choices=["cifar10", "cifar100", "eurosat", "tissue", "pathmnist"],
        help="dataset name",
    )
    parser.add_argument(
        "--num-labeled", type=int, default=40, help="number of labeled data"
    )
    parser.add_argument(
        "--prop-label-noise", type=float, default=0.2, help="prop  label noise"
    )
    parser.add_argument(
        "--prop-label-keep",
        type=float,
        default=0.5,
        help="prop of the labeled set to be kepts",
    )

    parser.add_argument(
        "--expand-labels", action="store_true", help="expand labels to fit eval steps"
    )
    parser.add_argument(
        "--arch",
        default="vit",
        type=str,
        choices=["wideresnet", "resnext", "vit"],
        help="dataset name",
    )
    parser.add_argument(
        "--total-steps", default=2**20, type=int, help="number of total steps to run"
    )
    parser.add_argument(
        "--eval-step", default=1024, type=int, help="number of eval steps to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument("--batch-size", default=16, type=int, help="train batchsize")
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.03,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--warmup", default=0, type=float, help="warmup epochs (unlabeled data based)"
    )
    parser.add_argument("--wdecay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--nesterov", action="store_true", default=True, help="use nesterov momentum"
    )
    parser.add_argument(
        "--use-ema", action="store_true", default=True, help="use EMA model"
    )
    parser.add_argument(
        "--use-dips", action="store_true", default=False, help="use dips"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Debug")
    parser.add_argument(
        "--iterative-dips",
        action="store_true",
        default=False,
        help="clean every 10 epochs",
    )
    parser.add_argument("--ema-decay", default=0.999, type=float, help="EMA decay rate")
    parser.add_argument(
        "--mu", default=7, type=int, help="coefficient of unlabeled batch size"
    )
    parser.add_argument(
        "--lambda-u", default=1, type=float, help="coefficient of unlabeled loss"
    )
    parser.add_argument("--T", default=1, type=float, help="pseudo label temperature")
    parser.add_argument(
        "--threshold", default=0.95, type=float, help="pseudo label threshold"
    )
    parser.add_argument(
        "--out", default="result", help="directory to output the result"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--seed", default=None, type=int, help="random seed")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="use 16-bit (mixed) precision through NVIDIA apex AMP",
    )
    parser.add_argument(
        "--opt_level",
        type=str,
        default="O1",
        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="don't use progress bar"
    )
    parser.add_argument(
        "--project-name", type=str, default="fixmatch-test", help="Name of the project"
    )
    parser.add_argument(
        "--noise-type", type=str, default="aggre_label", help="Name of the project"
    )
    parser.add_argument(
        "--super-epochs", type=int, default=100, help="Name of the project"
    )
    parser.add_argument(
        "--unlabeled-prop", type=float, default=1, help="Name of the project"
    )
    parser.add_argument(
        "--percentile-cleaning",
        type=float,
        default=50,
        help="Percentile for the first cleaning step",
    )
    parser.add_argument(
        "--percentile-iterative",
        type=float,
        default=10,
        help="Percentile for the cleaning steps inside FixMatch",
    )
    parser.add_argument(
        "--use_small_loss",
        type=bool,
        default=False,
        help="Whether or not to use small loss",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="CIFAR10",
        help="Name of the project for WandB",
    )

    args = parser.parse_args()

    # Load the WANDB YAML file
    with open("../wandb.yaml") as file:
        wandb_data = yaml.load(file, Loader=yaml.FullLoader)

    os.environ["WANDB_API_KEY"] = wandb_data["wandb_key"]
    wandb_entity = wandb_data["wandb_entity"]

    run = wandb.init(
        project=args.project_name,
        entity=wandb_entity,
    )

    arg_dict = vars(args)
    wandb.log(arg_dict)

    global best_acc

    if args.local_rank == -1:
        device = torch.device("cuda", args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",
    )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == "cifar10":
        args.num_classes = 10
        args.n_channels = 3
        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == "resnext":
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    if args.dataset == "eurosat":
        args.num_classes = 10
        args.n_channels = 3

        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == "resnext":
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    if args.dataset == "tissue":
        args.num_classes = 8
        args.n_channels = 1

        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == "resnext":
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    if args.dataset == "pathmnist":

        args.num_classes = 9
        args.n_channels = 3

        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == "resnext":
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == "cifar100":
        args.num_classes = 100
        args.n_channels = 3
        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == "resnext":
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    unshuffled_labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=False,
    )

    unlabeled_trainloader = DataLoader(
        dataset=unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

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

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint["best_acc"]
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    if args.amp:
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(
        args,
        labeled_trainloader,
        unshuffled_labeled_trainloader,
        unlabeled_trainloader,
        test_loader,
        model,
        optimizer,
        ema_model,
        scheduler,
        run,
    )

    # Finish the run
    wandb.finish()


def train(
    args,
    labeled_trainloader,
    unshuffled_labeled_trainloader,
    unlabeled_trainloader,
    test_loader,
    model,
    optimizer,
    ema_model,
    scheduler,
    run,
):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    dips = DIPS_Torch(dataloader=unshuffled_labeled_trainloader)

    for epoch in range(args.start_epoch, args.epochs):

        if epoch in [100 * k for k in range(2, 10)] or epoch == 2:
            torch.save(model.state_dict(), "../results/model.pth")
            # Save as artifact for version control.
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file("../results/model.pth")
            run.log_artifact(artifact)
        if epoch % 10 == 0 and epoch != 0 and args.iterative_dips:
            labeled_trainloader, unshuffled_labeled_trainloader, n_selected = (
                clean_train_loader(dips, unshuffled_labeled_trainloader, args)
            )
            dips = DIPS_Torch(dataloader=unshuffled_labeled_trainloader)
            wandb.log({"n_selected": n_selected})

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            model.train()
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            except Exception as e:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]

            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1
            ).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (
                F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask
            ).mean()

            loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if batch_idx % 100 == 0:
                test_loss, test_acc = test(args, test_loader, model, epoch)
                print("test acc", test_acc)
                log_dict = {"epoch": epoch + 1, "acc": test_acc, "batch_idx": batch_idx}

            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg,
                    )
                )
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar("train/1.train_loss", losses.avg, epoch)
            args.writer.add_scalar("train/2.train_loss_x", losses_x.avg, epoch)
            args.writer.add_scalar("train/3.train_loss_u", losses_u.avg, epoch)
            args.writer.add_scalar("train/4.mask", mask_probs.avg, epoch)
            args.writer.add_scalar("test/1.test_acc", test_acc, epoch)
            args.writer.add_scalar("test/2.test_loss", test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = (
                    ema_model.ema.module
                    if hasattr(ema_model.ema, "module")
                    else ema_model.ema
                )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                    "ema_state_dict": (
                        ema_to_save.state_dict() if args.use_ema else None
                    ),
                    "acc": test_acc,
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_best,
                args.out,
            )

            test_accs.append(test_acc)
            logger.info("Best top-1 acc: {:.2f}".format(best_acc))
            logger.info("Mean top-1 acc: {:.2f}\n".format(np.mean(test_accs[-20:])))

            log_dict = {
                "epoch": epoch + 1,
                "acc": test_acc,
                "best_acc": best_acc,
            }
            print("LOGGING TO WANDB...")
            wandb.log(log_dict)

        dips.on_epoch_end(net=model, device=args.device)

    if args.local_rank in [-1, 0]:
        args.writer.close()


def clean_train_loader(dips, unshuffled_loader, args):

    confidence = dips.confidence

    if len(confidence) > int(args.num_labeled * args.prop_label_keep):
        threshold_confidence = np.percentile(confidence, args.percentile_iterative)
    else:
        threshold_confidence = -1

    selected_points = np.where((confidence > threshold_confidence))[0]

    if len(selected_points) == 0:
        selected_points = np.arange(len(confidence))

    # extract the dataset from the unshuffled trainloader
    dataset = unshuffled_loader.dataset

    # only select the selected_points in the dataset
    dataset.data = dataset.data[selected_points]
    dataset.targets = dataset.targets[selected_points]

    # create a new dataloader
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    new_train_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )
    unshuffled_labeled_trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=False,
    )

    return new_train_loader, unshuffled_labeled_trainloader, len(selected_points)


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    )
                )
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == "__main__":
    main()
