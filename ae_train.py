import torch
from pathlib import Path
import os
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torch.distributed import all_gather
import torchaudio
from ae_settings import *
from ae_model import CNNAutoencoder
from distributed import init_distributed_mode

from custom_audio_dataset import BirdAudioDataset

from torch.utils.tensorboard import SummaryWriter

import scipy.stats as stats


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Pretrain a resnet model with VICReg", add_help=False
    )

    # Optim
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Effective batch size"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-7, help="Weight decay")

    # Checkpoints
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default="./exp",
        help="Path to the experiment folder, where all logs/checkpoints will be stored",
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=True,
        help="Whether or not to resume from a checkpoint",
    )

    # Running
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    # Distributed
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    return parser


def get_all_audio_files(audio_path, audio_file_cap):
    audio_files = []
    for root, dirs, files in os.walk(audio_path, topdown=False):
        for name in files:
            extension = os.path.splitext(name)[1]
            if extension == ".wav":
                audio_file = os.path.join(root, name)
                audio_files.append(audio_file)
                if len(audio_files) >= audio_file_cap:
                    return audio_files
    return audio_files


def train_one_epoch(model, data_loader, loss_fn, optimizer, device, epoch, writer, gpu):
    print(f"Length of dataloader dataset: {len(data_loader.dataset)}")
    for i, (inputs, _) in enumerate(data_loader, start=epoch * len(data_loader)):
        inputs = inputs.cuda(gpu, non_blocking=True)
        recons, _ = model(inputs)
        loss = loss_fn(recons, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss", loss.item(), i)
    if args.rank == 0:
        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        torch.save(state, args.exp_dir / "model.pth")


def train(
    model,
    bad,
    data_loader,
    prune_data_loader,
    loss_fn,
    optimizer,
    device,
    epochs,
    start_epoch,
    writer,
    gpu,
):
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Length of dataset: {len(bad)}")
        train_one_epoch(
            model, data_loader, loss_fn, optimizer, device, epoch, writer, gpu
        )
        if PRUNE_ANOMALIES and epoch % PRUNE_FREQUENCY == 0:
            print("Pruning anomalies!")
            bad = prune_anomalies(model, bad, prune_data_loader, loss_fn, gpu)
            data_loader, prune_data_loader = get_data_loaders(bad)
    print("Training is done.")
    if args.rank == 0:
        torch.save(model.state_dict(), args.exp_dir / "conv_autoencoder.pth")


def prune_anomalies(model, bad, prune_data_loader, loss_fn, gpu):
    indices = []
    losses = []

    for i, (inputs, _) in enumerate(prune_data_loader):
        inputs = inputs.cuda(gpu, non_blocking=True)
        recons, _ = model(inputs)
        loss = loss_fn(recons, inputs)
        indices.append(i)
        losses.append(loss)

    indices = torch.tensor(indices, dtype=torch.int32).cuda(gpu, non_blocking=True)
    losses = torch.tensor(losses, dtype=torch.float32).cuda(gpu, non_blocking=True)


    all_indices = [
        torch.zeros(len(prune_data_loader), dtype=torch.int32).cuda(gpu, non_blocking=True) for _ in range(8)
    ]
    all_losses = [
        torch.zeros(len(prune_data_loader), dtype=torch.float32).cuda(gpu, non_blocking=True) for _ in range(8)
    ]

    all_gather(all_indices, indices)
    all_gather(all_losses, losses)

    all_indices = torch.cat(all_indices).detach().cpu().numpy()
    all_losses = torch.cat(all_losses).detach().cpu().numpy()

    all_z_scores = stats.zscore(all_losses)

    normal_indices = all_indices[all_z_scores < 3]

    print(f"Pruned {len(all_indices) - len(normal_indices)} samples...")

    bad = torch.utils.data.Subset(bad, normal_indices)

    return bad


def main(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=((256 if TIGHT_CROP_MODE else 128) if CROPPED_MODE else 64),
    )

    audio_files = get_all_audio_files(AUDIO_PATH, AUDIO_FILE_CAP)
    print(f"Using {len(audio_files)} audio files!")

    bad = BirdAudioDataset(
        audio_files,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        gpu,
        crop_frequencies=CROPPED_MODE,
        tight_crop=TIGHT_CROP_MODE,
    )

    train_data_loader, prune_data_loader = get_data_loaders(bad)

    model = CNNAutoencoder().cuda(gpu)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if (args.exp_dir / "model.pth").is_file() and args.resume:
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    log_dir = f"theta_run_birds/lr{args.lr}_wd{args.wd}_bs{args.batch_size}_dim10_files{AUDIO_FILE_CAP}_{'tight' if TIGHT_CROP_MODE else 'std'}_{'prune' if PRUNE_ANOMALIES else 'noprune'}"
    writer = SummaryWriter(log_dir)
    print(f"Tensorboard logging at {log_dir}")

    train(
        model,
        bad,
        train_data_loader,
        prune_data_loader,
        loss_fn,
        optimizer,
        args.device,
        args.epochs,
        start_epoch,
        writer,
        gpu,
    )


def get_data_loaders(bad):
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(bad, shuffle=True)
    prune_sampler = torch.utils.data.distributed.DistributedSampler(bad, shuffle=False)

    train_data_loader = DataLoader(
        bad,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=train_sampler,
    )
    prune_data_loader = DataLoader(
        bad,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=prune_sampler
    )

    return train_data_loader, prune_data_loader


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    main(args)
