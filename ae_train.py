import torch
from pathlib import Path
import argparse
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from ae_model import CNNAutoencoder

from custom_audio_dataset import BirdAudioDataset

from torch.utils.tensorboard import SummaryWriter

AUDIO_FILE = "/grand/projects/BirdAudio/Morton_Arboretum/audio/set3/00004879/20210816_STUDY/20210816T063139-0500_Rec.wav"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Pretrain a resnet model with VICReg", add_help=False
    )

    # Optim
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Effective batch size"
    )
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")

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
        default=False,
        help="Whether or not to resume from a checkpoint",
    )

    return parser


def train_one_epoch(model, data_loader, loss_fn, optimizer, device, epoch, writer):
    for i, (inputs, _) in enumerate(data_loader, start=epoch * len(data_loader)):
        inputs = inputs.to(device)
        recons = model(inputs)
        loss = loss_fn(recons, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss", loss.item(), i)
    state = dict(
        epoch=epoch + 1,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    torch.save(state, args.exp_dir / "model.pth")


def train(model, data_loader, loss_fn, optimizer, device, epochs, start_epoch, writer):
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device, epoch, writer)
        print("---------------")
    print("Training is done.")
    torch.save(model.state_dict(), args.exp_dir / "conv_autoencoder.pth")


def main(args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    args.exp_dir.mkdir(parents=True, exist_ok=True)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    bad = BirdAudioDataset(
        AUDIO_FILE, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device
    )

    train_data_loader = DataLoader(bad, batch_size=args.batch_size)

    model = CNNAutoencoder()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model = model.to(device=device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if (args.exp_dir / "model.pth").is_file() and args.resume:
        print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_epoch = 0
    writer = SummaryWriter(
        f"theta_run_birds/lr{args.lr}_wd{args.wd}_bs{args.batch_size}"
    )

    train(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        args.epochs,
        start_epoch,
        writer,
    )


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    main(args)
