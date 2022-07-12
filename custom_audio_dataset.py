# Credit to Valerio Velardo for providing code for processing the Urban Sound Dataset; this code was adapted for processing the BirdAudio Dataset.

from torch.utils.data import Dataset
import pandas as pd
import torch
import math
import torchaudio
import os


class BirdAudioDataset(Dataset):
    def __init__(
        self,
        audio_file,
        transformation,
        target_sample_rate,
        num_samples,
        device,
    ):
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        signal, sr = torchaudio.load(audio_file)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        self.signal, self.sr = signal, sr

    def __len__(self):
        return math.ceil(self.signal.shape[1] / self.num_samples)

    def __getitem__(self, index):
        signal = self._get_signal_at_index(self.signal, index)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(device=self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_signal_at_index(self, signal, index):
        start_index = self.num_samples * index
        end_index = start_index + self.num_samples
        return signal[:, start_index:end_index]

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":

    AUDIO_FILE = "/grand/projects/BirdAudio/Morton_Arboretum/audio/set1/00023734/20210628_STUDY/20210628T234550-0500_Rec.wav"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device} device")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    bad = BirdAudioDataset(
        AUDIO_FILE, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device
    )

    print(f"There are {len(bad)} samples in the dataset")
    print(f"The shape of the first sample is {bad[0].shape}")
    print(f"The shape of the last sample is {bad[len(bad) - 1].shape}")