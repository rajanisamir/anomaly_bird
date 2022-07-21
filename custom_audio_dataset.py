# Credit to Valerio Velardo for providing code for processing the Urban Sound Dataset; this code was adapted for processing the BirdAudio Dataset.

from torch.utils.data import Dataset
import pandas as pd
import torch
import math
import torchaudio
import os
import wave


class BirdAudioDataset(Dataset):
    # Note: crop_frequencies only works when transformation is set to a Mel spectrogram with 128 mels; in this case, it takes the highest 64 mels. tight_crop only works when crop_frequencies is also set to True and the transformation is set to a Mel spectrogram with 256 mels.
    def __init__(
        self,
        audio_files,
        transformation,
        target_sample_rate,
        num_samples,
        device,
        num_seconds=None,
        crop_frequencies=False,
        tight_crop=False
    ):
        # Set up instance attributes
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.crop_frequencies = crop_frequencies

        # The num_seconds parameter may only be set to a non-None value if there is a single audio clip; in this case, it will load the requested number of seconds from this audio clip
        assert len(audio_files) == 1 or num_seconds is None

        self.signals = []

        for audio_file in audio_files:

            # Load the appropriate section of audio
            if num_seconds is None:
                signal, sr = torchaudio.load(audio_file)
            else:
                with wave.open(audio_file, "rb") as wave_file:
                    sample_rate = wave_file.getframerate()
                print(f"Original sample rate of audio is {sample_rate} Hz")
                print(f"Loading {num_seconds} seconds of audio...")
                num_frames = num_seconds * sample_rate
                signal, sr = torchaudio.load(audio_file, num_frames=num_frames)
            
            # Resample and mix down
            signal = signal.to(self.device)
            signal = self._resample_if_necessary(signal, sr)
            signal = self._mix_down_if_necessary(signal)

            self.signals.append(signal)

        print(f"Loaded {len(audio_files)} audio file(s)")

    def __len__(self):
        total_samples = sum([signal.shape[1] for signal in self.signals])
        return math.ceil(total_samples / self.num_samples)

    def __getitem__(self, index):
        signal = self._get_signal_at_index(index)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        signal = self._crop_frequencies_if_necessary(signal)
        return (signal, index)

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(
                device=self.device
            )
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _crop_frequencies_if_necessary(self, signal):
        if self.crop_frequencies:
            if self.tight_crop:
                signal = signal[:, 64:128]
            else:
                signal = signal[:, 64:]
        return signal

    def _get_signal_index_and_offset(self, start_sample):
        curr_sample = 0
        for (signal_index, signal) in enumerate(self.signals):
            signal_samples = signal.shape[1]
            if start_sample < curr_sample + signal_samples:
                signal_offset = start_sample - curr_sample
                return signal_index, signal_offset
            curr_sample += signal_samples

    def _get_signal_at_index(self, index):
        start_sample = self.num_samples * index
        signal_index, start_index = self._get_signal_index_and_offset(start_sample)
        signal = self.signals[signal_index]
        end_index = start_index + self.num_samples
        return signal[:, start_index:end_index]

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


if __name__ == "__main__":

    AUDIO_FILES = ["/grand/projects/BirdAudio/Morton_Arboretum/audio/set1/00023734/20210628_STUDY/20210628T234550-0500_Rec.wav", "/grand/projects/BirdAudio/Morton_Arboretum/audio/set3/00004879/20210816_STUDY/20210816T063139-0500_Rec.wav"]
    # AUDIO_FILES = ["20210816T063139-0500_Rec.wav"]
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

    bad = BirdAudioDataset(AUDIO_FILES, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    print(f"There are {len(bad)} samples in the dataset")
    print(f"The shape of the first sample is {bad[0][0].shape}")
    print(f"The shape of the last sample is {bad[len(bad) - 1][0].shape}")
