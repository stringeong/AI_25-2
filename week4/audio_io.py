import torch
import torchaudio
import io
import os
import tarfile
import tempfile
import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset
from pathlib import Path

SAMPLE_1 = "./audio_dataset/sample1.wav"
SAMPLE_2= "./audio_dataset/sample2.wav"
SAMPLE_3 = "./audio_dataset/sample3.wav"
SAMPLE_4 = "./audio_dataset/sample4.wav"

def _hide_seek(obj):
    class _wrapper:
        def __init__(self, obj):
            self.obj = obj

        def read(self, n):
            return self.obj.read(n)

    return _wrapper(obj)

print(torchaudio.info(str(SAMPLE_1)))
print(torchaudio.info(str(SAMPLE_2)))
print(torchaudio.info(str(SAMPLE_3)))
print(torchaudio.info(str(SAMPLE_4)))

waveform1, sample_rate1 = torchaudio.load(str(SAMPLE_1))
waveform2, sample_rate2 = torchaudio.load(str(SAMPLE_2))
waveform3, sample_rate3 = torchaudio.load(str(SAMPLE_3))
waveform4, sample_rate4 = torchaudio.load(str(SAMPLE_4))

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)

plot_waveform(waveform1, sample_rate1)
plot_specgram(waveform1, sample_rate1)

plot_waveform(waveform2, sample_rate2)
plot_specgram(waveform2, sample_rate2)

plot_waveform(waveform3, sample_rate3)
plot_specgram(waveform3, sample_rate3)

plot_waveform(waveform4, sample_rate4)
plot_specgram(waveform4, sample_rate4)

plt.tight_layout()
plt.show()


