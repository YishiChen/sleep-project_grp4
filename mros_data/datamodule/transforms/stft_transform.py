from functools import partial
from typing import List

import numpy as np
import matplotlib.pyplot as plt

# from scipy.signal import stft
from librosa import stft
from librosa import display
import librosa


class STFTTransform:
    """
    STFTTransform:
    """

    def __init__(self, fs: int, segment_size: int, step_size: int, nfft: int, *args, **kwargs) -> None:
        self.fs = fs
        self.segment_size = segment_size
        self.step_size = step_size
        self.nfft = nfft
        self.transform_fn = partial(stft, n_fft=self.nfft, hop_length=self.step_size, win_length=self.segment_size)

    def calculate_output_dims(self, window_size: int) -> List[int]:
        T = librosa.samples_to_frames(window_size, hop_length=self.step_size) + 1
        F = self.nfft // 2 + 1
        return [F, T]

    def __call__(self, X: np.ndarray, annotations: np.ndarray) -> np.ndarray:

        Zxx = np.abs(self.transform_fn(X)) ** 2
        Zxx = librosa.power_to_db(Zxx, top_db=50) / 50

        # show a plot
        # display.specshow(
        #     Zxx,
        #     sr=self.fs,
        #     hop_length=self.step_size,
        #     n_fft=self.nfft,
        #     win_length=self.segment_size,
        #     y_axis="hz",
        #     x_axis="time",
        # )
        # plt.show()

        return Zxx
