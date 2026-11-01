import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 160
WIN = 2 * FS          # 2 s decoding window
SOS = butter(4, [8, 30], btype="bandpass", fs=FS, output="sos")

def bandpass(x):      # x: (..., n_ch, n_times)
    return sosfiltfilt(SOS, x, axis=-1)
