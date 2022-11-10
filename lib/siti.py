import numpy as np
import skvideo.io
from pathlib import Path
from scipy import ndimage
from typing import Optional


class SiTi:
    filename: Path
    previous_frame: Optional[np.ndarray]

    def __init__(self, filename: Path):
        self.filename = filename

    def calc_siti(self, verbose=True):
        vreader = skvideo.io.vreader(fname=str(self.filename), as_grey=True)
        si = []
        ti = []
        self.previous_frame = None
        name = f'{self.filename.parts[-2]}/{self.filename.name}'

        for frame_counter, frame in enumerate(vreader):
            # Fix shape
            width = frame.shape[2]
            height = frame.shape[1]
            frame = frame.reshape((height, width)).astype(np.float)

            value_si = self._calc_si(frame)
            si.append(value_si)

            if (value_ti := self._calc_ti(frame)) is not None:
                ti.append(value_ti)
                if verbose:
                    print(f'\rCalculating SiTi - {name}: Frame {frame_counter}, si={value_si:.2f}, ti={value_ti:.3f}', end='')
        print('')
        return si, ti

    @staticmethod
    def _calc_si(frame: np.ndarray) -> (float, np.ndarray):
        """
        Calcule Spatial Information for a video frame. Calculate both vectors and so the magnitude.
        :param frame: A luma video frame in numpy ndarray format.
        :return: spatial information and sobel frame.
        """
        soby = ndimage.sobel(frame, axis=0)
        sobx = ndimage.sobel(frame, axis=1, mode="wrap")
        sobel = np.hypot(soby, sobx)
        si = sobel.std()
        return si

    def _calc_ti(self, frame: np.ndarray) -> (float, np.ndarray):
        """
        Calcule Temporal Information for a video frame. If is a first frame,
        the information is zero.
        :param frame: A luma video frame in numpy ndarray format.
        :return: Temporal information and diference frame. If first frame the
        diference is zero array on same shape of frame.
        """
        if self.previous_frame is not None:
            difference = frame - self.previous_frame
            ti = difference.std()
        else:
            ti = None

        self.previous_frame = frame
        return ti


def show(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
