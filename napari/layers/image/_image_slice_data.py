"""ImageSliceData class.
"""
from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from ..base import Layer

if TYPE_CHECKING:
    from ...types import ArrayLike


def dbg(*args):
    import inspect
    from pathlib import Path

    caller = inspect.getframeinfo(inspect.stack()[1][0])
    print(
        "{}:{} - {}".format(
            '/'.join(Path(caller.filename).parts[-2:]), caller.lineno, args
        )
    )
    return args


WaitingData = namedtuple('WaitingData', ('fut', 'source'))


class ImageSliceData:
    """The contents of an ImageSlice.

    Parameters
    ----------
    layer : Layer
        The layer that contains the data.
    indices : Tuple[Optional[slice], ...]
        The indices of this slice.
    image : ArrayList
        The image to display in the slice.
    thumbnail_source : ArrayList
        The source used to create the thumbnail for the slice.
    """

    def __init__(
        self,
        layer: Layer,
        indices: Tuple[Optional[slice], ...],
        image: ArrayLike,
        thumbnail_source: ArrayLike,
    ):
        dbg('new image slice')
        self.layer = layer
        self.indices = indices
        self.image = WaitingData(fut=None, source=image)  # (Future,DataSource)
        self.thumbnail_source = thumbnail_source

    def sig_load(self):
        assert type(self.image) is WaitingData
        fut = self.image.fut
        if fut is None:
            fut = self.image.source.compute()
            self.image = self.image._replace(fut=fut)

            def cb(_):
                self.load_sync()  # image becomes the ndarray
                self.layer._on_data_loaded(self, sync=False)

            fut.add_done_callback(cb)
            dbg(self.image.fut)

    def cancel(self):
        if (type(self.image) is WaitingData) and self.image.fut:
            dbg('CANCEL fut', self.image.fut)
            self.image.fut.cancel()

    def __del__(self):
        self.cancel()

    def is_ready(self) -> bool:
        if self.image.fut:
            dbg('check fut')
            return self.image.fut.done()
        else:
            dbg('sig load')
            self.sig_load()
            return self.is_ready()

    def load_sync(self) -> None:
        """Call asarray on our images to load them."""
        assert self.image.fut is not None
        self.image: np.ndarray = self.image.fut.result()
        if self.thumbnail_source is not None:
            self.thumbnail_source = np.asarray(self.thumbnail_source)

    def transpose(self, order: tuple) -> None:
        """Transpose our images.

        Parameters
        ----------
        order : tuple
            Transpose the image into this order.
        """
        self.image = self.image.transpose(order)

        if self.thumbnail_source is not None:
            self.thumbnail_source = np.transpose(self.thumbnail_source, order)
