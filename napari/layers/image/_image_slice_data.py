"""ImageSliceData class.
"""
from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Any, Optional, Tuple

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
        self.image = image
        self._fut: Optional[concurrent.futures.Future[Any]] = None
        self.thumbnail_source = thumbnail_source
        self.sig_load()

    def sig_load(self):
        def cb(_):
            self.load_sync()
            self.layer._on_data_loaded(self, sync=False)

        if self._fut is None:
            self._fut: concurrent.futures.Future[Any] = self.image.compute()
            self._fut.add_done_callback(cb)
            dbg(self._fut)

    def is_ready(self) -> bool:
        if self._fut:
            dbg('check fut')
            return self._fut.done()
        else:
            dbg('sig load')
            self.sig_load()
            return self.is_ready()

    def load_sync(self) -> None:
        """Call asarray on our images to load them."""
        assert self._fut is not None
        self.image = self._fut.result()
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
