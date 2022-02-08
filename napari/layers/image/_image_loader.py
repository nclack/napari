"""ImageLoader class.
"""
from ._image_slice_data import ImageSliceData


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


class ImageLoader:
    """The default synchronous ImageLoader."""

    def load(self, data: ImageSliceData) -> bool:
        """Load the ImageSliceData synchronously.

        Parameters
        ----------
        data : ImageSliceData
            The data to load.

        Returns
        -------
        bool
            True if load happened synchronously.
        """

        # HERE
        if data.is_ready():
            dbg("here")
            data.load_sync()
            return True
        else:
            dbg("here")
            return False

    def match(self, data: ImageSliceData) -> bool:
        """Return True if data matches what we are loading.

        Parameters
        ----------
        data : ImageSliceData
            Does this data match what we are loading?

        Returns
        -------
        bool
            Return True if data matches.
        """
        return True  # Always true for synchronous loader.
