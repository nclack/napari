from concurrent.futures import Future, ThreadPoolExecutor
from time import sleep
from typing import Any

import dask.array as da

import napari


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


class DataSource:
    def __init__(self, dask_array, exec) -> None:
        self._dask_array = dask_array
        self._exec = exec

    def compute(self) -> Future[Any]:
        def _work():
            sleep(1.0)
            out = self._dask_array.compute()
            return out

        fut: Future[Any] = self._exec.submit(_work)
        fut.add_done_callback(lambda ctx: dbg("done", ctx))
        return fut

    @property
    def shape(self):
        return self._dask_array.shape

    @property
    def dtype(self):
        return self._dask_array.dtype

    def __getitem__(self, val):
        da = self._dask_array.__getitem__(val)
        return DataSource(da, self._exec)


v = napari.Viewer()

print('exec started')
ds = DataSource(da.random.random((256, 256, 256)), ThreadPoolExecutor())
print('adding image...')
v.add_image([ds])
print('added')
print('napari run')
napari.run()
print('exec exit')

"""
# NOTES

## Current Approach

We need something that has (a) lazy-evaluation so that we only
load/compute on the data we end up looking querying and (b) can be materialized
asynchronously.

The `DataSource` object defined here wraps a dask array (giving us (a)) which
can be converted to a `concurrent.Futures` object via the `compute()` method.

The idea is that a `DataSource` gets to leverage dask's ability to aggregate
a compute graph by composing normal array operations up until the last second
where it needs to be realized as an array.  At that point, it gets converted
to a Future which can be eagerly but asynchronously evaluated.

## Status

It works for Image layers in a course way.  See Problems.

The `DataSource` above can be manipulated to increase the load time (via
`sleep`). The presentation of the data in napari is delayed, but otherwise
remains responsive.

## Problems

This effect of this approach would be made more clear with a suitable control.

At the moment, the future invokes a callback to update napari when data is
available. It's hard to do things like cancel the futures when they're
unnecessary.

Currently, this approach has been tested with image layers.  Image layers have
already been prepared to support this style using the framework Philip Winston
put in place. It is important to validate the approach on other layers.
"""
