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
