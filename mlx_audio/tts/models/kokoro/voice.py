import io
import pickle
import sys
import zipfile

import numpy as np


def load_voice_tensor(path: str) -> np.ndarray:
    """
    Load a voice pack .pt file into a NumPy array.
    Handles either flat layout or one extra top-level folder.
    """
    # map PyTorch storage names to NumPy dtypes
    _STORAGE_TO_DTYPE = {
        "FloatStorage": np.float32,
        "DoubleStorage": np.float64,
        "HalfStorage": np.float16,
        "IntStorage": np.int32,
        "LongStorage": np.int64,
        "ByteStorage": np.uint8,
        "CharStorage": np.int8,
        "ShortStorage": np.int16,
        "BoolStorage": np.bool_,
    }
    storages: dict[str, np.ndarray] = {}

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()

        # detect optional top-level prefix
        if "byteorder" in names:
            prefix = ""
        else:
            tops = {n.split("/", 1)[0] for n in names if "/" in n}
            prefix = (tops.pop() + "/") if len(tops) == 1 else ""

        try:
            byteorder = zf.read(f"{prefix}byteorder").decode("ascii").strip()
        except KeyError:
            byteorder = sys.byteorder

        # build a helper for retrieving raw storage blobs
        def _persistent_load(pid):
            typename, storage_type, root_key, _, numel = pid
            if typename != "storage":
                raise RuntimeError(f"Unknown persistent id: {typename}")
            if root_key not in storages:
                raw = zf.read(f"{prefix}data/{root_key}")
                name = storage_type.__name__
                try:
                    dtype = _STORAGE_TO_DTYPE[name]
                except KeyError:
                    raise RuntimeError(f"Unsupported storage type: {name}")
                if byteorder != sys.byteorder:
                    dtype = dtype.newbyteorder()
                storages[root_key] = np.frombuffer(raw, dtype=dtype, count=numel)
            return storages[root_key]

        # mimic torch._utils._rebuild_tensor_v2
        def _rebuild_tensor_v2(
            storage, storage_offset, size, stride, requires_grad, backward_hooks
        ):
            count = 1
            for d in size:
                count *= d
            segment = storage[storage_offset : storage_offset + count]
            return segment.reshape(size)

        class _NoTorchUnpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                return _persistent_load(pid)

            def find_class(self, module, name):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return _rebuild_tensor_v2
                return super().find_class(module, name)

        data_pkl = zf.read(f"{prefix}data.pkl")
        unpickler = _NoTorchUnpickler(io.BytesIO(data_pkl))
        return unpickler.load()
