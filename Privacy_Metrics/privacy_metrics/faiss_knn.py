import math
from typing import Tuple, Union

import faiss
import numpy as np

# docs Python: https://github.com/facebookresearch/faiss/wiki/Getting-started

class FaissKNeighbors:
    index: Union[faiss.IndexFlatL2, faiss.IndexIVFFlat]

    def __init__(self, k: int = 5) -> None:
        self.index = faiss.IndexFlatL2()
        self.k = k

    def fit(self, X) -> None:
        xb = X.astype(np.float32)
        size, dimension = X.shape
        nlist = round(math.sqrt(size))
        threshold_size = 50000

        # Use Index for large size dataframe
        if size > threshold_size:
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_L2
            )
            assert self.index is not None  # nosec: B101
            self.index.train(xb)

        # perform exhaustive search otherwise
        else:
            self.index = faiss.IndexFlatL2(dimension)

        assert self.index is not None  # nosec: B101
        self.index.add(xb)

    def predict(self, X):
        assert self.index is not None  # nosec: B101
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        distances = np.sqrt(distances)
        return distances, indices
