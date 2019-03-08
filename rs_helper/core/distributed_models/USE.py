from typing import List, Any

import numpy as np

from rs_helper.core.distributed_models.EmbeddingModel import EmbeddingModel


class USE(EmbeddingModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_model(self, **kwargs) -> Any:
        pass

    def inference(self, words: List[str]) -> np.ndarray:
        pass
