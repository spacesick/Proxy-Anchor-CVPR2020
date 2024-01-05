import os
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data.sampler import Sampler
from torchvision.datasets import LFWPeople, VisionDataset


class LFWDataset(LFWPeople):

  def __init__(
      self,
      root: str,
      split: str = "10fold",
      image_set: str = "funneled",
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
      download: bool = False,
  ) -> None:
    super().__init__(
        root,
        split,
        image_set,
        transform,
        target_transform,
        download
    )

    # Set class numbers to be in the range: [0, number of classes)
    self.targets = np.unique(self.targets, return_inverse=True)[1].tolist()

  def num_classes(self):
    return len(self.class_to_idx)
