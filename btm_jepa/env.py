"""Narrow compatibility fixes for the pinned LeFlow evaluation API."""

import numpy as np
import stable_worldmodel as swm


class World(swm.World):
    def reset(self, seed=None, options=None):
        # HDF5 provides np.int64 seeds; Gymnasium requires Python int seeds.
        if isinstance(seed, (list, tuple, np.ndarray)):
            seed = [int(x) for x in seed]
        elif seed is not None:
            seed = int(seed)
        return super().reset(seed=seed, options=options)
