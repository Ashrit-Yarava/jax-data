import jax
import jax.numpy as jnp

import numpy as np

def default_collate_fn(samples):
    X = jnp.array([sample[0] for sample in samples])
    Y = jnp.array([sample[1] for sample in samples])
    return X, Y

class Dataset(object):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

class Dataloader(object):
    def __init__(self, dataset, batch_size, shuffle=True, collate_fn=default_collate_fn):
        """
        Adapted from Pytorch Dataloader implementation.

        params:
            - dataset : a class with the __len__ and __getitem__ implemented.
            - batch_size : size of each batch.
            - shuffle : whether to shuffle the dataset upon epoch ending.
            - collate_fn : How the samples are collated.

        Note: To shuffle the dataset, call on_epoch_end().
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = len(self.dataset)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.collate_fn = collate_fn


    def __getitem__(self, idx):
        if len(self) <= idx:
            raise IndexError("Index out of range")
        indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        samples = []
        for i in indices:
            data = self.dataset[i]
            samples.append(data)
        return self.collate_fn(samples) 

    def __len__(self):
        return int(jnp.floor(len(self.dataset) / self.batch_size))

    def on_epoch_end(self):
        n = len(self.dataset)
        seq = np.arange(0, n)
        if self.shuffle:
            np.random.shuffle(seq)
            self.indices = seq
        else:
            self.indices = seq
