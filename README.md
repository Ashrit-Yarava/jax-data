# Jax Datasets
---
## data
---
To load data, first subclass the `jax_data.Dataset` class, implementing the __len__ and __getitem__ methods.
The `jax_data.Dataloader` class is a simplified adaptation of torch.utils.data.DataLoader.
