from torch.utils.data import DataLoader

from .dataset import MSCTDDataset
from .transforms import get_train_transforms, get_eval_transforms


def get_dataloaders(data_root, batch_size=32, num_workers=4):

    train_dataset = MSCTDDataset(
        data_root=data_root,
        split="train",
        transform=get_train_transforms()
    )

    dev_dataset = MSCTDDataset(
        data_root=data_root,
        split="dev",
        transform=get_eval_transforms()
    )

    test_dataset = MSCTDDataset(
        data_root=data_root,
        split="test",
        transform=get_eval_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, dev_loader, test_loader