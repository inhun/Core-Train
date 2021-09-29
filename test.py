from utils.datasets import *
from utils.utils import *
from utils.parse_config import *




data_config = parse_data_config('config/vehicle.data')
train_path = data_config["train"]

dataset = ListDataset(train_path, augment=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1,
    pin_memory=True
    )


if __name__ == '__main__':
    for batch_i, (_, imgs, targets, targets_distance) in enumerate(dataloader):
        print(targets)
        print(targets_distance)



