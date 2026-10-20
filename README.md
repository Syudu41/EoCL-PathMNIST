# EoCL-PathMNIST
This is a replication of EoCL on the PathMNIST dataset. 

> [!IMPORTANT]
> Under Construction!!

### Dataset:

> [!NOTE] 
> Refer to [MedMNIST dataset](https://medmnist.com/).

```python 
pip install medmnist
```

#### Usage Example

```python
# To use the standard 28-size (MNIST-like) version utilizing the downloaded files:
from medmnist import PathMNIST
train_dataset = PathMNIST(split="train")

# To enable automatic downloading by setting `download=True`:
from medmnist import NoduleMNIST3D
val_dataset = NoduleMNIST3D(split="val", download=True)

# Alternatively, you can access MedMNIST+ with larger image sizes by specifying the `size` parameter:
from medmnist import ChestMNIST
test_dataset = ChestMNIST(split="test", download=True, size=224)
```

### Workflow

- Image goes here

