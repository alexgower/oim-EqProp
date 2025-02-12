import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy import*
from copy import*


########## ML DATASET GENERATION ##########

class CustomDataset(Dataset):
    # Class to create a custom dataset

    def __init__(self, images, labels=None):
        self.x = images
        self.y = labels

    def __getitem__(self, i):
        data = self.x[i, :]
        target = self.y[i]

        return (data, target)

    def __len__(self):
        return (len(self.x))


class ReshapeTransform:
    # Class to reshape the data
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class NormalizeTransform:
    # Class to normalize data to [-1,1] range
    def __init__(self, use_negative_one_to_one=False):
        self.use_negative_one_to_one = use_negative_one_to_one

    def __call__(self, x):
        # ToTensor already normalizes to [0,1]
        if self.use_negative_one_to_one:
            x = 2.0 * x - 1.0
        return x


class ReshapeTransformTarget:
    # Class to reshape the target labels to one-hot encoding on multiple output neurons per class if needed
    # __init__ stores the number of classes and the number of output neurons, __call__ does the one-hot encoding with this stored structure

    def __init__(self, number_classes, args):
        self.number_classes = number_classes
        self.outputlayer = args.layersList[2]

    def __call__(self, target):
        # e.g. target = 3

        target = torch.tensor(target).unsqueeze(0).unsqueeze(1)


        target_onehot = -1*torch.ones((1, self.number_classes))

        # target.long() is used to convert the target to a long (int) tensor
        # e.g. target_onehot.scatter_(1, target.long(), 1) = tensor([[-1., 1., -1., -1, -1., -1., -1., -1., -1., -1.]]) if target = 1
        # e.g. target_onehot.scatter_(1, target.long(), 1).repeat_interleave(int(self.outputlayer/self.number_classes)) = tensor([[-1.,-1.,-1.,-1.,1.,1.,1.,1.,-1.,-1.,.....,-1.,-1.]]) if target = 1
        # i.e. we repeat -1. 4 times for class 0, and 1. 4 times for class 1, ...
        return target_onehot.scatter_(1, target.long(), 1).repeat_interleave(int(self.outputlayer/self.number_classes)).squeeze(0)



class DefineDataset(Dataset):
    '''
    Class to hold data and labels for the dataset, with the possibility to apply transformations to the data and labels too
    '''
    
    def __init__(self, images, labels=None, transforms=None, target_transforms=None):
        self.x = images
        self.y = labels
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, i):
        data = self.x[i, :]
        target = self.y[i]

        if self.transforms:
            data = self.transforms(data)

        if self.target_transforms:
            target = self.target_transforms(target)

        if self.y is not None:
            return (data, target)
        else:
            return data

    def __len__(self):
        return (len(self.x))






def generate_digits(args):
    '''
    Generate the dataloaders for digits dataset
    '''

    digits = load_digits()

    # Random_state sets reproducible seed for shuffle
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1, random_state=10, shuffle=True)

    # Apparently scikitlearn digits are in [0,16] so normalise by 8 to get [0,2] so mappable to [-1,1]
    # BUT NOTE THAT CURRENTLY THIS FUNCTION WILL JUST RETURN THE DIGITS IMAGES WITH EACH PIXEL BETWEEN 0 AND 2
    normalisation = 8 
    x_train, x_test = x_train / normalisation, x_test / normalisation

    # Use ReshapeTransformTarget to reshape the target labels to one-hot encoding on multiple output neurons per class if needed
    train_data = DefineDataset(x_train, labels=y_train, target_transforms=ReshapeTransformTarget(10, args))
    test_data = DefineDataset(x_test, labels=y_test, target_transforms=ReshapeTransformTarget(10, args))

    ## Data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader








def generate_mnist(args):
    '''
    Generate mnist dataloaders
    If mnist_positive_negative_remapping is True, remaps pixel values from [0,1] to [-1,1]
    '''
    N_class = 10

    # Use custom training and test data size
    N_data_train = args.N_data_train
    N_data_test = args.N_data_test

    with torch.no_grad():
        # Base transforms
        transforms_base = [
            torchvision.transforms.ToTensor(),
            NormalizeTransform(use_negative_one_to_one=args.mnist_positive_negative_remapping),
            ReshapeTransform((-1,))
        ]

        if args.mnist_positive_negative_remapping:
            print("\nMNIST data will be normalized to [-1,1] range")
        else:
            print("\nMNIST data will be normalized to [0,1] range")

        # Add augmentation for training if needed
        transforms_train = transforms_base.copy()
        if args.data_augmentation:
            transforms_train.insert(-1, torchvision.transforms.RandomAffine( # Insert in SECOND LAST position
                10, translate=(0.04, 0.04),
                scale=None, shear=None,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                fill=0
            ))

        ### Training data

        # Load the MNIST dataset, apply the transformations if appropraite from above, and target transformations to multiple output neurons per class if needed
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                transform=torchvision.transforms.Compose(transforms_train),
                                                target_transform=ReshapeTransformTarget(10, args))


        # # Now reduce the number of training data points to N_data_train, but keep the same number of data points per class
        # mnist_train_data, mnist_train_targets, comp = torch.empty(N_data_train,28,28,dtype=mnist_train.data.dtype), torch.empty(N_data_train,dtype=mnist_train.targets.dtype), torch.zeros(N_class)
        # idx_0, idx_1 = 0, 0
        # while idx_1 < N_data_train:
        #     class_data = mnist_train.targets[idx_0]
        #     if comp[class_data] < int(N_data_train/N_class):
        #         mnist_train_data[idx_1,:,:] = mnist_train.data[idx_0,:,:].clone()
        #         mnist_train_targets[idx_1] = class_data.clone()
        #         comp[class_data] += 1
        #         idx_1 += 1
        #     idx_0 += 1
        # mnist_train.data, mnist_train.targets = mnist_train_data, mnist_train_targets


        # Reduce training dataset size
        indices = []
        comp = torch.zeros(N_class)
        for idx, target in enumerate(mnist_train.targets):
            if comp[target] < N_data_train / N_class:
                indices.append(idx)
                comp[target] += 1
            if len(indices) == N_data_train:
                break
        
        mnist_train.data = mnist_train.data[indices]
        mnist_train.targets = mnist_train.targets[indices]

        ### Testing data
        mnist_test = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(transforms_base),
            target_transform=ReshapeTransformTarget(10, args)
        )

        # # Now reduce the number of testing data points to N_data_test, but keep the same number of data points per class
        # mnist_test_data, mnist_test_targets, comp = torch.empty(N_data_test,28,28,dtype=mnist_test.data.dtype), torch.empty(N_data_test,dtype=mnist_test.targets.dtype), torch.zeros(N_class)
        # idx_0, idx_1 = 0, 0
        # while idx_1 < N_data_test:
        #     class_data = mnist_test.targets[idx_0]
        #     if comp[class_data] < int(N_data_test/N_class):
        #         mnist_test_data[idx_1,:,:] = mnist_test.data[idx_0,:,:].clone()
        #         mnist_test_targets[idx_1] = class_data.clone()
        #         comp[class_data] += 1
        #         idx_1 += 1
        #     idx_0 += 1

        # mnist_test.data, mnist_test.targets = mnist_test_data, mnist_test_target


        # Reduce test dataset size
        indices = []
        comp = torch.zeros(N_class)
        for idx, target in enumerate(mnist_test.targets):
            if comp[target] < N_data_test / N_class:
                indices.append(idx)
                comp[target] += 1
            if len(indices) == N_data_test:
                break
        
        mnist_test.data = mnist_test.data[indices]
        mnist_test.targets = mnist_test.targets[indices]

        # Create the data loaders
        train_loader = torch.utils.data.DataLoader(
            mnist_train,
            batch_size=args.batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            mnist_test,
            batch_size=args.batch_size,
            shuffle=False
        )

        # Verify data ranges
        for batch, _ in train_loader:
            print(f"Data range verification:")
            print(f"Min pixel value: {batch.min():.4f}")
            print(f"Max pixel value: {batch.max():.4f}")
            break

        return train_loader, test_loader, mnist_train
    


