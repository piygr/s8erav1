from torchvision import transforms


def get_train_transforms(mean, std):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transforms


def get_test_transforms(mean, std):
    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return test_transforms