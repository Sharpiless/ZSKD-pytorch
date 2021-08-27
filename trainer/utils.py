import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomAffine, RandomRotation


def transformer(dataset):
    if dataset =='mnist':
        trans=transforms.Compose([  transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

        train_trans,test_trans=trans, trans
        
    elif dataset == 'cifar10' or dataset == 'cifar100':
        train_trans = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomAffine(degrees=0, translate=(0.0, 0.2), scale=(0.6, 1.0)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        test_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        
    return train_trans, test_trans


def adjust_learning_rate(optimizer, epoch):
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr