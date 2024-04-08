from torchvision.models import resnet18
import torch
import torchvision
import torchvision.transforms as transforms


from tunnel_methodsv2 import Rank, EarlyExit
from plotters import rank_plot, exits_plot

def analysis(rpath):

    # here you need to load your model with with pretrained weights
    model = resnet18()
    # define layers of the network you want to analyze
    layers = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)][::5]

    # here you need to load your data you want to analyze your model on

    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )
    train_data = torchvision.datasets.__dict__["CIFAR10"](
        root="./data/", train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.__dict__["CIFAR10"](
        root="./data/", train=False, download=True, transform=transform
    )

    # ---------------------------------------------------------------------------
    # Representations Spectra (Figure 2 -- numerical rank)
    # ---------------------------------------------------------------------------
    repr_spectra = Rank(
        test_data,
        model,
        layers,
        rpath=rpath,
        plotter = rank_plot
        )
    repr_spectra.analysis()
    repr_spectra.export('representations_spectra')
    repr_spectra.plot(name="representations_spectra")
    repr_spectra.clean_up()
    # ---------------------------------------------------------------------------
    # Early Exit ID (Figure 2 -- Linear probing ACC)
    # ---------------------------------------------------------------------------
    early_exit = EarlyExit(
        model=model,
        train_data=train_data,
        test_data=test_data,
        layers=layers,
        rpath=rpath,
        plotter = exits_plot
    )

    early_exit.analysis()
    early_exit.export(name="early_exits")
    early_exit.plot(name="early_exits")
    early_exit.clean_up()



   
if __name__ == "__main__":

    analysis(rpath="test_results")
