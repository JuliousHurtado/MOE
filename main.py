from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision import transforms

from models.simpleCNN import SimpleCNN
from metrics.c_score import CScoreMetric

from avalanche.training.strategies import BaseStrategy
from avalanche.benchmarks.classic import SplitCIFAR10, SplitMNIST, SplitCIFAR100
from avalanche.training.plugins import GDumbPlugin
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics

from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

import argparse


def parse_train_args():
    parser = argparse.ArgumentParser("Positive Forgetting")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--model", type=str, default='simplecnn')
    parser.add_argument("--n_simplecnn", type=int, default=1)

    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--n_experience", type=int, default=2)

    parser.add_argument("--use_gdump", action="store_true")
    parser.add_argument("--gdump_memory", type=int, default=5000)

    args = parser.parse_args()

    return args

def get_dataset(args):
    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ])

        benchmark = SplitCIFAR10(n_experiences=args.n_experience, seed=args.seed, 
                    train_transform=train_transform, eval_transform=val_transform )
        
        return benchmark, 10, [train_transform, val_transform]
    
    if args.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ])

        benchmark = SplitCIFAR100(n_experiences=args.n_experience, seed=args.seed, 
                    train_transform=train_transform, eval_transform=val_transform )
        
        return benchmark, 100, [train_transform, val_transform]

def get_model(args, num_classes):
    if args.model == 'simplecnn':
        return SimpleCNN([32,64,128], args.n_simplecnn, num_classes=num_classes)
    
    assert False, "Model {} not found".format(args.model)

def get_plugins(args):
    plugins = []

    if args.use_gdump:
        plugins.append(GDumbPlugin(mem_size=args.gdump_memory))

    return plugins

def main():
    args = parse_train_args()

    benchmark, num_classes, transform = get_dataset(args)
    model = get_model(args, num_classes)
    plugins = get_plugins(args)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        CScoreMetric(args.dataset, transform[0], transform[1]),
        benchmark=benchmark,
        loggers=[InteractiveLogger()],
        strict_checks=False
    )

    cl_strategy = BaseStrategy(
        model, optimizer, criterion, device='cuda',
        train_mb_size = args.batch_size, eval_mb_size = args.batch_size,
        train_epochs = args.epochs,
        plugins = plugins,
        evaluator = eval_plugin
    )

    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(benchmark.test_stream))

    print(results)

if __name__ == "__main__":
    main()
