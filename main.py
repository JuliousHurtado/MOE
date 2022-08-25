from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import torch

from models.simpleCNN import SimpleCNN
from models.mnist_model import SimpleNN
from models.resnet import resnet18

from avalanche.training import Naive
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics

from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger

from avalanche.training.storage_policy import ExperienceBalancedBuffer, ClassBalancedBuffer, \
    ReservoirSamplingBuffer

from training.storage_policy.c_score_policy import CScoreBuffer
from training.storage_policy.mof_policy import MeanOfFeaturesBuffer
from training.plugins.replay_mod import ReplayPluginMod

from datasets.get_dataset import get_mnist, _default_mnist_train_transform, _default_mnist_eval_transform, \
        get_imagenet, _default_imgenet_train_transform, _default_imgenet_val_transform

import argparse
import os


def parse_train_args():
    parser = argparse.ArgumentParser("Positive Forgetting")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--c_score_top_percentaje", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--model", type=str, default='simplecnn')
    parser.add_argument("--n_simplecnn", type=int, default=1)

    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--n_experience", type=int, default=2)

    parser.add_argument("--use_naive", action="store_true")

    parser.add_argument("--use_replay", action="store_true")
    parser.add_argument("--memory_size", type=int, default=5000)
    parser.add_argument("--buffer_mode", type=str, default='')
    parser.add_argument("--cscore_mode", type=str, default="random")
    parser.add_argument("--cscore_mix_upper", type=float, default=0.5)
    parser.add_argument("--cscore_min_bucket", type=float, default=0.9)

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
    
    if args.dataset == 'mnist':
        benchmark = get_mnist(args.n_experience)
        return benchmark, 10, [_default_mnist_train_transform, _default_mnist_eval_transform]
    
    if args.dataset == 'imagenet':
        benchmark = get_imagenet(args.n_experience)
        return benchmark, 1000, [_default_imgenet_train_transform, _default_imgenet_val_transform]

def get_model(args, num_classes):
    if args.dataset == 'mnist':
        return SimpleNN(num_classes=num_classes)
    if args.model == 'resnet':
        return resnet18(num_classes)
    if args.model == 'simplecnn':
        return SimpleCNN([32,64,128], args.n_simplecnn, num_classes=num_classes)
    
    assert False, "Model {} not found".format(args.model)

def get_storage_policy(args):
    if args.buffer_mode == 'cls_balance': # ring_buffer
        return ClassBalancedBuffer(
                max_size = args.memory_size,
                adaptive_size = True
            )

    if args.buffer_mode == 'task_balance':
        return ExperienceBalancedBuffer(
                max_size = args.memory_size,
                adaptive_size = True
            )

    if args.buffer_mode == 'reservoir': # random
        return ReservoirSamplingBuffer(
                max_size = args.memory_size
            )

    if args.buffer_mode == 'mean_features':
        return MeanOfFeaturesBuffer(
                max_size = args.memory_size,
                adaptive_size = True
        )
    
    if args.buffer_mode == 'c_score':
        return CScoreBuffer(max_size = args.memory_size,
                    name_dataset = args.dataset,
                    mode = args.cscore_mode,
                    mix_upper = args.cscore_mix_upper,
                    min_bucket = args.cscore_min_bucket
            )
        
    assert False, f"Reaply buffer {args.buffer_mode} unknow"

def get_strategy(args, model, optimizer, criterion, eval_plugin, device = 'cuda'):
    plugins = []
    strategy = Naive

    if args.use_naive:
        name_file = "naive_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, args.dataset, args.epochs, args.seed)

    if args.use_replay:
        storage_policy = get_storage_policy(args)
        plugins.append(ReplayPluginMod(mem_size = args.memory_size, \
                                batch_size = args.batch_size // 2,
                                batch_size_mem = args.batch_size // 2,
                                storage_policy = storage_policy,
                                task_balanced_dataloader = True))

        if args.cscore_mode == 'mix':
            name_file = "replay_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                args.dataset, args.epochs, args.memory_size, args.cscore_mode, \
                args.replay_mix_upper, args.seed)

        elif args.cscore_mode == 'caws':
            name_file = "replay_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                args.dataset, args.epochs, args.memory_size, args.cscore_mode, \
                args.replay_min_bucket, args.seed)

        else:
            name_file = "replay_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                args.dataset, args.epochs, args.memory_size, args.buffer_mode ,args.cscore_mode, \
                args.seed)

    cl_strategy = strategy(
            model, optimizer, criterion, device = device,
            plugins = plugins,
            train_mb_size = args.batch_size, eval_mb_size = args.batch_size,
            train_epochs = args.epochs,
            evaluator = eval_plugin
        )

    return cl_strategy, name_file

def main():
    args = parse_train_args()
    print(args)

    benchmark, num_classes, _ = get_dataset(args)
    model = get_model(args, num_classes)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum) 
    criterion = CrossEntropyLoss()

    # loggers = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        # CScoreMetric(args.dataset, transform[0], transform[1], top_percentaje=args.c_score_top_percentaje),
        benchmark=benchmark,
        strict_checks=False,
        # loggers = loggers
    )

    cl_strategy, name_file = get_strategy(args, model, optimizer, criterion, eval_plugin)
    cl_strategy.save_file_name = os.path.join('./results', name_file)

    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(benchmark.test_stream))

    # top_results = torch.load(cl_strategy.save_file_name)
    top_results = {}
    top_results['benchmark_results'] = results
    top_results['args'] = args

    torch.save(top_results, cl_strategy.save_file_name)

if __name__ == "__main__":
    main()
