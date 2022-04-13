from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import torch

from models.simpleCNN import SimpleCNN
from metrics.c_score import CScoreMetric

from avalanche.training.strategies import BaseStrategy, ICaRL
from avalanche.training.plugins import GDumbPlugin, ReplayPlugin, EWCPlugin, LwFPlugin, AGEMPlugin
from avalanche.benchmarks.classic import SplitCIFAR10, SplitMNIST, SplitCIFAR100
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics

from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin

from training.storage_policy.c_score_policy import CScoreBuffer
from training.plugins.agem_mod import AGEMPluginMod
from training.plugins.gdumb_mod import GDumbPluginMod

import argparse
import os


def parse_train_args():
    parser = argparse.ArgumentParser("Positive Forgetting")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--c_score_top_percentaje", type=float, default=0.2)

    parser.add_argument("--model", type=str, default='simplecnn')
    parser.add_argument("--n_simplecnn", type=int, default=1)

    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--n_experience", type=int, default=2)

    parser.add_argument("--use_naive", action="store_true")

    parser.add_argument("--use_gdumb", action="store_true")
    parser.add_argument("--use_gdumb_mod", action="store_true")
    parser.add_argument("--gdumb_memory", type=int, default=5000)
    parser.add_argument("--gdumb_buffer_mode", type=str, default="random")

    parser.add_argument("--use_replay", action="store_true")
    parser.add_argument("--replay_memory", type=int, default=5000)
    parser.add_argument("--use_custom_replay_buffer", action="store_true")
    parser.add_argument("--replay_buffer_mode", type=str, default="random")

    parser.add_argument("--use_ewc", action="store_true")
    parser.add_argument("--ewc_lambda", type=float, default=1)
    parser.add_argument("--ewc_mode", type=str, default="onlinesum") # separate onlinesum onlineweightedsum
    parser.add_argument("--ewc_decay_factor", type=float, default=0.1)

    parser.add_argument("--use_lwf", action="store_true")
    parser.add_argument("--lwf_alpha", type=float, default=0.5)
    parser.add_argument("--lwf_temperature", type=float, default=1)

    parser.add_argument("--use_agem", action="store_true")
    parser.add_argument("--use_agem_mod", action="store_true")
    parser.add_argument("--agem_pattern_per_exp", type=int, default=1000)
    parser.add_argument("--agem_sample_size", type=int, default=128)
    parser.add_argument("--agem_buffer_mode", type=str, default="random")

    parser.add_argument("--use_icarl", action="store_true")
    parser.add_argument("--icarl_memory_size", type=int, default=5000)
    parser.add_argument("--icarl_fixed_memory", action="store_false")

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

def get_storage_policy(args):
    if args.use_custom_replay_buffer:
        return CScoreBuffer(max_size = args.replay_memory,
                    name_dataset = args.dataset,
                    mode = args.replay_buffer_mode)
    
    return None

def get_strategy(args, model, optimizer, criterion, eval_plugin, device = 'cuda'):
    plugins = []
    strategy = BaseStrategy

    if args.use_naive:
        return strategy(
            model, optimizer, criterion, device = device,
            train_mb_size = args.batch_size, eval_mb_size = args.batch_size,
            train_epochs = args.epochs,
            evaluator = eval_plugin
        ), "naive_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, args.dataset, args.epochs, args.seed)

    if args.use_gdumb:
        plugins.append(GDumbPlugin(mem_size = args.gdumb_memory))
        name_file = "gdumb_{}_{}_{}_{}_{}_{}.pth".format(args.model, \
            args.n_experience, args.dataset, args.epochs, \
            args.gdumb_memory, args.seed)

    if args.use_gdumb_mod:
        plugins.append(GDumbPluginMod(mem_size = args.gdumb_memory, 
            name_dataset = args.dataset, mode = args.gdumb_buffer_mode))
        name_file = "gdumb_mod_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, \
            args.n_experience, args.dataset, args.epochs, args.gdumb_memory, \
            args.gdumb_buffer_mode, args.seed)

    if args.use_replay:
        storage_policy = get_storage_policy(args)
        plugins.append(ReplayPlugin(mem_size = args.replay_memory, \
                                storage_policy = storage_policy))
        name_file = "replay_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
            args.dataset, args.epochs, args.replay_memory, args.replay_buffer_mode, \
            args.seed)

    if args.use_ewc:
        plugins.append(EWCPlugin(ewc_lambda = args.ewc_lambda, mode = args.ewc_mode,
                decay_factor = args.ewc_decay_factor))
        name_file = "ewc_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, 
                    args.dataset, args.epochs, 
                    args.ewc_lambda, args.ewc_mode , args.ewc_decay_factor, args.seed)

    if args.use_lwf:
        plugins.append(LwFPlugin(alpha = args.lwf_alpha, temperature = args.lwf_temperature))
        name_file = "lwf_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience,
                    args.dataset, args.epochs, 
                    args.lwf_alpha, args.lwf_temperature, args.seed)

    if args.use_agem:
        plugins.append(AGEMPlugin(patterns_per_experience = args.agem_pattern_per_exp, 
                sample_size = args.agem_sample_size))
        name_file = "agem_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience,
                    args.dataset, args.epochs, 
                    args.agem_pattern_per_exp, args.agem_sample_size, args.seed)
    
    if args.use_agem_mod:
        plugins.append(AGEMPluginMod(patterns_per_experience = args.agem_pattern_per_exp, 
                sample_size = args.agem_sample_size, mode = args.agem_buffer_mode,
                name_dataset = args.dataset))
        name_file = "agem_mod_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, 
                    args.n_experience, 
                    args.dataset, args.epochs, 
                    args.agem_pattern_per_exp, args.agem_sample_size, 
                    args.agem_buffer_mode, args.seed)

    if args.use_icarl:
        return ICaRL(
            model.features, model.classifier, optimizer, device = device,
            memory_size = args.icarl_memory_size,
            fixed_memory = args.icarl_fixed_memory,
            buffer_transform = None,
            train_mb_size = args.batch_size, eval_mb_size = args.batch_size,
            train_epochs = args.epochs,
            evaluator = eval_plugin
        ), "icarl_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, 
                    args.dataset, args.epochs, 
                    args.icarl_memory_size, args.icarl_fixed_memory, args.seed)

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

    benchmark, num_classes, transform = get_dataset(args)
    model = get_model(args, num_classes)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        CScoreMetric(args.dataset, transform[0], transform[1], top_percentaje=args.c_score_top_percentaje),
        benchmark=benchmark,
        loggers=[TextLogger()],
        strict_checks=False
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

    # print(results)

    top_results = torch.load(cl_strategy.save_file_name)
    top_results['benchmark_results'] = results
    top_results['args'] = args

    torch.save(top_results, cl_strategy.save_file_name)

if __name__ == "__main__":
    main()
