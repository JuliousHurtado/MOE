from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import torch

from models.simpleCNN import SimpleCNN
from models.mnist_model import SimpleNN
from models.resnet import resnet18

from avalanche.training import Naive
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100, CORe50
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics

from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger

from avalanche.training.storage_policy import ExperienceBalancedBuffer, ClassBalancedBuffer, \
    ReservoirSamplingBuffer

from training.storage_policy.c_score_policy import CScoreBuffer
from training.storage_policy.moe_policy import MOEBuffer
from training.storage_policy.mof_policy import MeanOfFeaturesBuffer
from training.storage_policy.max_loss import MaxLossBuffer
from training.storage_policy.all_right import AllRightBuffer

from training.plugins.replay_mod import ReplayPluginMod
from training.plugins.moe_plugin import MOEPlugin
from training.plugins.gss import GSS_greedyPlugin
from training.plugins.extra_plugin import ExtraPlugin

from datasets.get_dataset import get_mnist, _default_mnist_train_transform, _default_mnist_eval_transform, \
                        get_tiny_imagenet

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
    parser.add_argument("--use_gss", action="store_true")

    parser.add_argument("--use_replay", action="store_true")
    parser.add_argument("--memory_size", type=int, default=5000)
    parser.add_argument("--buffer_mode", type=str, default='')
    parser.add_argument("--cscore_mode", type=str, default="random")
    parser.add_argument("--cscore_min_bucket", type=float, default=0.9)
    parser.add_argument("--max_loss_descending", action="store_true")

    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--use_moe_filter", action="store_true")
    parser.add_argument("--weigth_loss", action="store_true")
    parser.add_argument("--rem_neighbours", type=int, default=20)

    parser.add_argument("--true_labels", action="store_false")

    parser.add_argument("--use_proxy", action="store_true")
    parser.add_argument("--num_proxy", type=int, default=0)
    parser.add_argument("--num_neighbours", type=int, default=5)
    parser.add_argument("--perct_caws", type=str, default='mean')

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_memory", action="store_true")
    parser.add_argument("--save_score", action="store_true")

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
        
        return benchmark, 10, [train_transform, val_transform], [3,32,32]
    
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
        
        return benchmark, 100, [train_transform, val_transform], [3,32,32]
    
    if args.dataset == 'mnist':
        benchmark = get_mnist(args.n_experience)
        return benchmark, 10, [_default_mnist_train_transform, _default_mnist_eval_transform], [1,28,28]
    
    if args.dataset == 'tiny_imagenet':
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
        root='/home/jahurtado/codes/data/tiny_imagenet/tiny-imagenet-200'
        benchmark = get_tiny_imagenet(root, args.n_experience, train_transform=train_transform,
                                    eval_transform=val_transform)
        return benchmark, 200, [train_transform, val_transform], [3,64,64]
    
    if args.dataset == 'core50':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        path = '/mnt/nas/GrimaRepo/datasets/core50'
        benchmark = CORe50(scenario='nc', mini=True, dataset_root=path,
                        train_transform=train_transform, eval_transform=val_transform)

        return benchmark, 50, [train_transform, val_transform], [3,32,32]

def get_model(args, num_classes):
    if args.dataset == 'mnist':
        return SimpleNN(num_classes=num_classes)
    if args.model == 'resnet':
        return resnet18(num_classes)
    if args.model == 'simplecnn':
        return SimpleCNN([32,64,128], args.n_simplecnn, num_classes=num_classes, dataset=args.dataset)
    
    assert False, "Model {} not found".format(args.model)

def get_storage_policy(args):
    if args.buffer_mode == 'c_score':
        return CScoreBuffer(max_size = args.memory_size,
                    name_dataset = args.dataset,
                    mode = args.cscore_mode,
                    min_bucket = args.cscore_min_bucket,
                    use_proxy = args.use_proxy, 
                    num_proxy = args.num_proxy,
                    num_neighbours = args.num_neighbours,
                    perct_caws = args.perct_caws,
                    true_labels = args.true_labels,
                    save_score = args.save_score
            )

    if args.use_moe:
        if args.buffer_mode == 'all_right':
            return AllRightBuffer(
                max_size = args.memory_size,
                adaptive_size = True
            )
        else:
            return MOEBuffer(max_size = args.memory_size,
                        num_proxy = args.num_proxy,
                        num_neighbours = args.num_neighbours,
                        mode = args.buffer_mode,
                        filter_moe = args.use_moe_filter,
                        rem_neighbours = args.rem_neighbours 
                )

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
    
    if args.buffer_mode == 'max_loss':
        return MaxLossBuffer(
                max_size = args.memory_size,
                adaptive_size = True,
                descending = args.max_loss_descending
        )

    assert False, f"Reaply buffer {args.buffer_mode} unknow"

def get_strategy(args, model, optimizer, criterion, eval_plugin, data_size, device = 'cuda'):
    plugins = []
    strategy = Naive
    name_file = 'temp.pth'

    if args.use_naive:
        name_file = "naive_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, args.dataset, args.epochs, args.seed)

    if args.use_moe:
        storage_policy = get_storage_policy(args)
        if args.buffer_mode == 'extra':
            plugins.append(ExtraPlugin(mem_size = args.memory_size, \
                                    batch_size_mem = args.batch_size // 2,
                                    storage_policy = storage_policy,
                                    task_balanced_dataloader = True,
                                    weigth_loss = args.weigth_loss))
        else:
            plugins.append(MOEPlugin(mem_size = args.memory_size, \
                                    batch_size_mem = args.batch_size // 2,
                                    storage_policy = storage_policy,
                                    task_balanced_dataloader = True,
                                    weigth_loss = args.weigth_loss))

        if args.buffer_mode == 'c_score':
            if args.cscore_mode == 'caws':
                name_file = "moe_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                            args.dataset, args.epochs, args.memory_size, args.buffer_mode, args.num_proxy, \
                            args.num_neighbours, args.weigth_loss, args.use_moe_filter, args.cscore_mode, \
                            args.cscore_min_bucket, args.seed)
            else:
                name_file = "moe_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                            args.dataset, args.epochs, args.memory_size, args.buffer_mode, args.num_proxy, \
                            args.num_neighbours, args.weigth_loss, args.use_moe_filter, args.cscore_mode, \
                            args.perct_caws, args.true_labels, args.seed)
        else:
            name_file = "moe_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, args.dataset, args.epochs,
                            args.memory_size, args.buffer_mode, args.num_proxy, args.num_neighbours, args.weigth_loss, args.use_moe_filter, args.seed)

    if args.use_replay:
        storage_policy = get_storage_policy(args)
        plugins.append(ReplayPluginMod(mem_size = args.memory_size, \
                                batch_size = args.batch_size // 2,
                                batch_size_mem = args.batch_size // 2,
                                storage_policy = storage_policy,
                                task_balanced_dataloader = True))
        if args.cscore_mode == 'caws-perc':
            name_file = "replay_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                args.dataset, args.epochs, args.memory_size, args.buffer_mode, args.cscore_mode, \
                args.cscore_min_bucket, args.perct_caws, args.num_proxy, args.num_neighbours, args.seed)

        elif args.buffer_mode == 'moe':
            name_file = "replay_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                args.dataset, args.epochs, args.memory_size, args.buffer_mode, args.num_proxy, \
                args.num_neighbours, args.seed)
        
        elif args.buffer_mode == 'max_loss':
            name_file = "replay_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                args.dataset, args.epochs, args.memory_size, args.buffer_mode, args.max_loss_descending, \
                args.seed)

        elif args.use_proxy:
            name_file = "replay_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                args.dataset, args.epochs, args.memory_size, args.buffer_mode, args.cscore_mode, \
                args.cscore_min_bucket, args.num_proxy, args.num_neighbours, args.seed)

        elif args.cscore_mode == 'caws':
            name_file = "replay_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                args.dataset, args.epochs, args.memory_size, args.buffer_mode, args.cscore_mode, \
                args.cscore_min_bucket, args.seed)

        else:
            name_file = "replay_{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                args.dataset, args.epochs, args.memory_size, args.buffer_mode, args.cscore_mode, \
                args.seed)
    
    if args.use_gss:
        plugins.append(GSS_greedyPlugin(
            mem_size=args.memory_size, mem_strength=5, input_size=data_size
        ))
        name_file = "gss_{}_{}_{}_{}_{}_{}.pth".format(args.model, args.n_experience, \
                args.dataset, args.epochs, args.memory_size, \
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

    benchmark, num_classes, _, data_size = get_dataset(args)
    model = get_model(args, num_classes)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum) 
    criterion = CrossEntropyLoss()

    loggers = []
    if args.verbose:
        loggers.append(InteractiveLogger())

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        benchmark=benchmark,
        strict_checks=False,
        loggers = loggers
    )

    cl_strategy, name_file = get_strategy(args, model, optimizer, criterion, eval_plugin, data_size)
    cl_strategy.save_file_name = os.path.join('./results', name_file)

    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience, num_workers=2)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(benchmark.test_stream))

        # original_idx = torch.tensor(cl_strategy.plugins[0].storage_policy.buffer_groups[18].buffer._dataset.dataset._dataset_list[0]._indices)
        # current_idx = [ int(i) for i in cl_strategy.plugins[0].storage_policy.buffer_groups[18].buffer._indices ]
        # print(original_idx)
        # print(current_idx)
        # print(original_idx[current_idx])

    if args.save_memory:
        save_indices = {}
        for c in cl_strategy.plugins[0].storage_policy.buffer_groups.keys():
            original_idx = torch.tensor(cl_strategy.plugins[0].storage_policy.buffer_groups[c].buffer._dataset.dataset._dataset_list[0]._indices)
            current_idx = [ int(i) for i in cl_strategy.plugins[0].storage_policy.buffer_groups[c].buffer._indices ]
            save_indices[c] = original_idx[current_idx]

        if args.cscore_mode in ['upper', 'lower']:
            save_file_name = 'indices_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(args.dataset, args.epochs, args.memory_size, args.use_replay, args.use_gss,
                                                                    args.buffer_mode, args.use_proxy, args.num_proxy, args.cscore_mode)
        else:
            save_file_name = 'indices_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(args.dataset, args.epochs, args.memory_size, args.use_replay, args.use_gss,
                                                                    args.buffer_mode, args.use_proxy, args.num_proxy, args.num_neighbours)
        save_file_name = os.path.join('./results', save_file_name)
        torch.save(save_indices, save_file_name)
    else:
        # top_results = torch.load(cl_strategy.save_file_name)
        # for k,v in results[-1].items():
        #     if 'Top1_Acc_Stream/eval_phase' in k:
        #         print(v)
        top_results = {}
        top_results['benchmark_results'] = results
        top_results['args'] = args
        torch.save(top_results, cl_strategy.save_file_name)

    if args.save_score:
        save_file_name = 'scores_{}_{}_{}_{}_{}_{}_{}.pth'.format(args.dataset, args.epochs, args.memory_size, args.cscore_min_bucket,
                                                                    args.num_proxy, args.cscore_mode, args.num_neighbours)
        
        save_scores = cl_strategy.plugins[0].storage_policy.scores_saved

        save_file_name = os.path.join('./results', save_file_name)
        torch.save(save_scores, save_file_name)

    for k,v in results[-1].items():
        if 'Top1_Acc_Stream/eval_phase' in k:
            print(v)

if __name__ == "__main__":
    main()
