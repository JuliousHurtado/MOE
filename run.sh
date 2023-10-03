############
# Argument #
############
DATASET='cifar100'
MODEL='simplecnn' #'simplecnn'
N_EXP=5


python main.py --seed 44 --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs 1 --use_moe --weigth_loss --use_moe_filter --memory_size 500 --buffer_mode random --num_proxy 2 --num_neighbours 5

for e in 1 2 3 4 5 10
do
    for m in 500 1000
    do
        for n in 5 50 100
        do
            for s in 44 45 46
            do
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --memory_size $m --buffer_mode random --num_proxy 2 --num_neighbours $n
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --memory_size $m --buffer_mode random --num_proxy 3 --num_neighbours $n
            done
        done
    done
done



for e in 1 2 3 4 5 10
do
    for m in 500 1000
    do
        for n in 5 50 100
        do
            for s in 44 45 46
            do
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --memory_size $m --buffer_mode top --num_proxy 2 --num_neighbours $n
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --memory_size $m --buffer_mode top --num_proxy 3 --num_neighbours $n
            done
        done
    done
done



for m in 500 1000
do
    for e in 1 2 3 4 5 10
    do
        for n in 5 50 100
        do
            for s in 44 45 46
            do
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --weigth_loss --memory_size $m --buffer_mode c_score --cscore_mode caws --num_proxy 2 --cscore_min_bucket 0.9 --num_neighbours $n
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --weigth_loss --memory_size $m --buffer_mode c_score --cscore_mode caws --num_proxy 2 --cscore_min_bucket 0.8 --num_neighbours $n
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --weigth_loss --memory_size $m --buffer_mode c_score --cscore_mode caws --num_proxy 2 --cscore_min_bucket 0.7 --num_neighbours $n
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --weigth_loss --memory_size $m --buffer_mode c_score --cscore_mode caws --num_proxy 2 --cscore_min_bucket 0.6 --num_neighbours $n
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --weigth_loss --memory_size $m --buffer_mode c_score --cscore_mode caws --num_proxy 2 --cscore_min_bucket 0.5 --num_neighbours $n
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --weigth_loss --memory_size $m --buffer_mode c_score --cscore_mode caws --num_proxy 2 --cscore_min_bucket 0.4 --num_neighbours $n
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --weigth_loss --memory_size $m --buffer_mode c_score --cscore_mode caws --num_proxy 2 --cscore_min_bucket 0.3 --num_neighbours $n
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --weigth_loss --memory_size $m --buffer_mode c_score --cscore_mode caws --num_proxy 2 --cscore_min_bucket 0.2 --num_neighbours $n
                python main.py --seed $s --model $MODEL --dataset $DATASET --n_experience $N_EXP --epochs $e --use_moe --weigth_loss --memory_size $m --buffer_mode c_score --cscore_mode caws --num_proxy 2 --cscore_min_bucket 0.1 --num_neighbours $n
            done
        done
    done
done
