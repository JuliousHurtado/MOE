DATASET='cifar10'
EPOCHS=15
N_EXPERIENCE=5


##########
# Sec4.1 #
##########
python main.py --dataset $DATASET --n_experience $N_EXPERIENCE --seed 42 --use_naive --epochs $EPOCHS

python main.py --dataset $DATASET --n_experience $N_EXPERIENCE --seed 42 --use_gdumb_mod --gdumb_memory 2000 --epochs $EPOCHS --gdumb_buffer_mode random
python main.py --dataset $DATASET --n_experience $N_EXPERIENCE --seed 42 --use_gdumb_mod --gdumb_memory 2000 --epochs $EPOCHS --gdumb_buffer_mode upper # high-c
python main.py --dataset $DATASET --n_experience $N_EXPERIENCE --seed 42 --use_gdumb_mod --gdumb_memory 2000 --epochs $EPOCHS --gdumb_buffer_mode lower # low-c

python main.py --dataset $DATASET --n_experience $N_EXPERIENCE --seed 42 --use_replay --replay_memory 2000 --epochs $EPOCHS --use_custom_replay_buffer --replay_buffer_mode random
python main.py --dataset $DATASET --n_experience $N_EXPERIENCE --seed 42 --use_replay --replay_memory 2000 --epochs $EPOCHS --use_custom_replay_buffer --replay_buffer_mode upper
python main.py --dataset $DATASET --n_experience $N_EXPERIENCE --seed 42 --use_replay --replay_memory 2000 --epochs $EPOCHS --use_custom_replay_buffer --replay_buffer_mode lower

python main.py --dataset $DATASET --n_experience $N_EXPERIENCE --seed 42 --use_agem_mod --agem_pattern_per_exp 400 --epochs $EPOCHS --agem_buffer_mode random
python main.py --dataset $DATASET --n_experience $N_EXPERIENCE --seed 42 --use_agem_mod --agem_pattern_per_exp 400 --epochs $EPOCHS --agem_buffer_mode upper
python main.py --dataset $DATASET --n_experience $N_EXPERIENCE --seed 42 --use_agem_mod --agem_pattern_per_exp 400 --epochs $EPOCHS --agem_buffer_mode lower



##########
# Sec4.2 #
##########
########
# CAWS #
########
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode caws --replay_memory 2000 --replay_min_bucket 0.9
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode caws --replay_memory 2000 --replay_min_bucket 0.8
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode caws --replay_memory 2000 --replay_min_bucket 0.7
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode caws --replay_memory 2000 --replay_min_bucket 0.6
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode caws --replay_memory 2000 --replay_min_bucket 0.5
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode caws --replay_memory 2000 --replay_min_bucket 0.4
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode caws --replay_memory 2000 --replay_min_bucket 0.3
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode caws --replay_memory 2000 --replay_min_bucket 0.2
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode caws --replay_memory 2000 --replay_min_bucket 0.1
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode caws --replay_memory 2000 --replay_min_bucket 0.0

########
# COBS #
########
python main.py --n_experience $N_EXPERIENCE --seed 42 --use_replay --epochs $EPOCHS --dataset $DATASET --use_custom_replay_buffer --replay_buffer_mode cobs --replay_memory 2000 



##########
# Sec4.3 #
##########
########
# MIR  #
########
python main.py --dataset $DATASET --n_experience 5 --seed 42 --use_mir --replay_memory 500 --epochs 1 --use_custom_replay_buffer --replay_buffer_mode random --lr 0.1 --momentum 0 --batch_size 20
python main.py --dataset $DATASET --n_experience 5 --seed 42 --use_mir --replay_memory 500 --epochs 1 --use_custom_replay_buffer --replay_buffer_mode upper --lr 0.1 --momentum 0 --batch_size 20
python main.py --dataset $DATASET --n_experience 5 --seed 42 --use_mir --replay_memory 500 --epochs 1 --use_custom_replay_buffer --replay_buffer_mode lower --lr 0.1 --momentum 0 --batch_size 20
python main.py --dataset $DATASET --n_experience 5 --seed 42 --use_mir --replay_memory 500 --epochs 1 --use_custom_replay_buffer --replay_buffer_mode caws --lr 0.1 --momentum 0 --batch_size 20 --replay_min_bucket 0.5
python main.py --dataset $DATASET --n_experience 5 --seed 42 --use_mir --replay_memory 500 --epochs 1 --use_custom_replay_buffer --replay_buffer_mode cobs --lr 0.1 --momentum 0 --batch_size 20

python main.py --dataset $DATASET --n_experience 5 --seed 42 --use_mir --use_mir_replay --replay_memory 500 --epochs 1 --use_custom_replay_buffer --replay_buffer_mode random --lr 0.1 --momentum 0 --batch_size 20
python main.py --dataset $DATASET --n_experience 5 --seed 42 --use_mir --use_mir_replay --replay_memory 500 --epochs 1 --use_custom_replay_buffer --replay_buffer_mode upper --lr 0.1 --momentum 0 --batch_size 20
python main.py --dataset $DATASET --n_experience 5 --seed 42 --use_mir --use_mir_replay --replay_memory 500 --epochs 1 --use_custom_replay_buffer --replay_buffer_mode lower --lr 0.1 --momentum 0 --batch_size 20
python main.py --dataset $DATASET --n_experience 5 --seed 42 --use_mir --use_mir_replay --replay_memory 500 --epochs 1 --use_custom_replay_buffer --replay_buffer_mode caws --lr 0.1 --momentum 0 --batch_size 20 --replay_min_bucket 0.5
python main.py --dataset $DATASET --n_experience 5 --seed 42 --use_mir --use_mir_replay --replay_memory 500 --epochs 1 --use_custom_replay_buffer --replay_buffer_mode cobs --lr 0.1 --momentum 0 --batch_size 20