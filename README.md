# It's all About Consistency: A Study on Memory Composition for Replay-Based Methods in Continual Learning

Continual Learning methods strive to mitigate Catastrophic Forgetting (CF), where knowledge from previously learned tasks is lost when learning a new one. Among those algorithms, some maintain a subset of samples from previous tasks when training. These samples are referred to as a \textit{memory}. These methods have shown outstanding performance while being conceptually simple and easy to implement. Yet, despite their popularity, little has been done to understand which elements to be included into the memory. Currently, this memory is often filled via random sampling with no guiding principles that may aid in retaining previous knowledge. In this work, we propose a criterion based on the \textit{learning consistency of a sample} called Consistency AWare Sampling (CAWS). This criterion prioritizes samples that are easier to learn by deep networks. We perform studies on three different memory-based methods: AGEM, GDumb, and Experience Replay, on MNIST, CIFAR-10 and CIFAR-100 datasets. We show that using the most consistent elements yields performance gains when constrained by a compute budget; when under no such constrain, random sampling is a strong baseline. However, using CAWS on Experience Replay yields improved performance over the random baseline. Finally, we show that CAWS achieves similar results to a popular memory selection method while requiring significantly less computational resources.

Paper:

## Consistency Score

Before running the experiments it is necessary to download the [C-Score](https://pluskid.github.io/structural-regularity/).

## Populating Memory by Consistency

> python main.py --use_gdumb_mod --gdumb_buffer_mode [random | uppder | lower]
> 
> python main.py --use_replay --use_custom_replay_buffer --replay_buffer_mode [random | uppder | lower]
> 
> python main.py --use_agem_mod --agem_buffer_mode [random | uppder | lower]

We can select the following hyper-parameters, plus some specific for each method, like the memory size.

- --dataset [cifar10 | cifar100 | mnist]
- --n_experience 5
- --epochs 15

## CAWS | COBS

> python main.py --use_replay --use_custom_replay_buffer --replay_buffer_mode caws --replay_min_bucket 0.9

Where `--replay_min_bucket` is the threshold use for CAWS

> python main.py --use_replay --use_custom_replay_buffer --replay_buffer_mode cobs

## CAWS + MIR

Without MIR:

> python main.py --use_mir --use_custom_replay_buffer --replay_buffer_mode [random | uppder | lower | caws | cobs]

With MIR:

> python main.py --use_mir --use_mir_replay --use_custom_replay_buffer --replay_buffer_mode [random | uppder | lower | caws | cobs] 

For more details please see `run.sh`
