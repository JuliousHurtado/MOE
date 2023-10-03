# Memory Population in Continual Learning via Outlier Elimination

Catastrophic forgetting, the phenomenon of forgetting previously learned tasks when learning a new one, is a major hurdle in developing continual learning algorithms. A popular method to alleviate forgetting is to use a memory buffer, which stores a subset of previously learned task examples for use during training on new tasks. The de facto method of filling memory is by randomly selecting previous examples. However, this process could introduce outliers or noisy samples that could hurt the generalization of the model. This paper introduces Memory Outlier Elimination (MOE), a method for identifying and eliminating outliers in the memory buffer by choosing samples from label-homogeneous subpopulations. We show that a space with a high homogeneity is related to a feature space that is more representative of the class distribution. In practice, MOE removes a sample if it is surrounded by samples from different labels. We demonstrate the effectiveness of MOE on CIFAR-10, CIFAR-100, and CORe50, outperforming previous well-known memory population methods.

Paper:

## Consistency Score

Before running the experiments it is necessary to download the [C-Score](https://pluskid.github.io/structural-regularity/).
