# Pseudo-Asynchronous Local SGD: Robust and Efficient Data-Parallel Training

## Abstract

Following AI scaling trends, frontier models continue to grow in size and continue to be trained on larger datasets.  Training these models requires huge investments in exascale computational resources, which has in turn driven developtment of distributed deep learning methods. Data parallelism is an essential approach to speed up training, but it requires frequent global communication between workers, which can bottleneck training at the largest scales. In this work, we propose a method called Pseudo-Asynchronous Local SGD (PALSGD) to improve the efficiency of data-parallel training. PALSGD is an  extension of Local SGD (Stich, 2018) and DiLoCo (Douillard et al., 2023), designed to further reduce communication frequency by introducing a pseudo-synchronization mechanism. PALSGD allows the use of longer synchronization intervals compared to standard Local SGD. Despite the reduced communication frequency, the pseudo-synchronization approach ensures that model consistency is maintained, leading to performance results comparable to those achieved with more frequent synchronization. Furthermore, we provide a theoretical analysis of PALSGD, establishing its convergence and deriving its convergence rate. This analysis offers insights into the algorithm's behavior and performance guarantees. We evaluated PALSGD on image classification and language modeling tasks. Our results show that PALSGD achieves better performance in less time compared to existing methods like Distributed Data Parallel (DDP), and DiLoCo. Notably, PALSGD trains 18.4% faster than DDP on ImageNet-1K with ResNet-50, 24.4% faster than DDP on TinyStories with GPT-Neo-125M, and 21.1% faster than DDP on TinyStories with GPT-Neo-8M.

## Implementation

This repository contains the implementation of PALSGD and related distributed training algorithms. The codebase is organized into four main experimental directories, each focusing on different model architectures and datasets:

### Directory Structure

```
├── README.md
├── gptneo_125m/          # GPT-Neo 125M experiments on TinyStories
├── gptneo_8m/            # GPT-Neo 8M experiments on TinyStories  
├── resnet_cifar10/       # ResNet experiments on CIFAR-10
└── resnet_imagenet/      # ResNet experiments on ImageNet-1K
```

### Implementation Contents

Each directory contains:
- **Main training scripts**: Single-node and multi-node distributed training implementations
- **Algorithm implementations**: DDP, Local SGD, Post-Local SGD, and PALSGD variants
- **Configuration management**: Flexible hyperparameter configuration for different experimental setups
- **Data processing modules**: Dataset loading and preprocessing for each task
- **Utility functions**: GPU monitoring, model parameter analysis, and distributed training utilities

## Citation

```bibtex
@article{naganuma2025pseudo,
  title={Pseudo-Asynchronous Local SGD: Robust and Efficient Data-Parallel Training},
  author={Naganuma, Hiroki and Zhang, Xinzhi and Yue, Man-Chung and Mitliagkas, Ioannis and Witte, Philipp A and Hewett, Russell J and Lee, Yin Tat},
  journal={Transactions on Machine Learning Research}
  year={2025}
}
```

- [arXiv](https://arxiv.org/abs/2504.18454)
- [OpenReview](https://openreview.net/forum?id=8VTrvS5vN7)

## Paper authors

- [Hiroki Naganuma](https://hiroki11x.github.io/) †
- [Xinzhi Zhang](https://openreview.net/profile?id=~Xinzhi_Zhang2) †
- [Man-Chung Yue](https://manchungyue.com/)
- [Ioannis Mitliagkas](https://mitliagkas.github.io/)
- [Russell J. Hewett](https://www.rjh.io/) *
- [Philipp Andre Witte](https://www.microsoft.com/en-us/research/people/pwitte/) *
- [Yin Tat Lee](https://yintat.com/) *

> † denotes equal contribution
> \* denotes equal contribution
