# WRandAI

This is a repository for calculating the $W$ Randomness Coefficient of a set of algorithms on a suite of Machine Learning benchmarks. 

## Installation

```
pip install wrandai
```

## Usage


```python
from wrandai.wrandai import ww_randomness

n_algorithms = 10
n_benchmark_tests = 20
n_random_seeds = 15

random_rankings = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
print(f'{random_rankings[:, 0]} -> all seeds, one algorithm')
print(f'W Randomness Coefficient for Random Rankings: {ww_randomness(random_rankings)}')
perfect_rankings = random_rankings.T
print(f'W Randomness Coefficient for Perfect Rankings: {ww_randomness(random_rankings)}')


```

Please cite [our paper](https://arxiv.org/abs/2312.09015) if you use this in your work:

```
@article{leeney2023uncertainty,
  title={Uncertainty in GNN Learning Evaluations: A Comparison Between Measures for Quantifying Randomness in GNN Community Detection},
  author={Leeney, William and McConville, Ryan},
  journal={arXiv preprint arXiv:2312.09015},
  year={2023}
}
```