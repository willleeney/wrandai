# WRandAI

[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/willleeney.svg?style=social&label=Follow%20%40willleeney)](https://twitter.com/willleeney)
![GitHub Repo stars](https://img.shields.io/github/stars/willleeney/wrandai?style=social)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/willleeney/wrandai/main-workflow.yaml)


This is a repository for calculating the $W$ Randomness Coefficient of a set of algorithms on a suite of Machine Learning benchmarks. 

## Installation

```
pip install wrandai
```

## Usage

The following code demonstrates how to quantify the uncertainty of a Machine Learning benchmarking investigation. Here `n_benchmark_tests` represents the different metrics/datasets combinations that an algorithm can be tested on. The `wrandai.w_randomness()` function calculates the uncertainty due to the presence of testing on random seeds. The format for the absolute scores required to calculate the $W$ Randomness is `[benchmark_tests, seeds, algorithms]`. The function assumes that a higher score is better when calculating ranking. Use `return_ranks=True` to return the average rank of each algorithm across all the benchmark tests and random seeds. 


```python
from wrandai import wrandai 

n_algorithms = 10
n_benchmark_tests = 20
n_random_seeds = 10

perfect_results = np.zeros((n_benchmark_tests, n_random_seeds, n_algorithms))
random_results = np.zeros((n_benchmark_tests, n_random_seeds, n_algorithms))

for test in range(n_benchmark_tests):
    for seed in range(n_random_seeds):
        interpolation_values = np.linspace(0, 1, n_algorithms).round(3)
        perfect_results[test, seed, :] = interpolation_values
        np.random.shuffle(interpolation_values)
        random_results[test, seed, :] = interpolation_values
    
print(f"{random_results[0, :, 0]} best algorithm dependant on seed used [one test, all seeds, one algorithm]")
print(f'{perfect_results[0, :, 0]} best algorithm is best across all seeds [one test, all seeds, one algorithm]')

print(f"W_w Randomness Coefficient for Random Rankings: {wrandai.w_randomness(random_results, w_method='w_wasserstein')}")
print(f"W_w Randomness Coefficient for Perfect Rankings: {wrandai.w_randomness(perfect_results, w_method='w_wasserstein')}")

print(f"W_t Randomness Coefficient for Random Rankings: {wrandai.w_randomness(random_results, w_method='w_ties')}")
print(f"W_t Randomness Coefficient for Perfect Rankings: {wrandai.w_randomness(perfect_results, w_method='w_ties')}")

print(f"W Randomness Coefficient for Random Rankings: {wrandai.w_randomness(random_results, w_method='w_random_coeff')}")
print(f"W Randomness Coefficient for Perfect Rankings: {wrandai.w_randomness(perfect_results, w_method='w_random_coeff')}")
```

Please cite [this paper](https://arxiv.org/abs/2305.06026) or [this paper](https://arxiv.org/abs/2312.09015) (or both) if you use this in your work:

```
@inproceedings{ugle2023leeney,
  title={Uncertainty in GNN Learning Evaluations: The Importance of a Consistent Benchmark for Community Detection.},
  author={Leeney, William and McConville, Ryan},
  booktitle={Twelfth International Conference on Complex Networks \& Their Applications},
  year={2023},
  organization={Springer}
}

@article{leeney2023uncertainty2,
  title={Uncertainty in GNN Learning Evaluations: A Comparison Between Measures for Quantifying Randomness in GNN Community Detection},
  author={Leeney, William and McConville, Ryan},
  journal={arXiv preprint arXiv:2312.09015},
  year={2023}
}
```