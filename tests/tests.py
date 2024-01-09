from wrandai import wrandai
import numpy as np

random_rankings = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
perfect_rankings = random_rankings.T

print(f'{random_rankings[:, 0]} best algorithm completely dependant on seed used [all seeds, one algorithm]')
print(f'{perfect_rankings[:, 0]} best algorithm is best across all seeds [all seeds, one algorithm]')

print(f'W_w Randomness Coefficient for Random Rankings: {wrandai.w_wasserstein(random_rankings)}')
print(f'W_w Randomness Coefficient for Perfect Rankings: {wrandai.w_wasserstein(perfect_rankings)}')

print(f'W_t Randomness Coefficient for Random Rankings: {wrandai.w_ties(random_rankings)}')
print(f'W_t Randomness Coefficient for Perfect Rankings: {wrandai.w_ties(perfect_rankings)}')

print(f'W Randomness Coefficient for Random Rankings: {wrandai.w_random_coeff(random_rankings)}')
print(f'W Randomness Coefficient for Perfect Rankings: {wrandai.w_random_coeff(perfect_rankings)}')


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