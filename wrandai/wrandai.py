import numpy as np
from scipy.stats import wasserstein_distance

def w_randomness(result_object: np.ndarray, return_ranks: bool = False, w_method: str = 'w_wasserstein'):
    '''
        Calculates the `W` randomness of a set of benchmarking test, between algorithms over the random seeds
        that they were tested on. The benchmarking tests refer to the datasets and metrics that a machine       
        learning algorithm can be evaluated against in order to test performance. 

        `W = \\frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} 1 - \\frac{\sum_{i=1}^{a} \sum_{j=1}^{i-1} W1(i, j)}{\sum_{v=1}^{a} \\frac{v(v-1)}{2}}`
        
        Args:
            result_object (np.ndarray) [n_benchmarking_tests, n_seeds, n_algorithms]: The result object that 
                contains the scores of every algoritm on each of the seeds used for all benchmarking tests. 
            return_ranks (bool) default=False: Whether to return the algorithm rankings.
            w_method (str) {'w_wasserstein', 'w_ties' or 'w_random_coeff'}, default='w_wasserstein': Which method to use to calculate 
                the W Randomness Coefficient
        Returns:
            w_rand (int): The value of randomness present in this investigation where 1 is maximum randomness 
                and 0 is the minimum.
            (optional) algo_ranks (np.ndarray) [n_algorithms]: If return_ranks=True then this is the average 
                rank of each algorithm across all the benchmarking tests. 
    '''
    # get the chosen method for calculating randomness
    valid_w_methods = ['w_wasserstein', 'w_ties', 'w_random_coeff']
    assert w_method in valid_w_methods, NameError(f"w_method must be one of {valid_w_methods}")
    wrand_fn = globals().get(w_method)
    assert result_object.ndim == 3, ValueError('result matrix must be 3-dimensional')

    # iterate through each of the benchmarking tests
    w_rand = []
    all_ranks = np.zeros_like(result_object)
    for t, test in enumerate(result_object):
        # calculate the rank of each algorithm for all seeds on a single test
        rank_test = np.zeros_like(test)
        for rs, rs_test in enumerate(test):
            rank_test[rs] = rank_scores(rs_test)
        # calculate w_randomness coefficient
        w_rand.append(wrand_fn(rank_test))
        all_ranks[t] = rank_test
    w_rand = np.mean(np.array(w_rand))
    if return_ranks:
        algo_ranks = np.transpose(all_ranks, (2, 0, 1))
        algo_ranks = algo_ranks.reshape(algo_ranks.shape[0], algo_ranks.shape[1] * algo_ranks.shape[2])
        algo_ranks = np.mean(algo_ranks, axis=1)
        return w_rand, algo_ranks
    else:
        return w_rand


def nrule(n: int):
    '''
        Calculates the distance expected under no randomness as given by the nrule() sequence for n algorithms.
        math:`\sum_{i=1}^{n} \\frac{i(i-1)}{2}`

        Args:
            n (int): The number of different algorithms.
        Returns:
            distance (int): The distance expected where there is no variation of algorithm rankings 
                between seeds.
    '''
    distance = 0 
    for i in range(1,n+1):
        distance += (i*(i-1))/2
    return distance


def rank_scores(scores: np.ndarray):
    '''
        Calculates the ranking of algorithms from the absolute scores, returning the average of the ranks 
        where a tie has occured. This evaluates the best algorithm as the one with the highest score. 

        Args:
            scores (np.ndarray) [n_algorithms]: the scores object that contains the absolute
                performance of every algoritm on a single seed used for a single benchmarking test. 
        Returns:
            ranks (np.ndarray) [n_algorithms]: the rank of each algorithm
    '''
    assert scores.ndim == 1, ValueError('scores matrix must be 1-dimensional')
    # get indices in descending order
    indices = np.flip(np.argsort(scores))
    # initialise an array to store the ranks
    ranks = np.zeros_like(indices, dtype=float)
    # assign ranks to the sorted indices
    ranks[indices] = np.arange(len(scores)) + 1
    # find unique scores and their counts
    unique_scores, counts = np.unique(scores, return_counts=True)
    # calculate mean ranks for tied scores
    for score, count in zip(unique_scores, counts):
        if count > 1:
            score_indices = np.where(scores == score)[0]
            mean_rank = np.mean(ranks[score_indices])
            ranks[score_indices] = mean_rank
    return ranks


def w_wasserstein(rankings: np.ndarray):
    '''
        Calculates the `W` randomness of a single benchmarking test using the `W_w` Wasserstein Randomness.

        `W_w = 1 - \\frac{\sum_{i=1}^{a} \sum_{j=1}^{i-1} W1(i, j)}{\sum_{v=1}^{a} \\frac{v(v-1)}{2}}`
        
        Args:
            rankings (np.ndarray) [n_seeds, n_algorithms]: The ranking object that contains the ranks
                of every algoritm on each of the seeds used for a single benchmarking test. 
        Returns:
            W (int): The value of randomness present in this test where 1 is maximum randomness and 
                0 is the minimum.
    '''
    assert rankings.ndim == 2, ValueError('rankings matrix must be 2-dimensional')
    n_algorithms = rankings.shape[1]
    wass_agg = []
    for i in range(n_algorithms):
        for j in range(i):  # Iterate up to i to stay to the left of the diagonal
            wass_agg.append(wasserstein_distance(rankings[:, i], rankings[:, j]))
    # normalise the total distance between all algorithms by the distance expected 
    # under no randomness as given by the nrule() sequence for n algorithms 
    W = 1 - (np.sum(wass_agg) / nrule(n_algorithms))
    return W


def w_ties(rankings: np.ndarray):
    '''
        Calculates the `W` randomness of a single benchmarking test using the `W_t` Randomness Coefficient 
        based on Kendall's Coefficient of concordance that accounts for ties.

        `W_t = 1 -  \\frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}}  \\frac{12\sum_{i=1}^{a} 
        (R_i^2) - 3n^2a(a+1)^2}{n^2 a(a^2 -1) - n \sum_{j=1}^{n} (\sum_{i=1}^{g_j} (t^{3}_{i} - t_i))}`
        
        Args:
            rankings (np.ndarray) [n_seeds, n_algorithms]: The ranking object that contains the ranks
                of every algoritm on each of the seeds used for a single benchmarking test. 
        Returns:
            W (int): The value of randomness present in this test where 1 is maximum randomness and 
                0 is the minimum.
    '''
    assert rankings.ndim == 2, ValueError('rankings matrix must be 2-dimensional')
    n = rankings.shape[0]
    a = rankings.shape[1]

    Ti = np.zeros(n)
    for j in range(n):
        _, counts = np.unique(rankings[j, :], return_counts=True)
        tied_groups = counts[counts > 1]
        Ti[j] = np.sum(tied_groups ** 3 - tied_groups)

    T = np.sum(Ti) * n
    R = np.sum(rankings, axis=0)
    R = sum(r ** 2 for r in R)

    if (((n ** 2) * a * ((a ** 2)  - 1)) -  T) == 0.:
        W = 1 - ((12*R) - 3 * (n**2) * a *((a + 1) ** 2))
    else:
        W = 1 - ((12*R) - 3 * (n**2) * a *((a + 1) ** 2)) / (((n ** 2) * a * ((a ** 2)  - 1)) -  T)
    return float(W)


def w_random_coeff(rankings: np.ndarray):
    '''
        Calculates the `W` randomness of a single benchmarking test using the `W` Randomness Coefficient 
        based on Kendall's Coefficient of concordance.

        `W = 1 - \\frac{\sum_{i=1}^{a} \sum_{j=1}^{i-1} W1(i, j)}{\sum_{v=1}^{a} \\frac{v(v-1)}{2}}`
        
        Args:
            rankings (np.ndarray) [n_seeds, n_algorithms]: The ranking object that contains the ranks
                of every algoritm on each of the seeds used for a single benchmarking test. 
        Returns:
            W (int): The value of randomness present in this test where 1 is maximum randomness and 
                0 is the minimum.
    '''
    assert rankings.ndim == 2, ValueError('rankings matrix must be 2-dimensional')
    n = rankings.shape[0]
    a = rankings.shape[1]
    denom = n**2*(a**3-a)
    rating_sums = np.sum(rankings, axis=0)
    S = a*np.var(rating_sums)
    W = 1 - (12*S/denom)
    return W