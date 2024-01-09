import numpy as np
from scipy.stats import wasserstein_distance

def ww_randomness(result_object: np.ndarray, 
                  return_ranks: bool = False):
    '''
        Calculates the `W` randomness of a set of benchmarking test, between algorithms over the random seeds
        that they were tested on. The benchmarking tests refer to the datasets and metrics that a machine       
        learning algorithm can be evaluated against in order to test performance. 

        `W = \\frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} 1 - \\frac{\sum_{i=1}^{a} \sum_{j=1}^{i-1} W1(i, j)}{\sum_{v=1}^{a} \\frac{v(v-1)}{2}}`
        
        Args:
            result_object (np.ndarray) [n_benchmarking_tess, n_seeds, n_algorithms]: The result object that 
                contains the scores of every algoritm on each of the seeds used for all benchmarking tests. 
            return_ranks (bool): Whether to return the algorithm rankings.
        Returns:
            w_rand (int): The value of randomness present in this investigation where 1 is maximum randomness 
                and 0 is the minimum.
            (optional) algo_ranks (np.ndarray) [n_algorithms]: If return_ranks=True then this is the average 
                rank of each algorithm across all the benchmarking tests. 
    '''
    # iterate through each of the benchmarking tests
    w_rand = []
    all_ranks = np.zeros_like(result_object)
    for t, test in enumerate(result_object):
        # calculate the rank of each algorithm for all seeds on a single test
        rank_test = np.zeros_like(test)
        for rs, rs_test in enumerate(test):
            rank_test[rs] = rank_scores_w_ties(rs_test)
        # calculate w_randomness coefficient
        w_rand.append(w_rand_wasserstein(rank_test))
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

def w_rand_wasserstein(rankings: np.ndarray):
    '''
        Calculates the `W` randomness of a single benchmarking test.

        `1 - \\frac{\sum_{i=1}^{a} \sum_{j=1}^{i-1} W1(i, j)}{\sum_{v=1}^{a} \\frac{v(v-1)}{2}}`
        
        Args:
            rankings (np.ndarray) [n_seeds, n_algorithms]: The ranking object that contains the ranks
                of every algoritm on each of the seeds used for a single benchmarking test. 
        Returns:
            w (int): The value of randomness present in this test where 1 is maximum randomness and 
                0 is the minimum.
    '''
    n_algorithms = rankings.shape[1]
    wass_agg = []
    for i in range(n_algorithms):
        for j in range(i):  # Iterate up to i to stay to the left of the diagonal
            wass_agg.append(wasserstein_distance(rankings[:, i], rankings[:, j]))
    # normalise the total distance between all algorithms by the distance expected 
    # under no randomness as given by the nrule() sequence for n algorithms 
    w = 1 - (np.sum(wass_agg) / nrule(n_algorithms))
    return w

def rank_scores_w_ties(scores: np.ndarray):
    '''
        Calculates the ranking of algorithms from the absolute scores, returning the average of the ranks 
        where a tie has occured. 

        Args:
            scores (np.ndarray) [n_seeds, n_algorithms]: the scores object that contains the absolute
                performance of every algoritm on each of the seeds used for a single benchmarking test. 
        Returns:
            ranks (np.ndarray) [n_seeds, n_algorithms]: the rank of each algorithm on each of the seeds
    '''
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

