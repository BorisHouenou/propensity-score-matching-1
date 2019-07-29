# code adapted from:
# https://nbviewer.jupyter.org/github/kellieotto/StatMoments/blob/master/PSM.ipynb

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def propensity_score_match(treatment, control, mode, caliper=0.05):
    
propensity = LogisticRegression()
propensity = propensity.fit(data, data.Treated)
pscore = propensity.predict_proba(data)[:, 1]  # The predicted propensities by the model
print(pscore[:5])

data['Propensity'] = pscore

# pscore = pd.Series(data = pscore, index = data.index)

def Match(data, groups, propensity, mode='standard', caliper=0.05):
    '''
    Inputs:
    groups = Treatment assignments.  Must be 2 groups
    propensity = Propensity scores for each observation. Propensity and groups should be in the same order (matching indices)
    mode = 'standard' matching to the control sample to which there is the smallest absolute difference in propensity
           'mahalanobis' matches to sample with closest mahalanobis distance, from subset of control samples that have
           an absolute difference specified by 'caliper'
    caliper = ('standard' mode) Maximum difference in matched propensity scores. For now, this is a caliper on the raw
                                propensity; Austin reccommends using a caliper on the logit propensity.
              ('mahalanobis' mode) Maximum difference in propensity score, used to create a subset of samples for which
                                   to compare the Mahalanobis distance.

    Output:
    A series containing the individuals in the control group matched to the treatment group.
    Note that with caliper matching, not every treated individual may have a match.
    '''

    # Check inputs
    if any(propensity <= 0) or any(propensity >= 1):
        raise ValueError('Propensity scores must be between 0 and 1')
    elif not (0 < caliper < 1):
        raise ValueError('Caliper must be between 0 and 1')
    elif len(groups) != len(propensity):
        raise ValueError('groups and propensity scores must be same dimension')
    elif len(groups.unique()) != 2:
        raise ValueError('wrong number of groups')

    # Code groups as 0 and 1
    groups = groups == groups.unique()[0]
    N = len(groups)
    N1 = groups.sum()
    N2 = N - N1
    g1, g2 = propensity[groups == 1], (propensity[groups == 0])
    # Check if treatment groups got flipped - treatment (coded 1) should be the smaller
    if N1 > N2:
        N1, N2, g1, g2 = N2, N1, g2, g1

        # Randomly permute the smaller group to get order for matching
    morder = np.random.permutation(N1)
    matches = pd.Series(np.empty(N1), dtype='int16')
    matches[:] = np.nan


    if mode == 'standard':
        for m in morder:
            dist = abs(g1[m] - g2)
            if dist.min() <= caliper:
                matches[m] = dist.idxmin()
                g2 = g2.drop(matches[m], axis=0)
    elif mode == 'mahalanobis':
        # not coded yet
        pass
    return (matches)


matchings = Match(data.drop(['Treated', 'Propensity'], axis=1), data.Treated, data.Propensity)
g1, g2 = data.Propensity[data.Treated], data.Propensity[~data.Treated]
# test ValueError
# badtreat = data.Treated + data.Hispanic
# Match(badtreat, pscore)
print(matchings[:5])

matchings = [el for el in zip(g1, g2[matchings])]
print(matchings[:5])


pass
