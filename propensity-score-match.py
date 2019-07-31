# code adapted from:
# https://nbviewer.jupyter.org/github/kellieotto/StatMoments/blob/master/PSM.ipynb

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def propensity_score_match(X, y, mode='standard', caliper=0.05):
    '''
    :param X: DataFrame/Numpy array containing control and treatment samples. shape: (num_samples, num_features)
    :param y: Series/Numpy array containing treatment flag for sample, 1: treatment sample, 0: control sample. shape: (num_samples,)
    :param mode: 'standard' matching to the control sample to which there is the smallest absolute difference in propensity
                 'mahalanobis' (not implemented yet) matches to sample with closest mahalanobis distance,
                 from subset of control samples that have an absolute difference specified by 'caliper'
    :param caliper: ('standard' mode) maximum difference in matched propensity scores for a valid match to be made
                    ('mahalanobis' mode) maximum difference in propensity score, used to create a subset of samples
                    for which to compare the Mahalanobis distance.
    :return: Series, shape: (num_samples,). index is same as treatment, aligning with each treatment sample. value gives
             row number of the matched sample in control.  Row number is NaN if no match was found within the maximum specified by caliper.
    '''

    # check inputs
    if not (0 < caliper < 1):
        raise ValueError('Caliper must be between 0 and 1')

    # convert inputs to numpy array if necessary
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # generate propensity scores using logistic regression
    propensity_model = LogisticRegression()
    propensity_model = propensity_model.fit(X, y)
    p_score = propensity_model.predict_proba(X)[:, 1]  # estimate probability that sample is treatment from log-reg

    mask_treatment = (y == 1)


    #
    #
    # if mode == 'standard':
    #     for m in morder:
    #         dist = abs(g1[m] - g2)
    #         if dist.min() <= caliper:
    #             matches[m] = dist.idxmin()
    #             g2 = g2.drop(matches[m], axis=0)
    # elif mode == 'mahalanobis':
    #     # not coded yet
    #     pass
    # return (matches)
    #
    #
    # matchings = Match(data.drop(['Treated', 'Propensity'], axis=1), data.Treated, data.Propensity)
    # g1, g2 = data.Propensity[data.Treated], data.Propensity[~data.Treated]
    # # test ValueError
    # # badtreat = data.Treated + data.Hispanic
    # # Match(badtreat, pscore)
    # print(matchings[:5])
    #
    # matchings = [el for el in zip(g1, g2[matchings])]
    # print(matchings[:5])

    pass


# demonstrator code

treated = pd.DataFrame()
np.random.seed(42)

num_samples_treatment = 200
num_samples_control = 1000
treated['x'] = np.random.normal(0, 1, size=num_samples_treatment)
treated['y'] = np.random.normal(50, 20, size=num_samples_treatment)
treated['z'] = np.random.normal(0, 100, size=num_samples_treatment)
treated['treated'] = 1

control = pd.DataFrame()
# two different populations
control['x'] = np.append(np.random.normal(0, 3, size=num_samples_control),
                         np.random.normal(-1, 2, size=2 * num_samples_control))
control['y'] = np.append(np.random.normal(50, 30, size=num_samples_control),
                         np.random.normal(-100, 2, size=2 * num_samples_control))
control['z'] = np.append(np.random.normal(0, 200, size=num_samples_control),
                         np.random.normal(13, 200, size=2 * num_samples_control))
control['treated'] = 0

y = treated['treated'].append(control['treated'])
X = treated.drop('treated', axis=1).append(control.drop('treated', axis=1))
matchings = propensity_score_match(X, y)

# fig, ax = plt.subplots(figsize=(6,6))
plt.scatter(control['x'], control['y'], alpha=0.3, label='Control pool')
plt.scatter(treated['x'], treated['y'], label='Treated')
# plt.scatter(matched_df['x'], matched_df['y'], marker='x', label='Matched samples')
plt.legend()
plt.xlim(-1, 2)

pass
