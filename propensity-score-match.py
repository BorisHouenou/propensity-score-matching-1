# references:
# https://nbviewer.jupyter.org/github/kellieotto/StatMoments/blob/master/PSM.ipynb
# https://www.umanitoba.ca/faculties/health_sciences/medicine/units/chs/departmental_units/mchp/protocol/media/propensity_score_matching.pdf
# https://stats.stackexchange.com/questions/206832/matched-pairs-in-python-propensity-score-matching

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def propensity_score_match(X, y, mode='standard', caliper=0.05):
    '''
    :param X: DataFrame containing control and treatment samples. shape: (num_samples, num_features)
    :param y: Series containing corresponding treatment flag for each sample, 1: treatment sample, 0: control sample.
              shape: (num_samples,)
    :param mode: 'standard': matching to the control sample to which there is the smallest absolute difference in
                             propensity
                 'mahalanobis': (not implemented yet) matches to sample with closest mahalanobis distance,
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
    elif not (y[y == 1].size < y[y == 0].size):
        raise ValueError('Treatment sample pool must be smaller than control sample pool.')
    # elif not X.index.is_unique():
    #     raise ValueError('Index of X argument must have unique values')
    # elif not X.index.equals(y.index):
    #     raise ValueError('Index of X and y arguments must be the same')

    # convert inputs to numpy array if necessary
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # generate propensity scores using logistic regression
    propensity_model = LogisticRegression()
    propensity_model = propensity_model.fit(X, y)
    p_score = propensity_model.predict_proba(X)[:, 1]  # estimate probability that sample is treatment from log-reg

    # split p_score into control and treatment
    mask_treatment = (y == 1)  # mask to be used to retrieve only treatment samples
    p_treatment = p_score[mask_treatment]  # array containing pscore for only treatment samples
    p_control = p_score[~mask_treatment]  # array containing pscore for only control samples

    # initialise array for matches
    num_samples = y.size  # number of samples in X and y
    matches = np.vstack([np.arange(num_samples), np.full(num_samples, np.nan)]).T

    # save original index of each sample (in X) for use in the output
    idx_in_X_treatment = matches[mask_treatment, 0].astype(int)
    idx_in_X_control = matches[~mask_treatment, 0].astype(int)

    if mode == 'standard':
        for i in range(p_treatment.size):  # for each treatment sample
            dist = np.abs(p_control - p_treatment[i])  # find distance to p_score of all control samples
            j = np.argmin(dist)  # save index of smallest distance
            if dist[j] <= caliper:
                matches[idx_in_X_treatment[i], 1] = idx_in_X_control[j]  # save index in X of matched control sample
                p_control = np.delete(p_control, j, 0)  # remove sample for next iteration
                idx_in_X_control = np.delete(idx_in_X_control, j, 0)  # remove sample for next iteration
        matches = matches[~np.isnan(matches[:, 1])].astype(int)  # keep only rows for matched treatment samples
    elif mode == 'mahalanobis':
        # not coded yet
        raise RuntimeError('Error: mode \'mahalanobis\' is not written yet, please use mode \'standard\'.')

    return matches


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

# merge data and match by propensity score
y = treated['treated'].append(control['treated'])
X = treated.drop('treated', axis=1).append(control.drop('treated', axis=1))
matchings = propensity_score_match(X, y)

control_matched = control.iloc[matchings[:, 1], :]  # extract matched control samples

# fig, ax = plt.subplots(figsize=(6,6))
plt.scatter(control['x'], control['y'], alpha=0.3, label='Control pool')
plt.scatter(treated['x'], treated['y'], label='Treated')
plt.scatter(control_matched['x'], control_matched['y'], marker='x', label='Matched samples')
plt.legend()
plt.xlim(-1, 2)

pass
