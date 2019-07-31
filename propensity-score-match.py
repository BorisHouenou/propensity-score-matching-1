# references:
# https://nbviewer.jupyter.org/github/kellieotto/StatMoments/blob/master/PSM.ipynb
# https://www.umanitoba.ca/faculties/health_sciences/medicine/units/chs/departmental_units/mchp/protocol/media/propensity_score_matching.pdf
# https://stats.stackexchange.com/questions/206832/matched-pairs-in-python-propensity-score-matching

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def match_samples(treatment, control, mode='propensity_abs', caliper=0.05):
    '''
    :param X: DataFrame containing control and treatment samples. shape: (num_samples, num_features)
    :param y: Series containing corresponding treatment flag for each sample, 1: treatment sample, 0: control sample.
              shape: (num_samples,)
    :param mode: 'propensity_abs': matching to the control sample to which there is the smallest absolute difference
                                   in propensity, where propensity is estimated with a logistic regression
                 'propensity_mbis': (not implemented yet) matches to sample with closest mahalanobis distance, from
                                    subset of control samples that have an absolute difference specified by 'caliper'
                 'nn': treatment matched to the nearest neighbour in the feature space, using sklearn's NearestNeighbors
    :param caliper: 'propensity_abs' mode: maximum difference in matched propensity scores for a valid match to be made
                    'mahalanobis' mode: maximum difference in propensity score, used to create a subset of samples
                                        for which to compare the Mahalanobis distance.
    :return: numpy array, shape: (num_samples, 2). Column 1 gives row number of treatment sample, column 2 gives row
             number of matched control sample. Column 2 value is nan if no match was found within the maximum distance
             specified by caliper, if applicable.
    '''

    # check inputs
    if not (0 < caliper < 1):
        raise ValueError('Caliper must be between 0 and 1')
    num_samples_treatment, num_features_treatment = treatment.shape
    num_samples_control, num_features_control = control.shape
    if not (num_samples_treatment <= num_samples_control):
        raise ValueError('Treatment sample pool must not be bigger than control sample pool.')
    if not (num_features_treatment == num_features_control):
        raise ValueError('Treatment and control must have the same number of features.')
    mode_options = ['propensity_abs', 'propensity_mbis', 'nn']
    if mode not in mode_options:
        raise ValueError('Do not recognise value ' + str(mode) + ' for mode. ' +
                         'Accepted values: ' + str(mode_options) + '.')

    # elif not X.index.is_unique():
    #     raise ValueError('Index of X argument must have unique values')
    # elif not X.index.equals(y.index):
    #     raise ValueError('Index of X and y arguments must be the same')

    # convert inputs to numpy array if necessary
    if isinstance(treatment, pd.DataFrame):
        treatment = treatment.values
    if isinstance(control, pd.DataFrame):
        control = control.values


    # initialise array for matches
    matches = np.vstack([np.arange(num_samples_treatment), np.full(num_samples_treatment, np.nan)]).T

    if mode in ['propensity_abs', 'propensity_mbis']:
        # merge control and treatment for logistic regression
        y = np.append(np.ones(num_samples_treatment), np.zeros(num_samples_control))
        X = np.append(treatment, control, axis=0)

        # generate propensity scores using logistic regression
        propensity_model = LogisticRegression()
        propensity_model = propensity_model.fit(X, y)
        p_score = propensity_model.predict_proba(X)[:, 1]  # estimate probability that sample is treatment from log-reg

        # split p_score into control and treatment
        mask_treatment = (y == 1)  # mask to be used to retrieve only treatment samples
        p_treatment = p_score[mask_treatment]  # array containing pscore for only treatment samples
        p_control = p_score[~mask_treatment]  # array containing pscore for only control samples
        idx_control = np.arange(num_samples_control)  # to keep track of original index when deleting chosen control samples

        if mode =='propensity_abs':
            for i in range(num_samples_treatment):  # for each treatment sample
                dist = np.abs(p_control - p_treatment[i])  # find distance to p_score of all control samples
                j = np.argmin(dist)  # save index of smallest distance
                if dist[j] <= caliper:
                    matches[i, 1] = idx_control[j]  # save index of matched control sample
                    p_control = np.delete(p_control, j, 0)  # remove sample for next iteration
                    idx_control = np.delete(idx_control, j, 0)  # remove sample for next iteration
        elif mode == 'propensity_mbis':
            # not coded yet
            raise RuntimeError('Error: mode \'mahalanobis\' is not written yet, please use mode \'standard\'.')
    elif mode == 'nn':
        neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control)  # fit model to control samples
        distances, indices = neighbours.kneighbors(treatment)  # find nearest control sample to each treatment sample
        indices = indices.reshape(indices.shape[0])  # reshape to 1D array
        matches[:, 1] = indices  # save index of each control sample into matches output

    return matches.astype(int)


# demonstrator code

treated = pd.DataFrame()
np.random.seed(42)

num_samples_treatment = 200
num_samples_control = 1000
treated['x'] = np.random.normal(0, 1, size=num_samples_treatment)
treated['y'] = np.random.normal(50, 20, size=num_samples_treatment)
treated['z'] = np.random.normal(0, 100, size=num_samples_treatment)

control = pd.DataFrame()
# two different populations
control['x'] = np.append(np.random.normal(0, 3, size=num_samples_control),
                         np.random.normal(-1, 2, size=2 * num_samples_control))
control['y'] = np.append(np.random.normal(50, 30, size=num_samples_control),
                         np.random.normal(-100, 2, size=2 * num_samples_control))
control['z'] = np.append(np.random.normal(0, 200, size=num_samples_control),
                         np.random.normal(13, 200, size=2 * num_samples_control))

# match
matchings = match_samples(treated, control, mode='nn')

control_matched = control.iloc[matchings[:, 1], :]  # extract matched control samples

# fig, ax = plt.subplots(figsize=(6,6))
plt.scatter(control['x'], control['y'], alpha=0.3, label='Control pool')
plt.scatter(treated['x'], treated['y'], label='Treated')
plt.scatter(control_matched['x'], control_matched['y'], marker='x', label='Matched samples')
plt.legend()
plt.xlim(-1, 2)

pass
