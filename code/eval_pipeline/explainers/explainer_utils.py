import numpy as np
from eval_pipeline.utils import POSITIVE, NEGATIVE, UNKNOWN, unpack_batches

def dataset_aspects_to_onehot(dataset):
    """
    Encode the aspects of a dataset in a onehot vector.
    """
    # encode the aspect labels
    dataset = dataset.replace('', -1)
    dataset = dataset.replace('no majority', -1)
    dataset = dataset.replace('Negative', 0)
    dataset = dataset.replace('unknown', 1)
    dataset = dataset.replace('Positive', 2)
    reprs = dataset[['food_aspect_majority', 'service_aspect_majority', 'ambiance_aspect_majority', 'noise_aspect_majority']].astype(int).to_numpy()

    # get a mask for the no majority data
    minus_ones = reprs == -1
    reprs_no_minuses = reprs
    reprs_no_minuses[minus_ones] = 0

    # create onehot encodings for the aspect labels
    N = reprs_no_minuses.shape[0]
    reprs_no_minuses = reprs_no_minuses.flatten()
    reprs_onehot = np.zeros((reprs_no_minuses.size, reprs_no_minuses.max() + 1))
    reprs_onehot[np.arange(reprs_no_minuses.size), reprs_no_minuses] = 1
    reprs_onehot = reprs_onehot.reshape(N, -1, 3)

    # use the mask to set no majority data to the zero vector
    reprs_onehot[minus_ones] = np.zeros(reprs_onehot[minus_ones].shape)

    reprs_onehot = reprs_onehot.reshape(reprs_onehot.shape[0], -1)
    return reprs_onehot

# INLP and LEACE util functions
def treatment_to_label(x):
    if x == NEGATIVE:
        return 0
    elif x == UNKNOWN:
        return 1
    elif x == POSITIVE:
        return 2
    else:
        return None


def label_to_treatment(x):
    if x == 0:
        return NEGATIVE
    elif x == 1:
        return UNKNOWN
    elif x == 2:
        return POSITIVE
    else:
        return None

def sort_dev_embeddings(dev_preprocessed, treatment):
    """ Takes a (embeddings, concept_label, overall_label) tuple for a particular treatment (concept)
    and breaks these embeddings into a dictionary of {NEGATIVE, UNKNOWN, POSITIVE} samples for
    sampling during computation of counterfactual.
    """
    treatment_samples = {}
    embeddings = np.array(unpack_batches(dev_preprocessed[0]))
    labels = np.array(dev_preprocessed[1])
    unique_labels = np.array([0,1,2])
    assert (np.unique(labels) == unique_labels).all(), f"Incorrect sentiment labels for treatment {treatment}"
    for label in unique_labels:
        treatment_samples[label_to_treatment(label)] = embeddings[labels==label]
    return treatment_samples
