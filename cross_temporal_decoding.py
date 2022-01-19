import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.multiclass import OneVsOneClassifier


def cross_temporal_decoding(X_train, X_test, y_train, y_test, alpha=1):
    '''Cross-temporal decoding with cross-validation in subspace

    Parameters
    ----------
    X_train : np.array<trials * bins * neurons> - training data
    X_test : np.array<trials * bins * neurons> - testing data
    y_train : np.array<trials> - training labels
    y_test : np.array<trials> - testing labels
    alpha : float - the L2 "ridge" regularization parameter

    Returns
    -------
    accuracy : np.array<bins * bins> of floats
        The fraction of correctly classified trials for each pair of train and
        test bins
    predictions : np.array<bins * bins * test trials>
        The output of the classifier for each pair of train and test bins
    '''
    # Initialization of variables
    ntests, nbins, _ = X_test.shape

    predictions = np.empty((ntests, nbins, nbins))
    accuracies = np.empty((nbins, nbins))

    # Main loop going through every combination of training and testing time point
    for itrain in range(nbins):
        for itest in range(nbins):
            # Retrieving the data of the training and testing bins
            X_bin_train = X_train[:, itrain]
            X_bin_test = X_test[:, itest]

            # Fitting the ridge classifier using scikit-learn to get a multi-class classification
            model = OneVsOneClassifier(RidgeClassifier(alpha=alpha, solver='cholesky'))
            model.fit(X_bin_train, y_train)

            # Computing and saving predictions and accuracy on left out data
            bin_preds = model.predict(X_bin_test)
            accuracy = (bin_preds == y_test).mean(0)
            predictions[:, itrain, itest] = bin_preds
            accuracies[itrain, itest] = accuracy

    return accuracies, predictions



def CTD_non_repeated_training(X_train, X_test, y_train, y_test, alpha=1):
    '''Cross-temporal decoding with cross-validation in subspace

    Parameters
    ----------
    X_train : np.array<trials * bins * neurons> - training data
    X_test : np.array<trials * bins * neurons> - testing data
    y_train : np.array<trials> - training labels
    y_test : np.array<trials> - testing labels
    alpha : float - the L2 "ridge" regularization parameter

    Returns
    -------
    accuracy : np.array<bins * bins> of floats
        The fraction of correctly classified trials for each pair of train and
        test bins
    predictions : np.array<bins * bins * test trials>
        The output of the classifier for each pair of train and test bins
    '''
    # Initialization of variables
    ntests, nbins, _ = X_test.shape

    predictions = np.empty((ntests, nbins, nbins))
    accuracies = np.empty((nbins, nbins))

    # A classifier is trained on each bin, and then tested on every bins
    for itrain in range(nbins):
        X_bin_train = X_train[:, itrain]
        model = OneVsOneClassifier(RidgeClassifier(alpha=alpha, solver='cholesky'))
        model.fit(X_bin_train, y_train)

        for itest in range(nbins):
            X_bin_test = X_test[:, itest]
            bin_preds = model.predict(X_bin_test)
            accuracy = (bin_preds == y_test).mean(0)
            predictions[:, itrain, itest] = bin_preds
            accuracies[itrain, itest] = accuracy

    return accuracies, predictions


def CTD_vectorized_testing(X_train, X_test, y_train, y_test, alpha=1):
    '''Cross-temporal decoding with cross-validation in subspace

    Parameters
    ----------
    X_train : np.array<trials * bins * neurons> - training data
    X_test : np.array<trials * bins * neurons> - testing data
    y_train : np.array<trials> - training labels
    y_test : np.array<trials> - testing labels
    alpha : float - the L2 "ridge" regularization parameter

    Returns
    -------
    accuracy : np.array<bins * bins> of floats
        The fraction of correctly classified trials for each pair of train and
        test bins
    predictions : np.array<bins * bins * test trials>
        The output of the classifier for each pair of train and test bins
    '''
    # Initialization of variables
    ntests, nbins, _ = X_test.shape

    predictions = np.empty((ntests, nbins, nbins))
    accuracies = np.empty((nbins, nbins))

    ### A decoder is trained on each bin, and each decoder is tested on every bins
    for itrain in range(nbins):
        X_bin_train = X_train[:, itrain]
        model = OneVsOneClassifier(RidgeClassifier(alpha=alpha, solver='cholesky'))
        model.fit(X_bin_train, y_train)

        # The test data is reshaped to test all the bins in a single shot (much faster)
        X_test_ = X_test.reshape(np.product(X_test.shape[:2]), X_test.shape[2])
        preds = model.predict(X_test_)
        # The output is reshaped to the original shape of the test data
        preds = preds.reshape(ntests, nbins)
        accs = (preds == y_test[:, None]).mean(0)
        predictions[:, itrain, :] = preds
        accuracies[itrain, :] = accs

    return accuracies, predictions

###############################################################################
# All the following functions are part of the fully vectorized version

def balance_classes(X_train, y_train):
    '''Get the same number of trials for each class

    Parameters
    ----------
    X_train : np.array<trials * bins * neurons> - training data
    y_train : np.array<trials> - training labels

    Returns
    -------
    X_train_cut : np.array<bins * trials * neurons> - class-balanced training data
    y_train_cut : np.array<trials> - class-balanced training labels
    ntrials_per_class : int - the number of trials in each class
    '''
    labels = np.unique(y_train)
    _, traincounts = np.unique(y_train, return_counts=True)
    ntrials_per_class = np.min(traincounts)
    if not np.all(traincounts[0] == traincounts):
        ntrials_per_class = np.min(traincounts)
        keptind = []
        for iclass, count in enumerate(traincounts):
            if count > ntrials_per_class:
                inds = np.where(y_train == labels[iclass])[0][:-(count-ntrials_per_class)]
            else:
                inds = np.where(y_train == labels[iclass])[0]
            keptind.append(inds)
        keptind = np.concatenate(keptind)
    else:
        keptind = np.arange(len(y_train))
    y_train_cut = y_train[keptind]
    X_train_cut = X_train[:, keptind]

    return X_train_cut, y_train_cut, ntrials_per_class


def format_data_vec(X_train, y_train, X_test, ntrials_per_class):
    '''Format the data so that regressions can be vectorized

    Parameters
    ----------
    X_train : np.array<bins * trials * neurons> - training data
    y_train : np.array<trials> - training labels
    X_test : np.array<trials * bins * neurons> - testing data

    Returns
    -------
    X_train_vec : np.array<bins * regression * trials * neurons> - training data
        for vectorized training
    y_train_vec : np.array<regression * trials> - training labels for vectorized
        training
    X_test_vec : np.array<trials & bins * neurons> - testing data for vectorized
        testing
    '''
    nbins = X_train.shape[0]
    nclasses = len(np.unique(y_train))
    nestimators = (nclasses * (nclasses - 1)) // 2
    nsamples = ntrials_per_class * 2
    nfeatures = X_train.shape[-1]
    y_train_vec = np.empty((nestimators, nsamples))
    X_train_vec = np.empty((nbins, nestimators, nsamples, nfeatures))

    k = 0
    for c1 in range(nclasses):
        for c2 in range(c1+1, nclasses):
            cond = np.logical_or(y_train == c1, y_train == c2)
            y_train_vec[k, y_train[cond] == c1] = -1
            y_train_vec[k, y_train[cond] == c2] = 1
            X_train_vec[:, k] = X_train[:, cond]
            k += 1
    X_test_vec = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2])
    return X_train_vec, y_train_vec, X_test_vec


def cholesky_regression_vec(X_train, y_train, alpha=1):
    '''Vectorized version of cholesky regression on all data points and classes

    Parameters
    ----------
    X_train : np.array<bins * regression * trials * neurons> - training data
    y_train : np.array<regression * trials> - training labels
    alpha : float - ridge regularization parameter

    Returns
    -------
    coefs : np.array<bins * regression * neurons * 1> - regression coefficients
    intercepts : np.array<bins * regression * 1 * 1> - regression intercepts
    '''
    nfeatures = X_train.shape[-1]
    nclasses = len(np.unique(y_train))
    nsamples = X_train.shape[2] // nclasses
    nestimators = X_train.shape[1]

    X_offset = X_train.mean(2, keepdims=True)
    X_train -= X_offset

    # If the data set has more features than samples, the Cholesky method is
    # slightly different, here I still follow the scikit-learn code, with an
    # extra dimension
    if nfeatures > nsamples:
        XXT = X_train @ X_train.transpose((0, 1, 3, 2))
        XXT = XXT + np.eye(XXT.shape[-1])[None, None, ...] * alpha
        dual_coef = np.linalg.solve(XXT, y_train.reshape(1, nestimators, -1))
        coefs = X_train.transpose((0, 1, 3, 2)) @ dual_coef.reshape(dual_coef.shape[0], nestimators, -1, 1)
    else:
        XTX = X_train.transpose((0, 1, 3, 2)) @ X_train
        Xy = X_train.transpose((0, 1, 3, 2)) @ y_train.reshape((1, y_train.shape[0], -1, 1))
        XTX = XTX + np.eye(XTX.shape[-1])[None, None, ...] * alpha
        coefs = np.linalg.solve(XTX, Xy)

    intercepts = - X_offset @ coefs
    return coefs, intercepts


def predictions_vec(X_test_vec, coefs, intercepts, nclasses):
    '''Predict classes from concatenated test data

    Parameters
    ----------
    X_test_vec : np.array<trials & bins * neurons> - testing data
    coefs : np.array<bins * regression * neurons * 1> - regression coefficients
    intercepts : np.array<bins * regression * 1 * 1> - regression intercepts
    nclasses : int - number of classes

    Returns
    -------
    preds : np.array<bins (train) * trials & bins (test)> - predictions
    '''
    nbins = coefs.shape[0]
    scores = (X_test_vec @ coefs) + intercepts
    scores = scores.reshape(scores.shape[:-1])
    predictions = (scores > 0).astype(int)
    nsamples = predictions.shape[-1]
    predsT = predictions.transpose((0, 2, 1))
    scoresT = scores.transpose((0, 2, 1))
    votes = np.zeros((nbins, nsamples, nclasses))
    sum_of_confidences = np.zeros((nbins, nsamples, nclasses))
    k = 0
    for i in range(nclasses):
        for j in range(i + 1, nclasses):
            sum_of_confidences[:, :, i] -= scoresT[:, :, k]
            sum_of_confidences[:, :, j] += scoresT[:, :, k]
            votes[predsT[:, :, k] == 0, i] += 1
            votes[predsT[:, :, k] == 1, j] += 1
            k += 1
    transformed_confidences = (sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1)))
    preds = np.argmax(votes + transformed_confidences, 2)
    return preds


def vectorized_CTD(X_train, X_test, y_train, y_test, alpha=1):
    '''Vectorized cross-temporal decoding

    This is a vectorized version of the cross-temporal decoding algorithm. The
    six decoders (in a one vs one scheme) are trained simultaneously thanks to
    vectorization which considerably speeds up computations. Unfortunately it
    makes the code less readable. The decoding algorithm was inspired by scikit
    learn's implementation of ridge regression. Note that to be able to
    vectorize training and testing, each class must have the same number of
    training and testing trials.

    Parameters
    ----------
    X_train : np.array<trials * bins * neurons> - training data
    X_test : np.array<trials * bins * neurons> - testing data
    y_train : np.array<trials> - training labels
    y_test : np.array<trials> - testing labels
    alpha : float - the L2 "ridge" regularization parameter

    Returns
    -------
    accuracy : np.array<bins * bins> of floats
        The fraction of correctly classified trials for each pair of train and
        test bins
    predictions : np.array<bins * bins * test trials>
        The output of the classifier for each pair of train and test bins
    '''
    ntests, nbins, _ = X_test.shape

    nclasses = len(np.unique(y_train))

    X_train = X_train.transpose((1, 0, 2))

    # We need to have the exact same number of trials for each class to be able
    # to vectorize
    X_train_cut, y_train_cut, ntrials_per_class = balance_classes(X_train, y_train)
    # Format the data so that regressions for a single classification are now a
    # new dimension
    X_train_vec, y_train_vec, X_test_vec = format_data_vec(X_train_cut, y_train_cut, X_test, ntrials_per_class)
    # The following is the linear algebra method to obtain the coefficients of
    # each regression
    coefs, intercepts = cholesky_regression_vec(X_train_vec, y_train_vec, alpha=1)
    # Predict classes of test data
    preds = predictions_vec(X_test_vec, coefs, intercepts, nclasses)
    # Reshape concatenated test data into original shape
    preds = preds.reshape(nbins, ntests, nbins)

    accuracy = (preds == y_test[None, :, None]).mean(1)
    return accuracy, preds
