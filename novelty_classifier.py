import numpy as np
from thundersvm import OneClassSVM
from tqdm import tqdm


class NoveltyDetectionClassifier(object):
    def __init__(self, nu_high, nu_low, nu_resolution, gamma="scale"):
        self.nu_high = nu_high
        self.nu_low = nu_low
        self.nu_resolution = nu_resolution

        # -- Gamma of "auto" corresponds to 1/n_features
        # -- Gamma of "scale" corresponds to 1/(n_features * X.var())
        # -- Depending on whether the var is less than or greater than 1,
        # -- setting gamma to "scale" either leads to a smooth or complex decision boundary
        # -- Gamma can also be a floating point number
        self.gamma = gamma

        self.classifiers = []

    def __call__(self, X):
        return self.predict(X)

    def determine_gamma(self, X):
        if isinstance(self.gamma, (int, float)):
            return self.gamma

        n_features = X.shape[1]

        if self.gamma == "auto":
            return 1. / n_features
        if self.gamma == "scale":
            return 1. / (n_features * X.var())

        raise ValueError(self.gamma)

    def create_one_class_classifier(self, nu, X):
        gamma = self.determine_gamma(X)
        clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        return clf

    def create_family_of_classifiers(self, X):
        nu_range = np.arange(self.nu_low, self.nu_high, self.nu_resolution)
        classifiers = [self.create_one_class_classifier(nu, X) for nu in nu_range]
        return classifiers

    def fit(self, X):
        self.classifiers = self.create_family_of_classifiers(X)
        for classifier in tqdm(self.classifiers, desc="Fitting OC-SVMs"):  # type: OneClassSVM
            classifier.fit(X)

    def predict(self, X):  # TODO: Chunk up inference
        overall_predictions = []
        for classifier in self.classifiers:  # type: OneClassSVM
            clf_predictions = classifier.predict(X) == 1
            overall_predictions.append(clf_predictions)
        overall_predictions = np.array(overall_predictions)
        prediction_probabilities = np.mean(overall_predictions, axis=0)
        return prediction_probabilities
