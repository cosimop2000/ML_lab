"""
Class that models a Naive Bayes Classifier
"""

import numpy as np


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier.
    Training:
    For each class, a naive likelihood model is estimated for P(X/Y),
    and the prior probability P(Y) is computed.
    Inference:
    performed according with the Bayes rule:
    P = argmax_Y (P(X/Y) * P(Y))
    or
    P = argmax_Y (log(P(X/Y)) + log(P(Y)))
    """

    def __init__(self):
        """
        Class constructor
        """

        self._classes = None
        self._n_classes = 0

        self._eps = np.finfo(np.float32).eps

        # array of classes prior probabilities
        self._class_priors = []

        # array of probabilities of a pixel being active (for each class)
        self._pixel_probs_given_class = []

    def fit(self, X, Y):
        """
        Computes, for each class, a naive likelihood model (self._pixel_probs_given_class),
        and a prior probability (self.class_priors).
        Both quantities are estimated from examples X and Y.

        Parameters
        ----------
        X: np.array
            input MNIST digits. Has shape (n_train_samples, h, w)
        Y: np.array
            labels for MNIST digits. Has shape (n_train_samples,)
        """
        # X --> (N, 28,28), N = X.shape[0]
        # Y --> (N) 10
        y_classes, cnt = np.unique(Y, return_counts=True)
        self._classes = y_classes
        self._n_classes = len(y_classes)
        # cnt = np.sum(Y == c)

        self._class_priors = cnt
        self._class_priors = self._class_priors / X.shape[0]
        # for c in self._classes
        # clss_priors[c] = np.sum(Y == c) / X.shape[0]
        # append
        # se avessi trasformato _class_priors in np array avrei dovuto usare l'assegnamento con [c] e non append

        print(self._n_classes, self._classes, self._class_priors, np.sum(self._class_priors), X.shape[0])

        # likelihood

        # mean faccio la media in verticale tra gli N (axis = 0) quadranti 28 per 28
        # dei pixel attivi(1 True) per ogni classe (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

        for c in range(self._n_classes):
            # print(Y == c)
            prob_pixel_i = np.mean(X[Y == c], axis=0)
            self._pixel_probs_given_class.append(prob_pixel_i)
        # print(self._pixel_probs_given_class)

    def predict(self, X):
        """
        Performs inference on test data.
        Inference is performed according with the Bayes rule:
        P = argmax_Y (log(P(X/Y)) + log(P(Y)) - log(P(X)))

        Parameters
        ----------
        X: np.array
            MNIST test images. Has shape (n_test_samples, h, w).

        Returns
        -------
        prediction: np.array
            model predictions over X. Has shape (n_test_samples,)
        """
        # calcolo probabilità usando log

        n_test_images = X.shape[0]

        X = X.reshape((n_test_images, -1))

        # matrice per immagazzinare lo score di ogni classe per ogni esempio
        results = np.zeros((n_test_images, self._n_classes))

        for c in range(self._n_classes):
            # dato un esempio x e i suoi pixel xj
            # xj -> 1 il valore p(xj|yc) è uguale alla likelihood in posizione j
            # xj -> 0 probabilità di osservare il pixel j in uno stato off, 1 - p

            # calcolo log p (X|y=C)
            model_of_i = self._pixel_probs_given_class[c]
            model_of_i = np.reshape(model_of_i, (1, X.shape[1]))

            # print(model_of_i)
            mask_one = X == 1.0
            mask_zero = X == 0.0
            # print(mask_one)
            # print(mask_zero)

            probs = mask_one * model_of_i * (1. - model_of_i)
            # print(probs)
            probs = np.log(probs + self._eps)
            # print(probs)
            probs = np.sum(probs, axis=1)
            # print(probs)
            probs += np.log(self._class_priors[c])
            # print(probs)
            results[:, c] = probs
            # print(results)

        return np.argmax(results, axis=1), results

    @staticmethod
    def _estimate_pixel_probabilities(images):
        """
        [OPTIONAL!]
        Estimates pixel probabilities from data.

        Parameters
        ----------
        images: np.array
            images to estimate pixel probabilities from. Has shape (n_images, h, w)

        Returns
        -------
        pix_probs: np.array
            probabilities for each pixel of being 1, estimated from images.
            Has shape (h, w)
        """
        return None

    def get_log_likelihood_under_model(self, images, model):
        """
        [OPTIONAL!]
        Returns the likelihood of many images under a certain model.
        Naive:
        the likelihood of the image is the product of the likelihood of each pixel.
        or
        the log-likelihood of the image is the sum of the log-likelihood of each pixel.

        Parameters
        ----------
        images: np.array
            input images. Having shape (n_images, h, w).
        model: np.array
            a model of pixel probabilities, having shape (h, w)

        Returns
        -------
        lkl: np.array
            the likelihood of each pixel under the model, having shape (h, w).
        """
        return None