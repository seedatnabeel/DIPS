import numpy as np
import torch
import torch.nn as nn


class DIPS_Torch:
    def __init__(self, X=None, y=None, dataloader=None, sparse_labels: bool = False):
        """
        The function takes in the training data and the labels, and stores them in the class variables X
        and y. It also stores the boolean value of sparse_labels in the class variable _sparse_labels

        Args:
          X: the input data
          y: the true labels
          sparse_labels (bool): boolean to identify if labels are one-hot encoded or not. If not=True.
        Defaults to False
        """
        self.X = X
        self.y = y
        self.dataloader = dataloader
        self._sparse_labels = sparse_labels

        # placeholder
        self._gold_labels_probabilities = None
        self._true_probabilities = None
        self._grads = None
        self._labels = None
        self._logits = None

    def on_epoch_end(self, net, device="cpu", gradient=False, **kwargs):
        """
        The function computes the gold label and true label probabilities over all samples in the
        dataset

        We iterate through the dataset, and for each sample, we compute the gold label probability (i.e.
        the actual ground truth label) and the true label probability (i.e. the predicted label).

        We then append these probabilities to the `_gold_labels_probabilities` and `_true_probabilities`
        lists.

        We do this for every sample in the dataset, and for every epoch.

        Args:
          net: the neural network
          device: the device to use for the computation. Defaults to cpu
        """

        # Compute both the gold label and true label probabilities over all samples in the dataset
        gold_label_probabilities = (
            list()
        )  # gold label probabilities, i.e. actual ground truth label
        true_probabilities = list()  # true label probabilities, i.e. predicted label
        softmax = nn.Softmax(dim=-1)
        labels = list()
        logits = list()
        with torch.no_grad():
            # iterate through the dataset

            for idx, (x, y) in enumerate(self.dataloader):

                x = x.to(device)
                y = y.to(device)

                probabilities = net(x)
                logit = probabilities
                # forward pass
                probabilities = softmax(probabilities)

                y_true = y

                # one hot encode the labels
                y = torch.nn.functional.one_hot(
                    y.to(torch.int64), num_classes=probabilities.shape[-1]
                )

                # Now we extract the gold label and predicted true label probas
                # If the labels are binary [0,1]
                if len(torch.squeeze(y)) == 1:
                    # get true labels
                    true_probabilities = torch.tensor(probabilities)

                    # get gold labels
                    probabilities, y = torch.squeeze(
                        torch.tensor(probabilities)
                    ), torch.squeeze(y)

                    batch_gold_label_probabilities = torch.where(
                        y == 0, 1 - probabilities, probabilities
                    )

                # if labels are one hot encoded, e.g. [[1,0,0], [0,1,0]]
                elif len(torch.squeeze(y)) == 2:
                    # get true labels
                    batch_true_probabilities = torch.max(probabilities)

                    # get gold labels
                    batch_gold_label_probabilities = torch.masked_select(
                        probabilities, y.bool()
                    )
                else:

                    # get true labels
                    batch_true_probabilities = torch.max(probabilities, dim=1)[0]

                    # get gold labels
                    batch_gold_label_probabilities = torch.masked_select(
                        probabilities, y.bool()
                    )

                    batch_labels = y_true

                # move torch tensors to cpu as np.arrays()
                batch_gold_label_probabilities = (
                    batch_gold_label_probabilities.cpu().numpy()
                )
                batch_true_probabilities = batch_true_probabilities.cpu().numpy()

                batch_labels = batch_labels.cpu().numpy()

                # Append the new probabilities for the new batch
                gold_label_probabilities = np.append(
                    gold_label_probabilities, [batch_gold_label_probabilities]
                )
                true_probabilities = np.append(
                    true_probabilities, [batch_true_probabilities]
                )

                labels = np.append(labels, [batch_labels])

                logits = np.append(logits, [logit.detach().cpu().numpy()])

        # Append the new gold label probabilities
        if self._gold_labels_probabilities is None:  # On first epoch of training
            self._gold_labels_probabilities = np.expand_dims(
                gold_label_probabilities, axis=-1
            )
        else:
            stack = [
                self._gold_labels_probabilities,
                np.expand_dims(gold_label_probabilities, axis=-1),
            ]
            self._gold_labels_probabilities = np.hstack(stack)

        # Append the new true label probabilities
        if self._true_probabilities is None:  # On first epoch of training
            self._true_probabilities = np.expand_dims(true_probabilities, axis=-1)
        else:
            stack = [
                self._true_probabilities,
                np.expand_dims(true_probabilities, axis=-1),
            ]
            self._true_probabilities = np.hstack(stack)

        if self._labels is None:  # On first epoch of training
            self._labels = np.expand_dims(labels, axis=-1)
        else:
            stack = [
                self._labels,
                np.expand_dims(labels, axis=-1),
            ]
            self._labels = np.hstack(stack)

        if self._logits is None:  # On first epoch of training
            self._logits = np.expand_dims(logits, axis=-1)
        else:
            stack = [
                self._logits,
                np.expand_dims(logits, axis=-1),
            ]
            self._logits = np.hstack(stack)

    @property
    def get_grads(self):
        """
        Returns:
            Grad norm through training: np.array(n_samples, n_epochs)
        """
        return self._grads

    @property
    def gold_labels_probabilities(self) -> np.ndarray:
        """
        Returns:
            Gold label predicted probabilities of the "correct" label: np.array(n_samples, n_epochs)
        """
        return self._gold_labels_probabilities

    @property
    def true_probabilities(self) -> np.ndarray:
        """
        Returns:
            Actual predicted probabilities of the predicted label: np.array(n_samples, n_epochs)
        """
        return self._true_probabilities

    @property
    def confidence(self) -> np.ndarray:
        """
        Returns:
            Average predictive confidence across epochs: np.array(n_samples)
        """
        return np.mean(self._gold_labels_probabilities, axis=-1)

    @property
    def aleatoric(self):
        """
        Returns:
            Aleatric uncertainty of true label probability across epochs: np.array(n_samples): np.array(n_samples)
        """
        preds = self._gold_labels_probabilities
        return np.mean(preds * (1 - preds), axis=-1)

    @property
    def variability(self) -> np.ndarray:
        """
        Returns:
            Epistemic variability of true label probability across epochs: np.array(n_samples)
        """
        return np.std(self._gold_labels_probabilities, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        """
        Returns:
            Proportion of times a sample is predicted correctly across epochs: np.array(n_samples)
        """
        return np.mean(self._gold_labels_probabilities > 0.5, axis=-1)

    @property
    def entropy(self):
        """
        Returns:
            Predictive entropy of true label probability across epochs: np.array(n_samples)
        """
        X = self._gold_labels_probabilities
        return -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

    @property
    def mi(self):
        """
        Returns:
            Mutual information of true label probability across epochs: np.array(n_samples)
        """
        X = self._gold_labels_probabilities
        entropy = -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

        X = np.mean(self._gold_labels_probabilities, axis=1)
        entropy_exp = -1 * np.sum(X * np.log(X + 1e-12), axis=-1)
        return entropy - entropy_exp
