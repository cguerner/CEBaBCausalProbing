import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from .abstract_explainer import Explainer
from eval_pipeline.utils import unpack_batches
from scipy.linalg import null_space
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from eval_pipeline.utils import OVERALL_LABEL, DESCRIPTION, TREATMENTS, POSITIVE, NEGATIVE, UNKNOWN
from eval_pipeline.explainers.explainer_utils import treatment_to_label, sort_dev_embeddings

class INLP(Explainer):
    # TODO make this class dependent on a specific seed
    def __init__(self, output_path=None, treatments=TREATMENTS, plus=False, device='cpu', batch_size=32, verbose=False,
                 classifier_params: dict = None, nsamples=None):
        self.device = device
        self.batch_size = batch_size
        self.projection_matrices, self.inlp_classifiers, self.inlp_dev_samples = {}, {}, {}
        self.treatments = treatments
        self.plus = plus
        self.nsamples = nsamples
        if self.plus:
            assert self.nsamples is not None, "Need nsamples value"
        self.figure_path = None
        self.model = None
        if output_path:
            self.figure_path = os.path.join(output_path, 'inlp_figures')
            if not os.path.isdir(self.figure_path):
                os.makedirs(self.figure_path)
        self.verbose = verbose
        if not classifier_params:
            self.classifier_params = {'epochs': 5, 'learning_rate': 2e-5}

    def train_preprocess(self, dataset, treatment):
        treatment_labels = dataset[f'{treatment}_aspect_majority'].apply(treatment_to_label).dropna().astype(int)
        description = dataset[DESCRIPTION][treatment_labels.index].to_list()
        overall_labels = dataset[OVERALL_LABEL][treatment_labels.index].tolist()
        return self.model.get_embeddings(description), treatment_labels.tolist(), overall_labels

    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        self.model = classifier

        model_clf_head = self.model.get_classification_head()
        for treatment in self.treatments:
            preprocessed_dataset = self.train_preprocess(dataset, treatment)
            dev_preprocessed = self.train_preprocess(dev_dataset, treatment)
            self.inlp_dev_samples[treatment] = sort_dev_embeddings(dev_preprocessed, treatment)

            embeddings = np.array(unpack_batches(preprocessed_dataset[0]))
            dev_embeddings = np.array(unpack_batches(dev_preprocessed[0]))
            self.projection_matrices[treatment] = self.inlp_method(
                embeddings, preprocessed_dataset[1], 
                dev_embeddings, dev_preprocessed[1], 
                treatment
            )
            inlp_train_embeddings = embeddings @ self.projection_matrices[treatment]
            inlp_dev_embeddings = dev_embeddings @ self.projection_matrices[treatment]
            overall_labels = np.array(preprocessed_dataset[2])
            dev_overall_labels = np.array(dev_preprocessed[2])
            if self.plus:
                self.inlp_classifiers[treatment] = None
            else:
                self.inlp_classifiers[treatment] = self.train_clf(
                    inlp_train_embeddings, overall_labels, 
                    inlp_dev_embeddings, dev_overall_labels,
                    clf_model=deepcopy(model_clf_head),
                    clf_name=f'inlp_overall_task_{treatment}'
                ).float()
                                    

    def estimate_icace_old(self, pairs):
        probas_lst = []
        embeddings, intervention_type = self.test_preprocess(pairs)
        clf_head = self.model.get_classification_head()
        for idx, embedding in enumerate(embeddings):
            with torch.no_grad():
                inlp_clf = self.inlp_classifiers[intervention_type.iloc[idx]]
                probas = torch.softmax(clf_head(torch.tensor(embedding, dtype=torch.float32).to(self.model.device)), dim=0).cpu()
                inlp_probas = torch.softmax(inlp_clf(
                    torch.tensor(embedding @ self.projection_matrices[intervention_type.iloc[idx]]).to(
                        self.device).float()), dim=0)
                probas_lst.append((inlp_probas.cpu() - probas).numpy())
        return list(probas_lst)

    def estimate_icace(self, pairs):
        # TODO: lots of overlap between INLP and Leace here, put to utils
        probas_lst = []
        embeddings, int_types, int_bases, int_counters = self.test_preprocess(pairs)
        clf_head = self.model.get_classification_head()
        if self.plus:
            zipped_input = zip(embeddings, int_types, int_bases, int_counters)
            for idx, (h, int_type, int_base, int_counter) in enumerate(zipped_input):
                with torch.no_grad():
                    probas = torch.softmax(clf_head(torch.tensor(h, dtype=torch.float32).to(self.model.device)), dim=0).cpu().numpy()
                    
                    int_I_P = self.projection_matrices[int_type] # eraser matrix found by INLP
                    int_P = np.eye(int_I_P.shape[0]) - int_I_P # concept subspace projector
                    int_type_dev_samples = self.inlp_dev_samples[int_type][int_counter]
                    idx = np.arange(0, int_type_dev_samples.shape[0])
                    np.random.shuffle(idx)
                    all_int_probs = []
                    for other_h in int_type_dev_samples[idx[:self.nsamples]]:
                        newh = torch.tensor(h @ int_I_P + other_h @ int_P).to(self.device).float()
                        int_probs = torch.softmax(clf_head(newh), dim=0)
                        all_int_probs.append(int_probs.cpu().numpy())
                    all_int_probs = np.vstack(all_int_probs)
                    int_probas = all_int_probs.mean(axis=0)
            
                    probas_lst.append(int_probas - probas)
        else:
            for idx, embedding in enumerate(embeddings):
                with torch.no_grad():
                    inlp_clf = self.inlp_classifiers[int_types.iloc[idx]]
                    probas = torch.softmax(clf_head(torch.tensor(embedding, dtype=torch.float32).to(self.model.device)), dim=0).cpu()
                    inlp_probas = torch.softmax(inlp_clf(
                        torch.tensor(embedding @ self.projection_matrices[int_types.iloc[idx]]).to(
                            self.device).float()), dim=0)
                    probas_lst.append((inlp_probas.cpu() - probas).numpy())
        return list(probas_lst)


    def inlp_method(self, X_train, y_train, X_dev, y_dev, treatment, iterations=10):
        train_accuracies, dev_accuracies = [], []
        X_projected = X_train
        p_intersection = np.eye(X_projected[0].shape[0], X_projected[0].shape[0])
        for _ in np.arange(iterations):
            # TODO make a decision which linear model we train
            # clf = LogisticRegression(max_iter=200).fit(X_projected, y_train)
            clf = linear_model.SGDClassifier(alpha=.01, max_iter=1000, tol=1e-3).fit(X_projected, y_train)
            w = clf.coef_
            preds_on_train = clf.predict(X_projected)
            train_accuracies.append(accuracy_score(preds_on_train, y_train))
            dev_accuracies.append(accuracy_score(clf.predict(X_dev @ p_intersection), y_dev))
            b = null_space(w)
            p_null_space = b @ b.T
            p_intersection = p_intersection @ p_null_space
            X_projected = (p_null_space @ X_projected.T).T

        if self.figure_path:
            plt.plot(train_accuracies, label='train'), plt.plot(dev_accuracies, label='dev')
            plt.title(f'probing {treatment} per iteration')
            plt.legend()
            plt.savefig(os.path.join(self.figure_path, treatment))
            plt.clf()

        return p_intersection

    def train_clf(self, X_train, y_train, X_dev, y_dev, clf_model, clf_name):
        if self.verbose:
            print(f'starting training {clf_name}')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(clf_model.parameters(), lr=self.classifier_params['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=self.verbose)
        train_embeddings = torch.from_numpy(X_train).float().to(self.device)
        dev_embeddings = torch.from_numpy(X_dev).float().to(self.device)
        train_labels = torch.from_numpy(y_train).float().to(self.device)
        dev_labels = torch.from_numpy(y_dev).float().to(self.device)
        clf_model = clf_model.to(self.device)
        train_accuracies = []
        dev_accuracies = []
        for epoch in range(self.classifier_params['epochs']):
            logits = clf_model(train_embeddings)
            loss = criterion(logits, train_labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predicted = torch.argmax(logits, 1)
            train_accuracy = (predicted == train_labels).sum() / len(train_labels)
            with torch.no_grad():
                dev_accuracy = (torch.argmax(clf_model(dev_embeddings), 1) == dev_labels).sum() / len(dev_labels)

            if self.verbose:
                print(f'{clf_name}- epoch: {epoch} loss: {loss:.3f} accuracy: {train_accuracy :.3f}, dev: {dev_accuracy :.3f}')
            scheduler.step(train_accuracy)
            train_accuracies.append(train_accuracy.cpu())
            dev_accuracies.append(dev_accuracy.cpu())

        clf_model.eval()
        if self.figure_path:
            plt.plot(train_accuracies, label='train'), plt.plot(dev_accuracies, label='dev')
            plt.title(clf_name)
            plt.legend()
            plt.savefig(os.path.join(self.figure_path, clf_name))
            plt.clf()
        return clf_model

    def test_preprocess(self, df):
        # TODO move this function to utils because of duplicate with inlp, and set these strings to constant
        x = np.array(unpack_batches(self.model.get_embeddings(df['description_base'].tolist())))
        
        intervention_type = df['intervention_type']
        intervention_aspect_base = df['intervention_aspect_base']
        intervention_aspect_counterfactual = df['intervention_aspect_counterfactual']
        return x, intervention_type, intervention_aspect_base, intervention_aspect_counterfactual