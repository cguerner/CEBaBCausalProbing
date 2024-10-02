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
from concept_erasure import LeaceEraser

from eval_pipeline.utils import OVERALL_LABEL, DESCRIPTION, TREATMENTS, POSITIVE, NEGATIVE, UNKNOWN
from eval_pipeline.explainers.explainer_utils import treatment_to_label, sort_dev_embeddings

class LeacePlus(Explainer):
    # TODO make this class dependent on a specific seed
    def __init__(self, output_path=None, treatments=TREATMENTS, plus=False, device='cpu', batch_size=32, verbose=False,
                 classifier_params: dict = None, nsamples=None):
        self.device = device
        self.batch_size = batch_size
        self.projection_matrices, self.leaceplus_classifiers, self.leaceplus_dev_samples, self.leaceplus_erasers = {}, {}, {}, {}
        self.treatments = treatments
        self.plus = plus
        self.nsamples = nsamples
        if self.plus:
            assert self.nsamples is not None, "Need nsamples value"
        self.figure_path = None
        self.model = None
        if output_path:
            self.figure_path = os.path.join(output_path, 'leaceplus_figures')
            if not os.path.isdir(self.figure_path):
                os.makedirs(self.figure_path)
        self.verbose = verbose
        if not classifier_params:
            self.classifier_params = {'epochs': 5, 'learning_rate': 2e-5}

    def train_preprocess(self, dataset, treatment):
        """ Returns (embeddings of review text (description), concept labels, overall label (i saw only binary)"""
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
            self.leaceplus_dev_samples[treatment] = sort_dev_embeddings(dev_preprocessed, treatment)
            embeddings = np.array(unpack_batches(preprocessed_dataset[0]))
            dev_embeddings = np.array(unpack_batches(dev_preprocessed[0]))
            self.projection_matrices[treatment] = self.leace_method(
                embeddings, preprocessed_dataset[1], 
                dev_embeddings, dev_preprocessed[1], 
                treatment
            )
            #leace_train_embeddings = embeddings @ self.projection_matrices[treatment]["I_P"]
            #leace_dev_embeddings = dev_embeddings @ self.projection_matrices[treatment]["I_P"]
            #leace_train_embeddings = self.leaceplus_erasers[treatment](torch.from_numpy(embeddings)).numpy()
            #leace_dev_embeddings = self.leaceplus_erasers[treatment](torch.from_numpy(dev_embeddings)).numpy()
            leace_train_embeddings = embeddings @ self.projection_matrices[treatment]["I_P"] + self.projection_matrices[treatment]["bias"] @ self.projection_matrices[treatment]["P"]
            leace_dev_embeddings = dev_embeddings @ self.projection_matrices[treatment]["I_P"] + self.projection_matrices[treatment]["bias"] @ self.projection_matrices[treatment]["P"]
            overall_labels = np.array(preprocessed_dataset[2])
            dev_overall_labels = np.array(dev_preprocessed[2])
            if self.plus:
                self.leaceplus_classifiers[treatment] = None
            else:
                self.leaceplus_classifiers[treatment] = self.train_clf(
                    leace_train_embeddings, overall_labels, 
                    leace_dev_embeddings, dev_overall_labels,
                    clf_model=deepcopy(model_clf_head),
                    clf_name=f'leaceplus_overall_task_{treatment}'
                ).float()

    def estimate_icace_old(self, pairs):
        probas_lst = []
        embeddings, intervention_type, _, _ = self.test_preprocess(pairs)
        clf_head = self.model.get_classification_head()
        for idx, embedding in enumerate(embeddings):
            with torch.no_grad():
                leaceplus_clf = self.leaceplus_classifiers[intervention_type.iloc[idx]]
                probas = torch.softmax(clf_head(torch.tensor(embedding, dtype=torch.float32).to(self.model.device)), dim=0).cpu()
                leaceplus_probas = torch.softmax(
                    leaceplus_clf(
                        torch.tensor(embedding @ self.projection_matrices[intervention_type.iloc[idx]]).to(
                        self.device).float()), dim=0)
                probas_lst.append((leaceplus_probas.cpu() - probas).numpy())
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
                    
                    int_type_Ps = self.projection_matrices[int_type]
                    int_type_dev_samples = self.leaceplus_dev_samples[int_type][int_counter]
                    idx = np.arange(0, int_type_dev_samples.shape[0])
                    np.random.shuffle(idx)
                    all_int_probs = []
                    for other_h in int_type_dev_samples[idx[:self.nsamples]]:
                        newh = torch.tensor(h @ int_type_Ps["I_P"] + other_h @ int_type_Ps["P"]).to(self.device).float()
                        int_probs = torch.softmax(clf_head(newh), dim=0)
                        all_int_probs.append(int_probs.cpu().numpy())
                    all_int_probs = np.vstack(all_int_probs)
                    int_probas = all_int_probs.mean(axis=0)
                    probas_lst.append(int_probas - probas)
        else:
            for idx, embedding in enumerate(embeddings):
                with torch.no_grad():
                    #leaceplus_eraser = self.leaceplus_erasers[int_types.iloc[idx]]
                    leaceplus_clf = self.leaceplus_classifiers[int_types.iloc[idx]]
                    int_type_Ps = self.projection_matrices[int_types.iloc[idx]]
                    probas = torch.softmax(clf_head(torch.tensor(embedding, dtype=torch.float32).to(self.model.device)), dim=0).cpu()
                    #leaceplus_probas = torch.softmax(
                    #    leaceplus_clf(
                    #        leaceplus_eraser(torch.from_numpy(embedding)).to(
                    #        self.device).float()), dim=0)
                    newh = torch.tensor(embedding @ int_type_Ps["I_P"] + int_type_Ps["bias"] @ int_type_Ps["P"]).to(self.device).float()
                    leaceplus_probas = torch.softmax(
                        leaceplus_clf(newh), 
                        dim=0
                    )
                    probas_lst.append((leaceplus_probas.cpu() - probas).numpy())
        return list(probas_lst)

    def leace_method(self, X_train, y_train, X_dev, y_dev, treatment):
        X_torch = torch.from_numpy(X_train)
        #NOTE: y_train has 3 values {neg, unk, pos}
        y_torch = torch.tensor(y_train)

        eraser = LeaceEraser.fit(X_torch, y_torch)
        P = (eraser.proj_right.mH @ eraser.proj_left.mH).numpy()
        I_P = np.eye(X_train.shape[1]) - P
        bias = eraser.bias.numpy()
        self.leaceplus_erasers[treatment] = eraser

        clf = linear_model.SGDClassifier(alpha=.01, max_iter=1000, tol=1e-3).fit(X_train , y_train)
        train_accuracy = accuracy_score(clf.predict(X_train), y_train)
        dev_accuracy = accuracy_score(clf.predict(X_dev @ I_P), y_dev)

        if self.figure_path:
            plt.plot(train_accuracies, label='train'), plt.plot(dev_accuracies, label='dev')
            plt.title(f'probing {treatment} per iteration')
            plt.legend()
            plt.savefig(os.path.join(self.figure_path, treatment))
            plt.clf()

        res = {
            "P": P,
            "I_P": I_P,
            "bias": bias
        }
        return res

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
