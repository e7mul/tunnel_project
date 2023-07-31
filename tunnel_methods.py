import os
from abc import abstractmethod
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class LinearProbe(nn.Module):
    def __init__(self, feature_size, task_size, epochs=30, lr=1e-3):
        super().__init__()
        self.linear_probe = nn.Linear(feature_size, task_size)
        self.optimizer = torch.optim.Adam(self.linear_probe.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.best_val_acc = 0.0
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.linear_probe = self.linear_probe.cuda()

    def forward(self, x):
        return self.linear_probe(x)

    @staticmethod
    def create_loaders(dataset, split_ratio=(0.8, 0.2)):
        X, y = dataset
        dataset = TensorDataset(X, y)
        datasets = random_split(dataset, split_ratio)
        loader = partial(DataLoader, batch_size=512, shuffle=True, num_workers=0)
        loaders = [loader(dataset) for dataset in datasets]
        return loaders

    def train(self, X, y):
        train_loader, val_loader = self.create_loaders((X, y))
        for epoch in range(self.epochs):
            epoch_loss, total, correct = 0, 0, 0
            for e, (X, y) in enumerate(train_loader):
                if self.cuda:
                    X, y = X.cuda(), y.cuda()
                output = self.forward(X)
                loss = self.criterion(output, y) + 1e-3 * self.linear_probe.weight.norm()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss
                correct += torch.sum(torch.topk(output, axis=1, k=1)[1].squeeze(1) == y)
                total += len(X)
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(
                    f"Epoch {epoch}: \t Loss {epoch_loss/(e+1):.3f} \t Acc: {correct/total:.3f}"
                )
            val_acc = self.validation(val_loader)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_best_model()
        print(f"Best val acc: {self.best_val_acc:.3f}")

    @torch.no_grad()
    def validation(self, val_loader):
        self.linear_probe.eval()
        correct, total = 0, 0
        for X, y in val_loader:
            if self.cuda:
                X, y = X.cuda(), y.cuda()
            output = self.forward(X)
            correct += torch.sum(
                torch.topk(output, axis=1, k=1)[1].squeeze(1) == y
            ).item()
            total += len(X)
        self.linear_probe.train()
        return correct / total

    def evaluate(self, activations_dataset):
        val_activs = self.create_loaders(activations_dataset, [1.0])[0]
        return self.validation(val_activs)

    def save_best_model(self):
        W = self.linear_probe.state_dict()["weight"].clone().cpu().detach().numpy()
        b = self.linear_probe.state_dict()["bias"].clone().cpu().detach().numpy()
        self.best_model = {"W": W, "b": b}

    def load_best_model(self):
        self.linear_probe.state_dict()["weight"].copy_(
            torch.from_numpy(self.best_model["W"])
        )
        self.linear_probe.state_dict()["bias"].copy_(
            torch.from_numpy(self.best_model["b"])
        )
        if self.cuda:
            self.linear_probe.cuda()


class BaseAnalysis:
    def export(self, name):
        torch.save(self.result, os.path.join(self.rpath, name + ".pt"))

    def clean_up(self):
        for attr in self.attributes_on_gpu:
            try:
                a = getattr(self, attr)
                a.to("cpu")
                del a
            except AttributeError:
                pass
        del self
        torch.cuda.empty_cache()

    @abstractmethod
    def analysis(self):
        pass

    @abstractmethod
    def plot(self, path):
        pass


class RepresentationsSpectra(BaseAnalysis):
    def __init__(self, model, data, layers, rpath, MAX_REPR_SIZE=8000):
        self.model = model
        self.data = torch.utils.data.DataLoader(data, batch_size=256, shuffle=False)
        self.layers_to_analyze = layers
        self.handels = []
        self._insert_hooks()
        self.representations = {}
        self.rpath = rpath
        self.MAX_REPR_SIZE = MAX_REPR_SIZE
        os.makedirs(self.rpath, exist_ok=True)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model = self.model.cuda()
        self.attributes_on_gpu = ["model"]

    def _spectra_hook(self, name):
        def spectra_hook(model, input, output):
            representation_size = int(output.numel()/output.shape[0])
            output = output.flatten(1)
            if representation_size > self.MAX_REPR_SIZE:                
                output = output[:, np.random.choice(representation_size, self.MAX_REPR_SIZE, replace=False)]
            self.representations[name] = self.representations.get(name, []) + [output]
        return spectra_hook

    def _insert_hooks(self):
        for name, layer in self.model.named_modules():
            if name in self.layers_to_analyze:
                self.handels.append(layer.register_forward_hook(self._spectra_hook(name)))

    @torch.no_grad()
    def collect_representations(self):
        self.model.eval()
        with torch.no_grad():
            for x, *_ in self.data:
                if torch.cuda.is_available():
                    x = x.cuda()
                _ = self.model(x)
        for name, rep in self.representations.items():
            self.representations[name] = torch.cat(rep, dim=0).cpu().detach()
        for handle in self.handels:
            handle.remove()
        return self.representations

    def analysis(self):
        if len(self.representations) == 0:
            self.collect_representations()
        self.result = {"rank": {}}
        for name, rep in self.representations.items():
            rep = torch.cov(rep.T)
            self.result["rank"][name] = torch.linalg.matrix_rank(rep)

    def plot(self, name):
        effective_rank = self.result["rank"]
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.plot(list(effective_rank.keys()), list(effective_rank.values()), "o-")
        plt.savefig(os.path.join(self.rpath, name + ".png"), dpi=500)
        plt.close()


class EarlyExit(BaseAnalysis):
    def __init__(
        self,
        backbone,
        train_data,
        test_data,
        layers,
        rpath,
        MAX_REPRESENTATION_SIZE=10000,
    ):
        self.backbone = backbone
        self.result = {}
        self.handles = {}
        self.train_data = train_data
        self.test_data = test_data
        self.cuda = torch.cuda.is_available()
        self.layers = layers
        self.rpath = rpath
        self.MAX_REPRESENTATION_SIZE = MAX_REPRESENTATION_SIZE
        self.random_indices = {}
        self._clean_up()
        self.attributes_on_gpu = ["backbone"]

    def _clean_up(self):
        self.activs = torch.tensor([])
        self.labels = []
        for handle in self.handles.values():
            handle.remove()
        self.handles = {}

    def analysis(self):
        train_loader = DataLoader(self.train_data, batch_size=1024, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=1024, shuffle=True)

        for layer_name in self.layers:
            print("Linear probe for layer:", layer_name)
            X_train, y_train = self.collect_activations(train_loader, layer_name)
            self._clean_up()
            X_test, y_test = self.collect_activations(test_loader, layer_name)
            self.backbone = self.backbone.cpu()
            linear_head = self.train(X_train, y_train)
            self.result[layer_name] = linear_head.evaluate((X_test, y_test))
            self._clean_up()
        return self.result

    def _hook(self, m, i, o, layer_name):
        output = o.flatten(start_dim=1).cpu().detach()
        if output.shape[1] > self.MAX_REPRESENTATION_SIZE:
            if layer_name not in self.random_indices:
                random_indices = np.random.choice(
                    output.shape[1], self.MAX_REPRESENTATION_SIZE, replace=False
                )
                self.random_indices[layer_name] = random_indices

            output = output[:, self.random_indices[layer_name]]
        self.activs = torch.cat((self.activs, output))

    def _insert_hook(self, layer_name):
        for name, layer in self.backbone.named_modules():
            if name == layer_name:
                hook = partial(self._hook, layer_name=name)
                self.handles[name] = layer.register_forward_hook(hook)

    @torch.no_grad()
    def collect_activations(self, loader, layer_name):
        self._insert_hook(layer_name)
        self.backbone.eval()
        if self.cuda:
            self.backbone = self.backbone.cuda()
        for input, targets, *_ in loader:
            if self.cuda:
                input = input.cuda()
            self.backbone.forward(input)
            self.labels += [y.item() for y in targets]
        self.labels = torch.tensor(self.labels)
        self.backbone = self.backbone.cpu()
        return self.activs, self.labels

    def train(self, X, y):
        num_classes = len(set(self.labels))
        head = LinearProbe(self.activs[0].shape[-1], num_classes)
        head.train(X, y)
        return head

    def plot(self, name):
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.plot([i for i in self.result.values()], "o-")
        axs.grid()
        plt.title("Early Exits")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(self.rpath, name + ".png"), dpi=500)
        plt.close()


class LDA(RepresentationsSpectra):
        def analysis(self):
            self.input_shape = {}
            self.result = {'lda':{}, 'inter':{}, 'intra':{}}
            if len(self.representations) == 0:
                self.collect_representations_and_labels()
            for layer in self.representations.keys():
                self.result['lda'][layer], self.result['inter'][layer], self.result['intra'][layer] = self.discriminability(layer)
            return self.result


        def _spectra_hook(self, name):
            def spectra_hook(model, input, output):
                MAX_REPR_SIZE = 100000
                representation_size = int(output.numel()/output.shape[0])
                output = output.flatten(1)
                if representation_size > MAX_REPR_SIZE:                
                    output = output[:, np.random.choice(representation_size, MAX_REPR_SIZE, replace=False)]
                self.representations[name] = self.representations.get(name, []) + [output]
                self.input_shape[name] = [i for i in input[0].shape[2:]]
            return spectra_hook

        @torch.no_grad()
        def collect_representations_and_labels(self):
            self.labels = []
            self.model.eval()
            with torch.no_grad():
                for x, y, *_ in self.data:
                    if torch.cuda.is_available():
                        x = x.cuda()
                    _ = self.model(x)
                    self.labels += [i.item() for i in y]
            for name, rep in self.representations.items():
                self.representations[name] = torch.cat(rep, dim=0).cpu().detach()
            
            for handle in self.handels:
                handle.remove()
            return self.representations
            
        def discriminability(self, layer,):
            phi = self.representations[layer]
            print(layer, phi.shape)
            labels = np.array(self.labels)
            phi_norm = torch.nn.functional.normalize(phi,p=2,dim=1)
            class_centers = {i:phi_norm[labels==i].mean(dim=0) for i in np.unique(labels)}
            inter_class = torch.pdist(torch.stack([class_centers[i] for i in np.unique(labels)])).pow(2).mean()
            intra_class = np.mean([(phi_norm[labels==i] - class_centers[i]).norm(dim=1).pow(2).mean() for i in np.unique(labels)])
            return inter_class/intra_class, inter_class, intra_class
        
        def plot(self, name):
            fig, axs = plt.subplots(1, 1, figsize=(5, 4))
            axs.plot(self.result['inter'].values(), 'o-',label='Inter-class variance')
            axs.plot(self.result['intra'].values(),'o-', label='Intra-class variance')
            axs.set_xticks(np.arange(len(self.result['lda'].values())),np.arange(len(self.result['lda'].values()))+1, rotation=0, fontsize=10)
            axs.set_xlabel("Layer", fontsize=12)
            axs.set_xlim(-.4, len(self.result['inter'].values())-0.6)
            plt.legend(fontsize=12)
            print(os.path.join(self.rpath, name + ".png"))
            plt.savefig(os.path.join(self.rpath, name + ".png"), dpi=500)
            plt.close()


class TSNE(LDA):
        def analysis(self):
            self.input_shape = {}
            self.result = {}
            if len(self.representations) == 0:
                self.collect_representations_and_labels()
            for layer in self.representations.keys():
                self.result[layer] = self.tsne(layer)
            return self.result

        def tsne(self, layer):
            phi = self.representations[layer]
            X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=30, n_iter=1000).fit_transform(phi)
            return X_embedded
        
        def plot(self):
            for e, (key, val) in enumerate(self.result.items()):
                fig, axs = plt.subplots(1, 1, figsize=(6, 6))
                axs.scatter(val[:,0], val[:,1], s=0.5, label=key, color=plt.cm.Set1(np.array(self.labels)/10.))
                axs.axis('off')
                plt.savefig(os.path.join(self.rpath, f"t-sne_{key}.png"), dpi=500)


if __name__ == "__main__":
    pass
