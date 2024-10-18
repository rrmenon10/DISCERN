import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from typing import List

from src.utils.Config import Config

class Cluster:
    '''
    This saves the cluster name, cluster id
    '''
    cluster_class: int = None
    cluster_id: int = None
    num_clusters: int = None
    examples: List[str] = None
    labels: np.array = None
    logits: np.array = None
    misclassify_mask: np.array = None
    label_name: str = None

    def __init__(self, 
                 cluster_class: int, 
                 cluster_id: int,
                 examples: List[str],
                 cluster_labels: np.array,
                 cluster_logits: np.array = None,
                 misclassify_mask: np.array = None,
                 label_name: str = None):
        self.cluster_class = cluster_class
        self.cluster_id = cluster_id
        self.examples = examples
        self.labels = cluster_labels
        self.logits = cluster_logits
        self.misclassify_mask = misclassify_mask
        self.label_name = label_name
    
    def __str__(self):
        # And a name for the cluster
        cluster_name = f'Cluster {self.cluster_id}'
        if self.cluster_class is not None:
            cluster_name = f'Class {self.cluster_class} ({self.label_name}) {cluster_name}'
        return cluster_name
    
    def __repr__(self):
        # And a name for the cluster
        cluster_name = f'Cluster {self.cluster_id}'
        if self.cluster_class is not None:
            cluster_name = f'Class {self.cluster_class} ({self.label_name}) {cluster_name}'
        return cluster_name


class ClusterSet:
    '''
    This class is used to compute and store the clusters of the data set.
    It takes as input the examples, labels, and representations of the data set.
    It uses AgglomerativeClustering to compute the clusters.
    It stores the cluster labels and the examples in each cluster.
    '''

    def __init__(self,
                 config: Config,
                 examples: List[str],
                 labels: np.array,
                 logits: np.array,
                 label_names: List[str],
                 representations: np.array,
                 distance_threshold: float = None):

        self.config = config
        self.examples = examples
        self.labels = labels
        self.label_names = label_names
        self.logits = logits
        self.representations = representations
        self.distance_threshold = distance_threshold
        self.misclassify_mask = (logits.argmax(axis=1) != labels)

        self.num_labels = len(np.unique(labels))

        self.__name__ = f'ClusterSet_{self.config.cluster_embeddings}_{self.config.clustering_mode}_P{int(self.config.is_pca)}_L{int(self.config.include_logits)}_C{int(self.config.cluster_by_class)}'
        if self.distance_threshold is not None:
            self.__name__ += f'_DT{self.distance_threshold}'
        self.clusters = []
        
    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __len__(self):
        return len(self.clusters)
    
    def __next__(self):
        if self.index < len(self):
            cluster = self.clusters[self.index]
            # Also return the examples that do not belong to the cluster
            non_cluster_examples = [cluster_opp.examples for idx, cluster_opp in enumerate(self.clusters) if idx != self.index and cluster_opp.cluster_class == cluster.cluster_class]
            non_cluster_examples = [example for sublist in non_cluster_examples for example in sublist]
            self.index += 1
            return cluster, non_cluster_examples
        else:
            raise StopIteration
    
    def get_negative_examples(self, cluster: Cluster):
        # get idx of cluster
        cluster_idx = [idx for idx, cluster_opp in enumerate(self.clusters) if str(cluster_opp) == str(cluster)]
        non_cluster_examples = [cluster_opp.examples for idx, cluster_opp in enumerate(self.clusters) if idx not in cluster_idx and cluster_opp.cluster_class == cluster.cluster_class]
        if len(non_cluster_examples) == 0: non_cluster_examples = [cluster_opp.examples for idx, cluster_opp in enumerate(self.clusters) if idx not in cluster_idx]
        non_cluster_examples = [example for sublist in non_cluster_examples for example in sublist]
        return non_cluster_examples

    def _cluster(self,
                 representations: np.array,
                 cluster_class: int = None,
                 examples: List[str] = None,
                 labels: np.array = None,
                 logits: np.array = None,
                 misclassify_mask: np.array = None,
                 label_name: str = None):
        
        if examples is None:
            examples = self.examples
        if labels is None:
            labels = self.labels
        if logits is None:
            logits = self.logits
        if misclassify_mask is None:
            misclassify_mask = self.misclassify_mask
        
        agglomerative = AgglomerativeClustering(n_clusters=None,
                                                distance_threshold=self.distance_threshold).fit(representations)

        cluster_labels = agglomerative.labels_

        # calculate the number of clusters (may not be the same because of the distance threshold)
        num_clusters = len(np.unique(cluster_labels))

        # Analyze the composition of each cluster
        clusters = []
        for cluster in range(num_clusters):
            print(f"Cluster {cluster}: {np.sum(cluster_labels == cluster)}")
            cluster_examples = [example for example, label in zip(examples, cluster_labels) if label == cluster]
            clusters.append(Cluster(cluster_class, cluster, cluster_examples,
                                    labels[cluster_labels == cluster], logits[cluster_labels == cluster],
                                    misclassify_mask[cluster_labels == cluster], label_name))
        
        return clusters

    def cluster(self):

        if self.config.cluster_by_class:

            # Cluster by class
            for class_idx in range(self.num_labels):
                print(f"Class {class_idx}: {np.sum(self.labels == class_idx)}")
                representations = self.representations[self.labels == class_idx]
                examples = [example for example, label in zip(self.examples, self.labels) if label == class_idx]
                self.clusters.extend(self._cluster(representations, class_idx, examples, 
                                                   self.labels[self.labels == class_idx], self.logits[self.labels == class_idx],
                                                   self.misclassify_mask[self.labels == class_idx], self.label_names[class_idx]))
        
        else:
            representations = self.representations
            self.clusters.extend(self._cluster(representations))
    
    def print_misclassification_rates(self):

        for cluster_id, cluster in enumerate(iter(self)):
            cluster_name = cluster[0]
            cluster = self.clusters[cluster_id]
            misclassify_rate = np.mean(cluster.misclassify_mask)
            print(f"{cluster_name} misclassification rate: {misclassify_rate * 100:.2f}% (# examples: {cluster.misclassify_mask.shape[0]})")