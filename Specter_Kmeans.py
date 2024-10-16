import os
import re
import warnings
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer, models
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Suppress FutureWarning from sklearn
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Set environment variables for GPU usage
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class TextSummarizer:
    def __init__(self, model_name='sentence-transformers/allenai-specter'):
        # Load the specter model and apply mean pooling
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        # Create the SentenceTransformer model with the pooling layer
        self.embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    @staticmethod
    def remove_noise(text):
        """
        Clean the text by removing URLs, converting to lowercase, and handling common patterns.

        Args:
            text (str): Input text to clean.

        Returns:
            str: Cleaned text.
        """
        url_pattern = re.compile(r"http[s]?://\S+|www\.\S+")
        text = text.lower()
        text = text.replace("./.", ". ")
        text = text.replace("tp.", "tp ")
        cleaned_text = re.sub(url_pattern, " ", text)
        return cleaned_text

    @staticmethod
    def find_optimal_cluster(corpus_embeddings, cluster_range):
        """
        Find the optimal number of clusters based on the Silhouette Score.

        Args:
            corpus_embeddings (array-like): Input data matrix of shape (n_samples, n_features).
            cluster_range (range or list): Range of cluster numbers to evaluate.

        Returns:
            int: Optimal number of clusters that maximizes the Silhouette Score.
        """
        best_score = -1
        optimal_cluster = None

        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(corpus_embeddings)
            score = silhouette_score(corpus_embeddings, cluster_labels)

            if score > best_score:
                best_score = score
                optimal_cluster = n_clusters

        return optimal_cluster

    def summarize(self, content):
        """
        Summarizes the input text by extracting representative sentences using sentence embeddings and KMeans clustering.

        Args:
            content (str): Input text to summarize.

        Returns:
            list: A list of sentences that represent the summary.
        """
        # Tokenize content into sentences
        corpus = nltk.sent_tokenize(content)
        # Encode sentences to obtain embeddings
        corpus_embeddings = self.embedder.encode(corpus)

        # Determine the optimal number of clusters
        values = range(2, len(corpus_embeddings))
        num_clusters = self.find_optimal_cluster(corpus_embeddings, values)

        # Perform KMeans clustering
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)

        # Select representative sentences from each cluster
        centroid_sentences = []
        for i in range(num_clusters):
            cluster_indices = np.where(clustering_model.labels_ == i)[0]
            cluster_embeddings = corpus_embeddings[cluster_indices]
            centroid_embedding = np.mean(cluster_embeddings, axis=0)
            # Select the sentence closest to the centroid
            centroid_sentence = corpus[np.argmin(np.linalg.norm(cluster_embeddings - centroid_embedding, axis=1))]
            centroid_sentences.append(centroid_sentence)

        return centroid_sentences