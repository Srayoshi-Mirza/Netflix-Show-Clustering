from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class NetflixClustering:
    def __init__(self):
        self.kmeans_models = {}
        self.best_k = None
        self.best_model = None
        self.pca = None
        
    def find_optimal_clusters(self, data, k_range=range(2, 11)):
        """Find optimal number of clusters using elbow method and silhouette score"""
        inertias = []
        silhouette_scores = []
        k_values = list(k_range)
        
        print("Finding optimal number of clusters...")
        
        for k in k_values:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(data, cluster_labels)
            silhouette_scores.append(sil_score)
            
            # Store model
            self.kmeans_models[k] = kmeans
            
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
        
        # Find best k based on silhouette score
        best_k_idx = np.argmax(silhouette_scores)
        self.best_k = k_values[best_k_idx]
        self.best_model = self.kmeans_models[self.best_k]
        
        print(f"\nBest k = {self.best_k} with silhouette score = {silhouette_scores[best_k_idx]:.3f}")
        
        return {
            'k_values': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'best_k': self.best_k
        }
    
    def plot_elbow_curve(self, results):
        """Plot elbow curve and silhouette scores"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Elbow curve
        ax1.plot(results['k_values'], results['inertias'], 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(results['k_values'], results['silhouette_scores'], 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=results['best_k'], color='red', linestyle='--', alpha=0.7, label=f'Best k={results["best_k"]}')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def fit_final_model(self, data):
        """Fit the final clustering model with optimal k"""
        if self.best_k is None:
            raise ValueError("Please run find_optimal_clusters first!")
        
        print(f"Fitting final model with k={self.best_k}")
        
        # Get the best model
        final_labels = self.best_model.fit_predict(data)
        
        # Calculate final metrics
        silhouette_avg = silhouette_score(data, final_labels)
        calinski_score = calinski_harabasz_score(data, final_labels)
        
        print(f"Final Silhouette Score: {silhouette_avg:.3f}")
        print(f"Final Calinski-Harabasz Score: {calinski_score:.2f}")
        
        return final_labels
    
    def analyze_clusters(self, data, labels, feature_names, original_data=None):
        """Analyze characteristics of each cluster"""
        results = {}
        
        print("\n=== CLUSTER ANALYSIS ===")
        
        for cluster_id in sorted(np.unique(labels)):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {len(cluster_data)} ({len(cluster_data)/len(data)*100:.1f}%)")
            
            # Feature analysis
            cluster_stats = {}
            for i, feature in enumerate(feature_names):
                feature_values = cluster_data.iloc[:, i]
                cluster_stats[feature] = {
                    'mean': feature_values.mean(),
                    'std': feature_values.std()
                }
                print(f"  {feature}: mean={feature_values.mean():.3f}, std={feature_values.std():.3f}")
            
            results[cluster_id] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data)/len(data)*100,
                'stats': cluster_stats
            }
            
            # If original data is provided, show some examples
            if original_data is not None:
                print(f"  Sample titles:")
                sample_titles = original_data.loc[cluster_mask, 'title'].head(3).tolist()
                for title in sample_titles:
                    print(f"    - {title}")
        
        return results
    
    def reduce_dimensions_for_viz(self, data):
        """Reduce dimensions using PCA for visualization"""
        self.pca = PCA(n_components=2, random_state=42)
        data_2d = self.pca.fit_transform(data)
        
        print(f"PCA Explained Variance Ratio: {self.pca.explained_variance_ratio_}")
        print(f"Total Explained Variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return data_2d
    
    def plot_clusters_2d(self, data_2d, labels, title="Netflix Shows Clustering"):
        """Plot clusters in 2D space"""
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
        
        # Add cluster centers if available
        if self.best_model is not None:
            centers_2d = self.pca.transform(self.best_model.cluster_centers_)
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.xlabel(f'PCA Component 1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PCA Component 2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(title)
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return plt.gcf()
    
    def create_cluster_summary(self, original_data, labels):
        """Create a summary of cluster characteristics"""
        df_with_clusters = original_data.copy()
        df_with_clusters['cluster'] = labels
        
        summary = {}
        
        for cluster_id in sorted(np.unique(labels)):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            summary[f'Cluster {cluster_id}'] = {
                'count': len(cluster_data),
                'percentage': f"{len(cluster_data)/len(df_with_clusters)*100:.1f}%",
                'top_type': cluster_data['type'].mode().iloc[0] if len(cluster_data) > 0 else 'Unknown',
                'avg_release_year': f"{cluster_data['release_year'].mean():.0f}" if cluster_data['release_year'].notna().any() else 'Unknown',
                'top_rating': cluster_data['rating'].mode().iloc[0] if len(cluster_data) > 0 else 'Unknown',
                'top_genres': cluster_data['listed_in'].str.split(',').explode().str.strip().mode().head(3).tolist()
            }
        
        return summary

# Import or define NetflixDataPreprocessor before usage
from data_preprocessing import NetflixDataPreprocessor  # Adjust the import path as needed

# Usage example:
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = NetflixDataPreprocessor()
    
    # Load and process data
    df = preprocessor.load_data('data/raw/Netflix_Dataset.csv')
    preprocessor.basic_info(df)
    
    # Clean and prepare data
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_features(df_clean)
    clustering_data, feature_names = preprocessor.prepare_clustering_data(df_features)
    
    # Save processed data
    df_features.to_csv('data/processed/netflix_processed.csv', index=False)
    clustering_data.to_csv('data/processed/clustering_features.csv', index=False)
    
    print("Data preprocessing completed!")
    
    # Clustering
    print("\n" + "="*50)
    print("STARTING CLUSTERING ANALYSIS")
    print("="*50)
    
    clusterer = NetflixClustering()
    
    # Find optimal clusters
    results = clusterer.find_optimal_clusters(clustering_data)
    clusterer.plot_elbow_curve(results)
    
    # Fit final model
    final_labels = clusterer.fit_final_model(clustering_data)
    
    # Analyze clusters
    cluster_analysis = clusterer.analyze_clusters(clustering_data, final_labels, feature_names, df_clean)
    
    # Visualization
    data_2d = clusterer.reduce_dimensions_for_viz(clustering_data)
    clusterer.plot_clusters_2d(data_2d, final_labels)
    
    # Create summary
    summary = clusterer.create_cluster_summary(df_clean, final_labels)
    print("\n=== CLUSTER SUMMARY ===")
    for cluster, info in summary.items():
        print(f"{cluster}: {info['count']} shows ({info['percentage']})")
        print(f"  Main type: {info['top_type']}")
        print(f"  Avg year: {info['avg_release_year']}")
        print(f"  Top rating: {info['top_rating']}")
        print(f"  Top genres: {', '.join(info['top_genres'])}")
        print()
    
    print("Clustering analysis completed!")