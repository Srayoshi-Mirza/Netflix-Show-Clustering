import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Simple Netflix clustering analysis
    """
    print("Netflix Show Clustering Analysis")
    print("=" * 40)
    
    # Step 1: Load data
    print("1. Loading data...")
    try:
        df = pd.read_csv('data/raw/netflix_titles.csv')
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("❌ Please place netflix_titles.csv in data/raw/ folder")
        return
    
    # Step 2: Quick data overview
    print("\n2. Data overview...")
    print(f"Missing values:\n{df.isnull().sum().head()}")
    print(f"Content types: {df['type'].value_counts().to_dict()}")
    
    # Step 3: Simple preprocessing
    print("\n3. Preprocessing data...")
    df_clean = df.copy()
    
    # Handle missing values
    df_clean['director'] = df_clean['director'].fillna('Unknown')
    df_clean['cast'] = df_clean['cast'].fillna('Unknown')
    df_clean['country'] = df_clean['country'].fillna('Unknown')
    df_clean['rating'] = df_clean['rating'].fillna('Unknown')
    
    # Create simple features
    features = pd.DataFrame()
    
    # Feature 1: Type (Movie=0, TV Show=1)
    features['is_tv_show'] = (df_clean['type'] == 'TV Show').astype(int)
    
    # Feature 2: Release year (normalized)
    release_years = df_clean['release_year'].fillna(df_clean['release_year'].median())
    features['release_year_norm'] = (release_years - release_years.min()) / (release_years.max() - release_years.min())
    
    # Feature 3: Rating categories (simplified)
    rating_map = {'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 'TV-Y': 1, 'TV-Y7': 2, 'TV-PG': 2, 'TV-14': 3, 'TV-MA': 4}
    features['rating_level'] = df_clean['rating'].map(rating_map).fillna(2.5) / 4  # Normalize
    
    # Feature 4: Top genres (binary features)
    all_genres = df_clean['listed_in'].dropna().str.split(',').explode().str.strip()
    top_genres = all_genres.value_counts().head(5).index
    
    for genre in top_genres:
        features[f'has_{genre.lower().replace(" ", "_").replace("-", "_")}'] = df_clean['listed_in'].fillna('').str.contains(genre, case=False).astype(int)
    
    print(f"✓ Created {features.shape[1]} features for clustering")
    print(f"Features: {list(features.columns)}")
    
    # Step 4: Find optimal clusters
    print("\n4. Finding optimal number of clusters...")
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(features, labels)
        silhouette_scores.append(sil_score)
        
        print(f"k={k}: Silhouette Score = {sil_score:.3f}")
    
    # Choose best k
    best_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"✓ Best k = {best_k}")
    
    # Step 5: Final clustering
    print(f"\n5. Final clustering with k={best_k}...")
    
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(features)
    
    df_clean['cluster'] = final_labels
    
    # Step 6: Analyze clusters
    print("\n6. Cluster Analysis:")
    print("-" * 30)
    
    for cluster_id in range(best_k):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data)} shows, {len(cluster_data)/len(df_clean)*100:.1f}%):")
        
        # Most common type
        top_type = cluster_data['type'].mode().iloc[0]
        print(f"   Main type: {top_type} ({(cluster_data['type']==top_type).sum()}/{len(cluster_data)})")
        
        # Average release year
        avg_year = cluster_data['release_year'].mean()
        print(f"  Avg release year: {avg_year:.0f}")
        
        # Most common rating
        top_rating = cluster_data['rating'].mode().iloc[0]
        print(f"  Top rating: {top_rating}")
        
        # Top genres
        cluster_genres = cluster_data['listed_in'].dropna().str.split(',').explode().str.strip()
        top_cluster_genres = cluster_genres.value_counts().head(3)
        print(f"  Top genres: {', '.join(top_cluster_genres.index.tolist())}")
        
        # Sample titles
        sample_titles = cluster_data['title'].head(3).tolist()
        print(f"  Sample titles: {', '.join(sample_titles)}")
    
    # Step 7: Visualization
    print("\n7. Creating visualizations...")
    
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)
    
    # Plot clusters
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Cluster scatter plot
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=final_labels, cmap='tab10', alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Netflix Shows Clustering')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Cluster sizes
    plt.subplot(1, 2, 2)
    cluster_sizes = pd.Series(final_labels).value_counts().sort_index()
    plt.pie(cluster_sizes.values, labels=[f'Cluster {i}' for i in cluster_sizes.index], autopct='%1.1f%%')
    plt.title('Cluster Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Step 8: Save results
    print("\n8. Saving results...")
    
    # Create results directory if it doesn't exist
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save clustered data
    df_clean.to_csv('results/netflix_clustered.csv', index=False)
    
    # Save cluster summary
    summary = []
    for cluster_id in range(best_k):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id]
        summary.append({
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data)/len(df_clean)*100,
            'main_type': cluster_data['type'].mode().iloc[0],
            'avg_release_year': cluster_data['release_year'].mean(),
            'top_rating': cluster_data['rating'].mode().iloc[0]
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('results/cluster_summary.csv', index=False)
    
    print("✓ Results saved to results/ folder")
    print("\n Netflix clustering analysis completed!")
    print(f" Found {best_k} distinct clusters in Netflix content")
    
    return df_clean, final_labels, features

# This is the important part that was missing!
if __name__ == "__main__":
    try:
        df_result, labels, features = main()
        print("\n Analysis complete! Check the results folder for output files.")
    except Exception as e:
        print(f"\n Error occurred: {e}")
        print("Make sure you have:")
        print("1. Downloaded netflix_titles.csv to data/raw/")
        print("2. Installed required packages: pip install pandas numpy scikit-learn matplotlib seaborn")
        print("3. Created the necessary folders: data/raw/ and results/")