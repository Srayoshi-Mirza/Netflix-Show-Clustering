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
    
    # Step 2: Check and display actual columns
    print("\n2. Dataset columns:")
    print(f"Actual columns: {list(df.columns)}")
    
    # Map your columns to expected names
    column_mapping = {
        'Show_Id': 'show_id',
        'Category': 'type',
        'Title': 'title',
        'Director': 'director',
        'Cast': 'cast',
        'Country': 'country',
        'Release_Date': 'date_added',
        'Rating': 'rating',
        'Duration': 'duration',
        'Type': 'listed_in',
        'Description': 'description'
    }
    
    # Rename columns if needed
    df = df.rename(columns=column_mapping)
    
    # Check if 'Release_Year' exists or needs to be created
    if 'Release_Year' in df.columns:
        df = df.rename(columns={'Release_Year': 'release_year'})
    elif 'date_added' in df.columns:
        # Try to extract year from date_added if release_year doesn't exist
        df['release_year'] = pd.to_datetime(df['date_added'], errors='coerce').dt.year
    else:
        # Use a default year if nothing is available
        df['release_year'] = 2020
    
    print(f"✓ Columns standardized")
    print(f"Working columns: {list(df.columns)}")
    
    # Step 3: Quick data overview
    print("\n3. Data overview...")
    print(f"Missing values:\n{df.isnull().sum().head()}")
    
    if 'type' in df.columns:
        print(f"Content types: {df['type'].value_counts().to_dict()}")
    else:
        print("⚠️ 'type' column not found")
    
    # Step 4: Simple preprocessing
    print("\n4. Preprocessing data...")
    df_clean = df.copy()
    
    # Handle missing values
    df_clean['director'] = df_clean['director'].fillna('Unknown')
    df_clean['cast'] = df_clean['cast'].fillna('Unknown')
    df_clean['country'] = df_clean['country'].fillna('Unknown') if 'country' in df_clean.columns else 'Unknown'
    df_clean['rating'] = df_clean['rating'].fillna('Unknown') if 'rating' in df_clean.columns else 'Unknown'
    df_clean['listed_in'] = df_clean['listed_in'].fillna('Unknown') if 'listed_in' in df_clean.columns else 'Unknown'
    
    # Create simple features
    features = pd.DataFrame()
    
    # Feature 1: Type (Movie=0, TV Show=1)
    if 'type' in df_clean.columns:
        # Handle different possible values
        df_clean['type'] = df_clean['type'].str.strip()
        features['is_tv_show'] = df_clean['type'].str.contains('TV|Series', case=False, na=False).astype(int)
    else:
        features['is_tv_show'] = 0  # Default to movies if type not available
    
    # Feature 2: Release year (normalized)
    if 'release_year' in df_clean.columns:
        release_years = df_clean['release_year'].fillna(df_clean['release_year'].median())
        release_years = pd.to_numeric(release_years, errors='coerce').fillna(2020)
        features['release_year_norm'] = (release_years - release_years.min()) / (release_years.max() - release_years.min() + 1)
    else:
        features['release_year_norm'] = 0.5  # Default middle value
    
    # Feature 3: Rating categories (simplified)
    rating_map = {
        'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 
        'TV-Y': 1, 'TV-Y7': 2, 'TV-PG': 2, 'TV-14': 3, 'TV-MA': 4,
        'NR': 3, 'UR': 3, 'Not Rated': 3, 'Unknown': 3
    }
    
    if 'rating' in df_clean.columns:
        features['rating_level'] = df_clean['rating'].map(rating_map).fillna(2.5) / 4
    else:
        features['rating_level'] = 0.5  # Default middle value
    
    # Feature 4: Top genres (binary features)
    if 'listed_in' in df_clean.columns:
        all_genres = df_clean['listed_in'].dropna().str.split(',').explode().str.strip()
        top_genres = all_genres.value_counts().head(5).index
        
        for genre in top_genres:
            genre_clean = genre.lower().replace(" ", "_").replace("-", "_").replace("&", "and")
            features[f'has_{genre_clean}'] = df_clean['listed_in'].fillna('').str.contains(genre, case=False).astype(int)
    
    print(f"✓ Created {features.shape[1]} features for clustering")
    print(f"Features: {list(features.columns)}")
    
    # Step 5: Find optimal clusters
    print("\n5. Finding optimal number of clusters...")
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, min(8, len(features)//10 + 2))  # Adjust k_range based on dataset size
    
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
    
    # Step 6: Final clustering
    print(f"\n6. Final clustering with k={best_k}...")
    
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(features)
    
    df_clean['cluster'] = final_labels
    
    # Step 7: Analyze clusters
    print("\n7. Cluster Analysis:")
    print("-" * 30)
    
    for cluster_id in range(best_k):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data)} shows, {len(cluster_data)/len(df_clean)*100:.1f}%):")
        
        # Most common type
        if 'type' in cluster_data.columns and not cluster_data['type'].isna().all():
            top_type = cluster_data['type'].mode()
            if len(top_type) > 0:
                top_type_val = top_type.iloc[0]
                print(f"  📺 Main type: {top_type_val} ({(cluster_data['type']==top_type_val).sum()}/{len(cluster_data)})")
        
        # Average release year
        if 'release_year' in cluster_data.columns:
            avg_year = pd.to_numeric(cluster_data['release_year'], errors='coerce').mean()
            if not np.isnan(avg_year):
                print(f"  📅 Avg release year: {avg_year:.0f}")
        
        # Most common rating
        if 'rating' in cluster_data.columns and not cluster_data['rating'].isna().all():
            top_rating = cluster_data['rating'].mode()
            if len(top_rating) > 0:
                print(f"  🏷️  Top rating: {top_rating.iloc[0]}")
        
        # Top genres
        if 'listed_in' in cluster_data.columns:
            cluster_genres = cluster_data['listed_in'].dropna().str.split(',').explode().str.strip()
            if len(cluster_genres) > 0:
                top_cluster_genres = cluster_genres.value_counts().head(3)
                print(f"  🎭 Top genres: {', '.join(top_cluster_genres.index.tolist())}")
        
        # Sample titles
        if 'title' in cluster_data.columns:
            sample_titles = cluster_data['title'].head(3).tolist()
            print(f"  📋 Sample titles: {', '.join(sample_titles)}")
    
    # Step 8: Visualization
    print("\n8. Creating visualizations...")
    
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
    
    # Step 9: Save results
    print("\n9. Saving results...")
    
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
        summary_dict = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data)/len(df_clean)*100
        }
        
        if 'type' in cluster_data.columns:
            top_type = cluster_data['type'].mode()
            summary_dict['main_type'] = top_type.iloc[0] if len(top_type) > 0 else 'Unknown'
        
        if 'release_year' in cluster_data.columns:
            avg_year = pd.to_numeric(cluster_data['release_year'], errors='coerce').mean()
            summary_dict['avg_release_year'] = avg_year if not np.isnan(avg_year) else 'Unknown'
        
        if 'rating' in cluster_data.columns:
            top_rating = cluster_data['rating'].mode()
            summary_dict['top_rating'] = top_rating.iloc[0] if len(top_rating) > 0 else 'Unknown'
        
        summary.append(summary_dict)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('results/cluster_summary.csv', index=False)
    
    print("✓ Results saved to results/ folder")
    print("\n🎉 Netflix clustering analysis completed!")
    print(f"📊 Found {best_k} distinct clusters in Netflix content")
    
    return df_clean, final_labels, features

if __name__ == "__main__":
    try:
        df_result, labels, features = main()
        print("\n✨ Analysis complete! Check the results folder for output files.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")