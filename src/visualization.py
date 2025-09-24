import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.decomposition import PCA

class NetflixVisualizer:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_data_overview(self, df):
        """Create overview plots of the dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Content type distribution
        df['type'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Content Type Distribution')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # Release year distribution
        df['release_year'].hist(bins=30, ax=axes[0,1])
        axes[0,1].set_title('Release Year Distribution')
        axes[0,1].set_xlabel('Year')
        
        # Top 10 countries
        top_countries = df['country'].str.split(',').explode().str.strip().value_counts().head(10)
        top_countries.plot(kind='barh', ax=axes[1,0])
        axes[1,0].set_title('Top 10 Countries')
        
        # Rating distribution
        df['rating'].value_counts().head(10).plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Rating Distribution')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def plot_cluster_characteristics(self, df_with_clusters):
        """Plot characteristics of each cluster"""
        n_clusters = df_with_clusters['cluster'].nunique()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cluster sizes
        cluster_sizes = df_with_clusters['cluster'].value_counts().sort_index()
        cluster_sizes.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Cluster Sizes')
        axes[0,0].set_xlabel('Cluster')
        axes[0,0].set_ylabel('Number of Shows')
        
        # Type distribution by cluster
        type_by_cluster = pd.crosstab(df_with_clusters['cluster'], df_with_clusters['type'])
        type_by_cluster.plot(kind='bar', stacked=True, ax=axes[0,1])
        axes[0,1].set_title('Content Type by Cluster')
        axes[0,1].legend(title='Type')
        
        # Release year by cluster
        df_with_clusters.boxplot(column='release_year', by='cluster', ax=axes[1,0])
        axes[1,0].set_title('Release Year Distribution by Cluster')
        axes[1,0].set_xlabel('Cluster')
        
        # Rating distribution by cluster
        # Create a simple rating score for visualization
        rating_map = {'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 'TV-Y': 1, 'TV-Y7': 2, 'TV-PG': 2, 'TV-14': 3, 'TV-MA': 4}
        df_viz = df_with_clusters.copy()
        df_viz['rating_score'] = df_viz['rating'].map(rating_map)
        df_viz.boxplot(column='rating_score', by='cluster', ax=axes[1,1])
        axes[1,1].set_title('Rating Level by Cluster')
        axes[1,1].set_xlabel('Cluster')
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def create_genre_wordcloud(self, df, cluster_id=None):
        """Create word cloud for genres"""
        if cluster_id is not None:
            cluster_data = df[df['cluster'] == cluster_id]
            title = f'Cluster {cluster_id} Genres'
        else:
            cluster_data = df
            title = 'All Netflix Genres'
        
        # Extract all genres
        all_genres = cluster_data['listed_in'].dropna().str.split(',').explode().str.strip()
        genre_text = ' '.join(all_genres)
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(genre_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()
        return wordcloud
