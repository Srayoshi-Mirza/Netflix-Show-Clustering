import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class NetflixDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, filepath):
        """Load Netflix dataset"""
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def basic_info(self, df):
        """Display basic information about the dataset"""
        print("=== Dataset Info ===")
        print(f"Shape: {df.shape}")
        print("\n=== Missing Values ===")
        print(df.isnull().sum())
        print("\n=== Data Types ===")
        print(df.dtypes)
        return df.info()
    
    def clean_data(self, df):
        """Clean missing values and inconsistencies"""
        # Create a copy
        df_clean = df.copy()
        
        # Fill missing values
        df_clean['director'] = df_clean['director'].fillna('Unknown Director')
        df_clean['cast'] = df_clean['cast'].fillna('Unknown Cast')
        df_clean['country'] = df_clean['country'].fillna('Unknown Country')
        df_clean['date_added'] = df_clean['date_added'].fillna('Unknown Date')
        df_clean['rating'] = df_clean['rating'].fillna('Unknown Rating')
        
        # Clean duration column
        df_clean = self._clean_duration(df_clean)
        
        # Clean genres (listed_in)
        df_clean['listed_in'] = df_clean['listed_in'].fillna('Unknown Genre')
        
        print(f"Data cleaned. New shape: {df_clean.shape}")
        return df_clean
    
    def _clean_duration(self, df):
        """Clean and standardize duration column"""
        # Separate movies (minutes) and TV shows (seasons)
        df['duration_type'] = df['duration'].apply(
            lambda x: 'movie' if 'min' in str(x) else 'tv_show' if 'Season' in str(x) else 'unknown'
        )
        
        # Extract numeric duration
        df['duration_numeric'] = df['duration'].apply(self._extract_duration_numeric)
        
        return df
    
    def _extract_duration_numeric(self, duration):
        """Extract numeric value from duration"""
        if pd.isna(duration):
            return 0
        # Extract numbers from duration string
        numbers = re.findall(r'\d+', str(duration))
        return int(numbers[0]) if numbers else 0
    
    def create_features(self, df):
        """Create features for clustering"""
        features_df = df.copy()
        
        # 1. Type encoding (Movie = 0, TV Show = 1)
        features_df['type_encoded'] = features_df['type'].apply(
            lambda x: 0 if x == 'Movie' else 1
        )
        
        # 2. Release year normalization
        features_df['release_year_normalized'] = self._normalize_year(features_df['release_year'])
        
        # 3. Duration normalization (separate for movies and TV shows)
        features_df['duration_normalized'] = self._normalize_duration(features_df)
        
        # 4. Rating categories
        features_df['rating_encoded'] = self._encode_ratings(features_df['rating'])
        
        # 5. Genre features (simplified - top genres only)
        genre_features = self._create_genre_features(features_df['listed_in'])
        features_df = pd.concat([features_df, genre_features], axis=1)
        
        return features_df
    
    def _normalize_year(self, years):
        """Normalize release years"""
        # Handle missing years
        years_clean = years.fillna(years.median())
        return (years_clean - years_clean.min()) / (years_clean.max() - years_clean.min())
    
    def _normalize_duration(self, df):
        """Normalize duration separately for movies and TV shows"""
        duration_norm = pd.Series(index=df.index, dtype=float)
        
        # Normalize movies (minutes)
        movies_mask = df['type'] == 'Movie'
        if movies_mask.any():
            movie_durations = df.loc[movies_mask, 'duration_numeric']
            movie_durations_clean = movie_durations.fillna(movie_durations.median())
            duration_norm.loc[movies_mask] = (movie_durations_clean - movie_durations_clean.min()) / (movie_durations_clean.max() - movie_durations_clean.min())
        
        # Normalize TV shows (seasons)
        tv_mask = df['type'] == 'TV Show'
        if tv_mask.any():
            tv_durations = df.loc[tv_mask, 'duration_numeric']
            tv_durations_clean = tv_durations.fillna(tv_durations.median())
            duration_norm.loc[tv_mask] = (tv_durations_clean - tv_durations_clean.min()) / (tv_durations_clean.max() - tv_durations_clean.min())
        
        return duration_norm.fillna(0)
    
    def _encode_ratings(self, ratings):
        """Encode ratings into numerical categories"""
        # Define rating hierarchy
        rating_order = {
            'G': 1, 'TV-Y': 1, 'TV-G': 1,
            'PG': 2, 'TV-Y7': 2, 'TV-Y7-FV': 2,
            'PG-13': 3, 'TV-PG': 3,
            'R': 4, 'TV-14': 4,
            'NC-17': 5, 'TV-MA': 5,
            'NR': 3, 'UR': 3, 'Unknown Rating': 3
        }
        
        encoded = ratings.map(rating_order).fillna(3)  # Default to 3 for unknown
        return (encoded - 1) / 4  # Normalize to 0-1 range
    
    def _create_genre_features(self, genres_column):
        """Create binary features for top genres"""
        # Get top 10 most common genres
        all_genres = []
        for genres in genres_column.dropna():
            all_genres.extend([g.strip() for g in genres.split(',')])
        
        top_genres = pd.Series(all_genres).value_counts().head(10).index.tolist()
        
        # Create binary features
        genre_features = pd.DataFrame(index=genres_column.index)
        
        for genre in top_genres:
            genre_features[f'genre_{genre.lower().replace(" ", "_").replace("-", "_")}'] = genres_column.apply(
                lambda x: 1 if isinstance(x, str) and genre in x else 0
            )
        
        return genre_features
    
    def prepare_clustering_data(self, df):
        """Prepare final dataset for clustering"""
        # Select clustering features
        feature_columns = [
            'type_encoded',
            'release_year_normalized',
            'duration_normalized',
            'rating_encoded'
        ]
        
        # Add genre features
        genre_cols = [col for col in df.columns if col.startswith('genre_')]
        feature_columns.extend(genre_cols)
        
        clustering_data = df[feature_columns].fillna(0)
        
        print(f"Clustering data prepared: {clustering_data.shape}")
        print(f"Features: {feature_columns}")
        
        return clustering_data, feature_columns

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