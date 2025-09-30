# Project Name: Netflix Content Clustering: A Data-Driven Approach to Content Strategy & Recommendation Systems

### Background: 
Netflix hosts over 8,800 titles across movies and TV shows, serving 200+ million subscribers globally. Understanding content organization patterns is crucial for content acquisition strategy, recommendation systems, and competitive positioning. This project leverages unsupervised machine learning to uncover hidden patterns in Netflix's content library.

### Objective:  
Apply K-means clustering to Netflix's catalog to identify distinct content groups, analyze their characteristics, and extract strategic insights for content strategy and recommendation system development.

# Business Problem

### The Challenge
Netflix hosts 8,800+ titles across 190+ countries, serving 200M+ subscribers. Key questions:
- How is content naturally organized beyond traditional genres?
- What content patterns drive user engagement?
- Where are the strategic content gaps?
- How can we improve content recommendations?

### The Solution
Use unsupervised machine learning to:
1. **Discover** hidden patterns in content characteristics
2. **Analyze** cluster compositions and strategies
3. **Identify** content gaps and acquisition opportunities
4. **Inform** recommendation system development

### Business Impact
- **Content Strategy:** Data-driven acquisition decisions worth millions
- **Recommendation Systems:** Better "More Like This" suggestions → higher retention
- **Competitive Intelligence:** Understand Netflix's positioning vs competitors
- **Portfolio Optimization:** Balance content mix across segments

## 📊 Dataset

**Source:** [Kaggle - Netflix Movies and TV Shows](https://www.kaggle.com/datasets/rohitgrewal/netflix-data)


**Features:**
**Size:** 7789 titles × 11 features
-	`Show_Id`  - Unique identifier
-	`Category` - Movie or TV Show
-	`Title` - Content title
-	`Director` - Director name(s)
-	`Cast` - Cast members
-	`Country` - Country of production
-	`Release_Date` - Date added to Netflix
-	`Rating` - Content rating (PG, R, TV-MA, etc.)
-	`Duration` - Length (minutes for movies, seasons for TV shows)
-	`Type` - Genres/categories
-	`Description` - Content synopsis


**Goals:**
1. Implement robust clustering algorithm with silhouette score optimization
2. Discover 3-5 meaningful content segments with distinct characteristics
3. Identify strategic content gaps worth investigating for investment
4. Build foundation for similarity-based recommendation engine
5. Create portfolio-quality project demonstrating end-to-end ML skills

**Expected Outcomes:**
- Clustered Netflix catalog with clear segment definitions
- Visual representations of content organization patterns
- Strategic insights report with business recommendations
- Documentation suitable for technical and non-technical audiences
- Reproducible code for portfolio demonstration

## 📁 Project Structure

```
netflix-show-clustering/
│
├── data/
│   ├── raw/
│   │   └── netflix_titles.csv          # Original dataset from Kaggle
│   └── processed/
│       ├── netflix_processed.csv       # Cleaned dataset
│       ├── clustering_features.csv     # Feature matrix for clustering
│       └── feature_info.csv            # Feature metadata
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA and visualization
│   ├── 02_preprocessing.ipynb          # Data cleaning & feature engineering
│   └── 03_clustering_analysis.ipynb    # Clustering and evaluation
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Data cleaning functions
│   ├── clustering.py                   # Clustering algorithms
│   └── visualization.py                # Plotting functions
│
├── results/
│   ├── netflix_clustered.csv           # Dataset with cluster labels
│   ├── cluster_summary.csv             # Cluster statistics
│   ├── cluster_feature_means.csv       # Average features per cluster
│   ├── pca_coordinates.csv             # 2D visualization data
│   ├── clustering_insights.txt         # Business insights report
│   └── *.png                           # Visualization images
│
├── main.py                             # Single-file execution script
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── LICENSE                             # MIT License

```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (optional, for notebooks)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/netflix-show-clustering.git
cd netflix-show-clustering
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
1. Go to [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/rohitgrewal/netflix-data)
2. Download `netflix_titles.csv`
3. Place in `data/raw/` folder

---

## 💻 Usage

### Option 1: Quick Start (Single Script)
```bash
# Run complete analysis with one command
python main.py
```

**Output:** 
- Clustered dataset saved to `results/`
- Visualizations displayed and saved
- Console output with cluster analysis

### Option 2: Step-by-Step (Jupyter Notebooks)
```bash
# Start Jupyter Notebook
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_preprocessing.ipynb
# 3. notebooks/03_clustering_analysis.ipynb
```

**Benefits:**
- Interactive exploration
- Cell-by-cell execution
- Modify parameters easily
- See intermediate results

### Option 3: Python Modules
```python
from src.data_preprocessing import NetflixDataPreprocessor
from src.clustering import NetflixClustering

# Load and preprocess data
preprocessor = NetflixDataPreprocessor()
df = preprocessor.load_data('data/raw/netflix_titles.csv')
df_clean = preprocessor.clean_data(df)

# Perform clustering
clusterer = NetflixClustering()
results = clusterer.find_optimal_clusters(features)
```

---

## Methodology

### 1. Data Exploration
- Analyze data structure and quality
- Identify missing values patterns
- Explore distributions and relationships
- Visualize content characteristics

### 2. Data Preprocessing
**Cleaning:**
- Handle missing values (director, cast, country)
- Standardize date formats
- Clean duration fields

**Feature Engineering:**
- Binary encoding: Content type (Movie/TV Show)
- Normalization: Release year, content age
- Rating encoding: Hierarchical mapping (G→1, TV-MA→5)
- Genre features: Top 10 genres as binary features
- Country features: Top 5 countries as binary features
- Duration normalization: Separate for movies (minutes) and TV shows (seasons)

**Final Features (14 total):**
- 4 basic features (categories, year, rating, duration)
- 10 genre features (dramas, comedies, etc.)

### 3. Clustering Algorithm

**K-Means Clustering:**
- Algorithm: K-means with k-means++ initialization
- Distance metric: Euclidean distance
- K range tested: 2-10 clusters

**Optimization:**
- Elbow method for inertia analysis
- Silhouette score maximization (primary metric)
- Calinski-Harabasz score
- Davies-Bouldin score

### 4. Dimensionality Reduction
- **PCA** for 2D visualization
- Explained variance analysis
- Feature importance via loadings

### 5. Evaluation Metrics
- **Silhouette Score** (0.25-0.35 achieved)
- **Calinski-Harabasz Index**
- **Davies-Bouldin Index**
- **Visual Inspection** of cluster separation

### Visualizations Generated
1. **Cluster Evaluation Metrics** (elbow, silhouette curves)
2. **2D PCA Scatter Plot** with cluster colors
3. **Cluster Size Distribution** (pie chart)
4. **Feature Heatmap** by cluster
5. **Content Type Distribution** by cluster
6. **Release Year Box Plots** by cluster
7. **Rating Distribution** by cluster


## 🛠️ Technologies Used

**Programming Language:**
- Python 3.8+

**Core Libraries:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Data visualization
- `seaborn` - Statistical visualizations

**Machine Learning:**
- K-means clustering
- PCA (Principal Component Analysis)
- Silhouette analysis
- Feature engineering and scaling

**Development Tools:**
- Jupyter Notebook - Interactive development
- Git - Version control
- VS Code - Code editor

---

## 🔮 Future Work

### Technical Improvements
- [ ] Try DBSCAN and Hierarchical Clustering
- [ ] Add text analysis of descriptions (TF-IDF, NLP)
- [ ] Include cast/director features
- [ ] Time-series analysis of content additions
- [ ] Deep learning embeddings for content similarity

### Business Extensions
- [ ] User behavior clustering
- [ ] Regional content analysis by country
- [ ] Competitive benchmarking (Netflix vs Disney+ vs Prime)
- [ ] Recommendation system prototype
- [ ] A/B testing framework for recommendations

### Advanced Analysis
- [ ] Sentiment analysis of descriptions
- [ ] Network analysis of cast/director connections
- [ ] Temporal cluster evolution
- [ ] Predictive modeling for content success

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

**Areas for Contribution:**
- Additional clustering algorithms
- Enhanced visualizations
- Feature engineering ideas
- Documentation improvements
- Bug fixes and optimizations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com
- Portfolio: [yourwebsite.com](https://yourwebsite.com)

**Project Link:** [https://github.com/yourusername/netflix-show-clustering](https://github.com/yourusername/netflix-show-clustering)

---

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the Netflix dataset
- [Netflix](https://www.netflix.com/) for the inspiration
- [scikit-learn](https://scikit-learn.org/) community for excellent ML tools
- Open source community for various libraries used

---

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ and Python**

</div>