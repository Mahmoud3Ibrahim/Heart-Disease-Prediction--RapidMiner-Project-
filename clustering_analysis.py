"""
K-Means clustering analysis for heart disease patient segmentation.
Implements cluster optimization and outlier detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from preprocessing import HeartDiseasePreprocessor


class HeartDiseaseClusteringAnalysis:
    """K-Means clustering analysis for patient segmentation."""

    def __init__(self, random_state=2001):
        self.random_state = random_state
        self.preprocessor = HeartDiseasePreprocessor()
        self.best_k = None
        self.kmeans_model = None
        self.cluster_results = None

    def find_optimal_clusters(self, X, k_range=(2, 20)):
        """
        Find optimal number of clusters using elbow method and silhouette score.

        Args:
            X: Feature matrix
            k_range: Tuple of (min_k, max_k) for testing

        Returns:
            Dictionary with evaluation metrics for each k
        """
        print(f"\nTesting K-Means clustering for k={k_range[0]} to {k_range[1]}...")

        results = {
            'k': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': []
        }

        for k in range(k_range[0], k_range[1] + 1):
            # Fit K-Means
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            # Calculate metrics
            results['k'].append(k)
            results['inertia'].append(kmeans.inertia_)
            results['silhouette'].append(silhouette_score(X, labels))
            results['davies_bouldin'].append(davies_bouldin_score(X, labels))

            if k % 5 == 0:
                print(f"k={k}: Inertia={kmeans.inertia_:.2f}, "
                      f"Silhouette={results['silhouette'][-1]:.3f}")

        return pd.DataFrame(results)

    def select_best_k(self, metrics_df):
        """
        Select best k based on silhouette score.

        Args:
            metrics_df: DataFrame with clustering metrics

        Returns:
            Optimal k value
        """
        # Use silhouette score - higher is better
        best_idx = metrics_df['silhouette'].idxmax()
        best_k = metrics_df.loc[best_idx, 'k']

        print(f"\nBest k selected: {best_k}")
        print(f"Silhouette score: {metrics_df.loc[best_idx, 'silhouette']:.3f}")

        return int(best_k)

    def fit_kmeans(self, X, n_clusters=None):
        """
        Fit K-Means clustering model.

        Args:
            X: Feature matrix
            n_clusters: Number of clusters (if None, uses best_k)
        """
        if n_clusters is None:
            if self.best_k is None:
                raise ValueError("Must set n_clusters or run find_optimal_clusters first")
            n_clusters = self.best_k

        print(f"\nFitting K-Means with {n_clusters} clusters...")
        self.kmeans_model = KMeans(n_clusters=n_clusters,
                                   random_state=self.random_state,
                                   n_init=10)
        labels = self.kmeans_model.fit_predict(X)

        return labels

    def analyze_clusters(self, df, labels):
        """
        Analyze cluster characteristics and detect outliers.

        Args:
            df: Original dataframe
            labels: Cluster labels

        Returns:
            DataFrame with cluster statistics
        """
        df_clustered = df.copy()
        df_clustered['Cluster'] = labels

        # Calculate cluster statistics
        cluster_stats = []

        for cluster_id in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            n_patients = len(cluster_data)

            # Check if outlier (small cluster)
            is_outlier = n_patients <= 30

            stats = {
                'Cluster_ID': cluster_id,
                'Patient_Count': n_patients,
                'Is_Outlier': is_outlier,
                'Avg_Age': cluster_data['Age'].mean(),
                'Avg_Cholesterol': cluster_data['Cholesterol'].mean(),
                'Avg_MaxHR': cluster_data['MaxHR'].mean(),
                'HeartDisease_Rate': cluster_data['HeartDisease'].mean()
            }

            cluster_stats.append(stats)

        cluster_stats_df = pd.DataFrame(cluster_stats)

        # Print summary
        print(f"\nCluster Analysis Summary:")
        print(f"Total clusters: {len(cluster_stats_df)}")
        print(f"Outlier clusters (<=30 patients): {cluster_stats_df['Is_Outlier'].sum()}")
        print(f"\nCluster distribution:")
        print(cluster_stats_df[['Cluster_ID', 'Patient_Count', 'Is_Outlier', 'HeartDisease_Rate']])

        self.cluster_results = cluster_stats_df
        return cluster_stats_df

    def visualize_clusters(self, metrics_df, cluster_stats_df, output_dir='results'):
        """
        Create visualizations for clustering analysis.

        Args:
            metrics_df: Clustering metrics dataframe
            cluster_stats_df: Cluster statistics dataframe
            output_dir: Directory to save plots
        """
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100

        # 1. Elbow curve and silhouette score
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Elbow curve
        axes[0].plot(metrics_df['k'], metrics_df['inertia'], marker='o', linewidth=2)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11)
        axes[0].set_title('Elbow Method for Optimal k', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Silhouette score
        axes[1].plot(metrics_df['k'], metrics_df['silhouette'], marker='o',
                    color='green', linewidth=2)
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[1].set_ylabel('Silhouette Score', fontsize=11)
        axes[1].set_title('Silhouette Score by k', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/clustering_optimization.png', bbox_inches='tight')
        print(f"Saved: {output_dir}/clustering_optimization.png")
        plt.close()

        # 2. Cluster distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Patient count by cluster
        colors = ['red' if x else 'steelblue' for x in cluster_stats_df['Is_Outlier']]
        axes[0].bar(cluster_stats_df['Cluster_ID'], cluster_stats_df['Patient_Count'],
                   color=colors, alpha=0.7)
        axes[0].axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Outlier Threshold (30)')
        axes[0].set_xlabel('Cluster ID', fontsize=11)
        axes[0].set_ylabel('Number of Patients', fontsize=11)
        axes[0].set_title('Patient Distribution Across Clusters', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Heart disease rate by cluster
        axes[1].bar(cluster_stats_df['Cluster_ID'], cluster_stats_df['HeartDisease_Rate'],
                   color='coral', alpha=0.7)
        axes[1].set_xlabel('Cluster ID', fontsize=11)
        axes[1].set_ylabel('Heart Disease Rate', fontsize=11)
        axes[1].set_title('Heart Disease Rate by Cluster', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/cluster_distribution.png', bbox_inches='tight')
        print(f"Saved: {output_dir}/cluster_distribution.png")
        plt.close()

    def run_analysis(self, data_path, k_range=(2, 20), output_dir='results'):
        """
        Run complete clustering analysis pipeline.

        Args:
            data_path: Path to heart disease CSV file
            k_range: Range of k values to test
            output_dir: Directory to save results

        Returns:
            Tuple of (cluster_stats_df, metrics_df)
        """
        print("="*70)
        print("HEART DISEASE CLUSTERING ANALYSIS")
        print("="*70)

        # Load and preprocess data
        print("\n[1/5] Loading and preprocessing data...")
        df = self.preprocessor.load_data(data_path)
        df_processed = self.preprocessor.prepare_for_clustering(df)

        # Prepare feature matrix (exclude target variable)
        feature_cols = [col for col in df_processed.columns if col != 'HeartDisease']
        X = df_processed[feature_cols].values

        # Find optimal clusters
        print(f"\n[2/5] Finding optimal number of clusters...")
        metrics_df = self.find_optimal_clusters(X, k_range)

        # Select best k
        print(f"\n[3/5] Selecting best k...")
        self.best_k = self.select_best_k(metrics_df)

        # Fit final model
        print(f"\n[4/5] Fitting final K-Means model...")
        labels = self.fit_kmeans(X, n_clusters=self.best_k)

        # Analyze clusters
        print(f"\n[5/5] Analyzing cluster characteristics...")
        cluster_stats_df = self.analyze_clusters(df, labels)

        # Save results
        metrics_df.to_csv(f'{output_dir}/clustering_metrics.csv', index=False)
        cluster_stats_df.to_csv(f'{output_dir}/cluster_statistics.csv', index=False)
        print(f"\nResults saved to {output_dir}/")

        # Visualize
        self.visualize_clusters(metrics_df, cluster_stats_df, output_dir)

        print("\n" + "="*70)
        print("CLUSTERING ANALYSIS COMPLETED")
        print("="*70)

        return cluster_stats_df, metrics_df


def main():
    """Run clustering analysis."""
    analyzer = HeartDiseaseClusteringAnalysis(random_state=2001)
    cluster_stats, metrics = analyzer.run_analysis('data/heart.csv', k_range=(2, 15))


if __name__ == '__main__':
    main()
