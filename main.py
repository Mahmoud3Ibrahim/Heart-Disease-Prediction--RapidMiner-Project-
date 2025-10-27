"""
Main script to run heart disease prediction analysis.
Executes both clustering and classification analyses.
"""

import sys
import os
from clustering_analysis import HeartDiseaseClusteringAnalysis
from classification_analysis import HeartDiseaseClassification


def main():
    """Run complete analysis pipeline."""

    # Configuration
    data_path = 'data/heart.csv'
    output_dir = 'results'
    random_state = 2001

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("\n")
    print("="*70)
    print(" "*15 + "HEART DISEASE PREDICTION ANALYSIS")
    print("="*70)
    print("\nThis script performs two types of analysis:")
    print("  1. K-Means Clustering - Patient segmentation")
    print("  2. Decision Tree Classification - Disease prediction")
    print("\n")

    # Run clustering analysis
    try:
        print("\n" + "#"*70)
        print("# PART 1: CLUSTERING ANALYSIS")
        print("#"*70 + "\n")

        clustering = HeartDiseaseClusteringAnalysis(random_state=random_state)
        cluster_stats, metrics = clustering.run_analysis(
            data_path=data_path,
            k_range=(2, 15),
            output_dir=output_dir
        )

    except Exception as e:
        print(f"\nError in clustering analysis: {str(e)}")
        import traceback
        traceback.print_exc()

    # Run classification analysis
    try:
        print("\n" + "#"*70)
        print("# PART 2: CLASSIFICATION ANALYSIS")
        print("#"*70 + "\n")

        classification = HeartDiseaseClassification(random_state=random_state)
        results = classification.run_analysis(
            data_path=data_path,
            output_dir=output_dir
        )

    except Exception as e:
        print(f"\nError in classification analysis: {str(e)}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n" + "="*70)
    print(" "*20 + "ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll results and visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - clustering_metrics.csv")
    print("  - cluster_statistics.csv")
    print("  - clustering_optimization.png")
    print("  - cluster_distribution.png")
    print("  - feature_importance.csv")
    print("  - classification_report.csv")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - performance_metrics.png")
    print("  - decision_tree.png")
    print("\n")


if __name__ == '__main__':
    main()
