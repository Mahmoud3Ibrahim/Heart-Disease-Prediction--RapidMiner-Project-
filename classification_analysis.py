"""
Decision Tree classification for heart disease prediction.
Implements cross-validation and train-test split evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            precision_score, recall_score, f1_score)
from preprocessing import HeartDiseasePreprocessor


class HeartDiseaseClassification:
    """Decision Tree classification for heart disease prediction."""

    def __init__(self, random_state=2001):
        self.random_state = random_state
        self.preprocessor = HeartDiseasePreprocessor()
        self.model = None
        self.feature_names = None
        self.cv_results = None
        self.test_results = None

    def build_model(self, max_depth=10, min_samples_split=4, min_samples_leaf=5):
        """
        Build Decision Tree classifier.

        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node

        Returns:
            DecisionTreeClassifier instance
        """
        model = DecisionTreeClassifier(
            criterion='gini',  # Similar to gain ratio
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            ccp_alpha=0.01  # Pruning parameter
        )

        return model

    def cross_validation_evaluation(self, X, y, cv_folds=10):
        """
        Perform k-fold cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with CV results
        """
        print(f"\n[Cross-Validation] Running {cv_folds}-fold cross-validation...")

        # Build model for CV
        cv_model = self.build_model(max_depth=10)

        # Perform cross-validation
        cv_scores = cross_val_score(cv_model, X, y, cv=cv_folds, scoring='accuracy')

        results = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'scores': cv_scores
        }

        print(f"Cross-Validation Accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")

        self.cv_results = results
        return results

    def train_test_evaluation(self, X, y, test_size=0.3):
        """
        Perform train-test split evaluation.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set

        Returns:
            Dictionary with test results
        """
        print(f"\n[Train-Test Split] Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Build and train model
        self.model = self.build_model(max_depth=7)
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred
        }

        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")

        self.test_results = results
        return results

    def get_feature_importance(self, top_n=10):
        """
        Get feature importance from trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print(f"\nTop {top_n} Important Features:")
        print(importance_df.head(top_n).to_string(index=False))

        return importance_df

    def visualize_results(self, output_dir='results'):
        """
        Create visualizations for classification results.

        Args:
            output_dir: Directory to save plots
        """
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150

        # 1. Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = self.test_results['confusion_matrix']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Disease', 'Heart Disease'],
                   yticklabels=['No Disease', 'Heart Disease'], ax=ax)

        ax.set_ylabel('Actual', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_title('Confusion Matrix - Test Set', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', bbox_inches='tight')
        print(f"Saved: {output_dir}/confusion_matrix.png")
        plt.close()

        # 2. Feature Importance
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_ylabel('Feature', fontsize=11)
        ax.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', bbox_inches='tight')
        print(f"Saved: {output_dir}/feature_importance.png")
        plt.close()

        # 3. Model Performance Comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            self.test_results['accuracy'],
            self.test_results['precision'],
            self.test_results['recall'],
            self.test_results['f1']
        ]

        bars = ax.bar(metrics, values, color=['steelblue', 'coral', 'lightgreen', 'gold'], alpha=0.7)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_metrics.png', bbox_inches='tight')
        print(f"Saved: {output_dir}/performance_metrics.png")
        plt.close()

        # 4. Decision Tree Visualization (simplified)
        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(self.model, filled=True, feature_names=self.feature_names,
                 class_names=['No Disease', 'Heart Disease'],
                 rounded=True, fontsize=10, max_depth=3, ax=ax)
        ax.set_title('Decision Tree Structure (Depth 3)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/decision_tree.png', bbox_inches='tight', dpi=150)
        print(f"Saved: {output_dir}/decision_tree.png")
        plt.close()

    def run_analysis(self, data_path, output_dir='results'):
        """
        Run complete classification analysis pipeline.

        Args:
            data_path: Path to heart disease CSV file
            output_dir: Directory to save results

        Returns:
            Dictionary with CV and test results
        """
        print("="*70)
        print("HEART DISEASE CLASSIFICATION ANALYSIS")
        print("="*70)

        # Load and preprocess data
        print("\n[1/5] Loading and preprocessing data...")
        df = self.preprocessor.load_data(data_path)
        df_processed = self.preprocessor.prepare_for_classification(df)

        # Prepare features and target
        target_col = 'HeartDisease'
        feature_cols = [col for col in df_processed.columns if col != target_col]

        X = df_processed[feature_cols].values
        y = df_processed[target_col].values
        self.feature_names = feature_cols

        print(f"Features: {len(feature_cols)}")
        print(f"Target distribution: {np.bincount(y)}")

        # Cross-validation
        print("\n[2/5] Performing cross-validation...")
        cv_results = self.cross_validation_evaluation(X, y, cv_folds=10)

        # Train-test split
        print("\n[3/5] Training and evaluating model...")
        test_results = self.train_test_evaluation(X, y, test_size=0.3)

        # Feature importance
        print("\n[4/5] Analyzing feature importance...")
        importance_df = self.get_feature_importance(top_n=10)

        # Save results
        importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)

        # Create classification report DataFrame
        report_dict = classification_report(test_results['y_test'], test_results['y_pred'],
                                           output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(f'{output_dir}/classification_report.csv')

        print(f"\nResults saved to {output_dir}/")

        # Visualize
        print("\n[5/5] Creating visualizations...")
        self.visualize_results(output_dir)

        print("\n" + "="*70)
        print("CLASSIFICATION ANALYSIS COMPLETED")
        print("="*70)

        return {
            'cv_results': cv_results,
            'test_results': test_results,
            'feature_importance': importance_df
        }


def main():
    """Run classification analysis."""
    classifier = HeartDiseaseClassification(random_state=2001)
    results = classifier.run_analysis('data/heart.csv')


if __name__ == '__main__':
    main()
