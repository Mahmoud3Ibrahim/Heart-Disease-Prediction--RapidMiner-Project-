# Heart Disease Prediction using RapidMiner

## Project Overview
This project analyzes heart disease dataset using RapidMiner to predict patient risk through two different methodologies: **K-Means Clustering** and **Decision Tree Classification**. The analysis helps identify patient segments and predict heart disease risk based on various health factors.

## Dataset Information
- **Records**: 918 patient records
- **Attributes**: 12 key health attributes
  - Age, Sex, ChestPainType, RestingBP, Cholesterol
  - FastingBS, RestingECG, MaxHR, ExerciseAngina
  - Oldpeak, ST_Slope, HeartDisease (target)

## Methodology

### Part 1: Clustering Analysis (`heart_disease_model_Clustring.xml`)

#### Data Preprocessing Pipeline:
1. **Data Cleaning**
   - Remove duplicate records
   - Generate unique ID for each patient
   - Handle missing values using average imputation for numerical attributes
   
2. **Feature Engineering**
   - Convert numerical to binomial: FastingBS
   - Transform nominal to numerical using dummy coding
   - Normalize numerical features (Age, Cholesterol, MaxHR, Oldpeak, RestingBP)
   - Range transformation (0-1 scaling)

3. **Clustering Process**
   - **Parameter Optimization**: Loop through k-values (2-80) to find optimal number of clusters
   - **K-Means Algorithm**: Applied with Euclidean distance measure
   - **Final Clustering**: Selected k=12 clusters based on performance
   - **Outlier Detection**: Filter clusters with ≤30 patients
   - **Cluster Analysis**: Generate cluster statistics and patient counts

#### Key Features:
- Automated cluster optimization
- Outlier detection and removal
- Cluster performance evaluation using "Avg. within centroid distance"

### Part 2: Decision Tree Classification (`heart_disease_model_tree.xml`)

#### Data Preprocessing:
1. **Missing Value Treatment**
   - Replace missing values with average for numerical attributes
   - Handle categorical variables appropriately

2. **Feature Engineering**
   - Convert numerical to binomial: FastingBS, HeartDisease
   - Transform Sex to binomial (M/F)
   - **Create Risk Categories**:
     - Age_Group: Teen, Young Adult, Adult, Middle Age, Senior, Elderly
     - Cholesterol_Level: Zero, Normal, Borderline, High
     - MaxHR_Level: Low, Normal, High
     - Oldpeak_Risk: Normal, Moderate, Severe
     - RestingBP_Level: Normal, Elevated, High

3. **Model Development**
   - **Cross-Validation**: 10-fold cross-validation for robust evaluation
   - **Train-Test Split**: 70%-30% split for independent validation
   - **Decision Tree Parameters**:
     - Criterion: Gain Ratio
     - Max Depth: 10 (CV) / 7 (Split)
     - Pruning: Applied with confidence 0.1/0.25
     - Min leaf size: 5, Min split size: 4

#### Evaluation Metrics:
- **Cross-Validation**: Accuracy assessment
- **Split Validation**: Accuracy, Classification Error, Precision, Recall

## Technical Implementation

### RapidMiner Operators Used:
- **Data Processing**: Remove Duplicates, Replace Missing Values, Normalize
- **Feature Engineering**: Generate Columns, Nominal to Numerical/Binomial
- **Clustering**: K-Means, Cluster Distance Performance
- **Classification**: Parallel Decision Tree, Cross Validation
- **Evaluation**: Performance Classification, Apply Model

### Key Parameters:
- **Random Seed**: 2001 (for reproducibility)
- **Clustering Distance**: Euclidean Distance
- **Tree Criterion**: Gain Ratio (Gini index equivalent)
- **Cross-Validation Folds**: 10

## Results and Insights

### Clustering Analysis:
- Successfully segmented patients into 12 distinct clusters
- Identified outlier clusters with small patient populations
- Enabled patient risk stratification based on health factors

### Decision Tree Classification:
- Built interpretable models for heart disease prediction
- Validated using both cross-validation and holdout methods
- Generated actionable insights through categorical risk levels

## Project Structure
```
heart-disease-prediction/
├── models/
│   ├── heart_disease_model_Clustring.xml
│   └── heart_disease_model_tree.xml
├── data/
│   └── heart_failure.csv
├── documentation/
│   └── methodology.md
└── README.md
```

## Tools and Technologies
- **RapidMiner Studio 11.0.001**
- **Machine Learning Techniques**: K-Means Clustering, Decision Trees
- **Validation Methods**: K-Fold Cross-Validation, Train-Test Split
- **Data Processing**: Normalization, Missing Value Imputation, Feature Engineering

## Key Achievements
- **Comprehensive Analysis**: Implemented both unsupervised (clustering) and supervised (classification) approaches
- **Robust Validation**: Multiple validation techniques ensure model reliability
- **Feature Engineering**: Created meaningful categorical variables for better interpretation
- **Scalable Methodology**: Automated parameter optimization for clustering

## Usage
1. Load the dataset into RapidMiner Local Repository as "heart_failure"
2. Import the XML process files
3. Execute the clustering analysis for patient segmentation
4. Run the decision tree process for classification and prediction
5. Analyze results and model performance metrics

## Future Enhancements
- Integration of ensemble methods
- Advanced feature selection techniques
- Real-time prediction capabilities
- Extended validation on external datasets
