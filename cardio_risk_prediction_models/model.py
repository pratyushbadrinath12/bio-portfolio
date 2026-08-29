# Cardiovascular Risk Prediction: Cox Regression vs Decision Tree Analysis
# Complete implementation for comparing survival analysis and machine learning approaches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CardiovascularRiskAnalysis:
    """
    A comprehensive analysis class comparing Cox regression and Decision Tree
    for cardiovascular disease risk prediction.
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.cox_model = None
        self.dt_model = None
        self.results = {}
        
    def load_sample_data(self):
        """
        Create a synthetic Framingham-style dataset for demonstration.
        In practice, you would load: pd.read_csv('framingham.csv')
        """
        np.random.seed(42)
        n_samples = 4000

        # Generate realistic cardiovascular risk factors
        data = {
            'age': np.random.normal(50, 15, n_samples).clip(25, 80),
            'sex': np.random.choice([0, 1], n_samples, p=[0.52, 0.48]),  # 0=female, 1=male
            'education': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
            'currentSmoker': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'cigsPerDay': np.random.poisson(5, n_samples).clip(0, 40),
            'BPMeds': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'prevalentStroke': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'prevalentHyp': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.88, 0.12]),
            'totChol': np.random.normal(240, 40, n_samples).clip(150, 400),
            'sysBP': np.random.normal(130, 20, n_samples).clip(90, 200),
            'diaBP': np.random.normal(85, 15, n_samples).clip(60, 120),
            'BMI': np.random.normal(26, 4, n_samples).clip(18, 40),
            'heartRate': np.random.normal(75, 12, n_samples).clip(50, 120),
            'glucose': np.random.normal(85, 25, n_samples).clip(60, 200)
        }

        # Create survival times and events
        risk_score = (
            0.05 * data['age'] +
            0.3 * data['sex'] +
            0.4 * data['currentSmoker'] +
            0.01 * data['cigsPerDay'] +
            0.3 * data['prevalentHyp'] +
            0.4 * data['diabetes'] +
            0.002 * data['totChol'] +
            0.01 * data['sysBP'] +
            0.05 * data['BMI']
        )

        # Generate survival times (exponential distribution)
        baseline_hazard = 0.01
        hazard = baseline_hazard * np.exp(risk_score - np.mean(risk_score))
        survival_time = np.random.exponential(1 / hazard)

        # Follow-up time (10 years max)
        follow_up_time = np.minimum(survival_time, 10)
        event = (survival_time <= 10).astype(int)  # 1 if event occurred, 0 if censored

        data['time'] = follow_up_time
        data['TenYearCHD'] = event

        # Use synthetic data instead of reading from CSV
        self.data = pd.DataFrame(data)

        # Handle missing values (simulate some)
        missing_indices = np.random.choice(n_samples, int(0.02 * n_samples), replace=False)
        self.data.loc[missing_indices[:len(missing_indices)//3], 'totChol'] = np.nan
        self.data.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'BMI'] = np.nan
        self.data.loc[missing_indices[2*len(missing_indices)//3:], 'glucose'] = np.nan

        print("Sample cardiovascular dataset created successfully!")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Event rate: {self.data['TenYearCHD'].mean():.2%}")

        return self.data
    
    def preprocess_data(self):
        """
        Clean and preprocess the dataset for both models.
        """
        print("\n=== Data Preprocessing ===")

        # Handle missing values
        print(f"Missing values before cleaning:")
        print(self.data.isnull().sum())

        # Fill missing values with median for numeric variables
        for col in self.data.columns:
            if self.data[col].dtype in ['float64', 'int64']:
                self.data[col].fillna(self.data[col].median(), inplace=True)
            else:
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)

        print(f"Missing values after cleaning: {self.data.isnull().sum().sum()}")
        print(f"\nMissing values after cleaning: {self.data.isnull().sum().sum()}")

        # Feature selection for models
        feature_cols = ['age', 'sex', 'currentSmoker', 'cigsPerDay', 'BPMeds',
                        'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
                        'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

        X = self.data[feature_cols]
        y = self.data['TenYearCHD']
        time = self.data['time']  # Use actual survival/censoring times

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test, self.time_train, self.time_test = \
            train_test_split(X, y, time, test_size=0.3, random_state=42, stratify=y)

        # Standardize features for Cox regression
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"Features: {len(feature_cols)}")
        
    def fit_cox_regression(self):
        """
        Fit Cox Proportional Hazards model.
        """
        print("\n=== Cox Proportional Hazards Model ===")
        
        # Prepare data for Cox regression
        cox_data = self.X_train_scaled.copy()
        cox_data['time'] = self.time_train
        cox_data['event'] = self.y_train
        
        # Fit Cox model
        self.cox_model = CoxPHFitter()
        self.cox_model.fit(cox_data, duration_col='time', event_col='event')
        
        print("Cox model fitted successfully!")
        print("\nHazard Ratios (exp(coef)):")
        hazard_ratios = np.exp(self.cox_model.params_)
        for feature, hr in hazard_ratios.items():
            print(f"{feature:15}: {hr:.3f}")
        
        # Calculate concordance index
        cox_test_data = self.X_test_scaled.copy()
        cox_test_data['time'] = self.time_test
        cox_test_data['event'] = self.y_test
        
        c_index = self.cox_model.concordance_index_
        print(f"\nConcordance Index (training): {c_index:.3f}")
        
        # Predict risk scores for test set
        risk_scores = self.cox_model.predict_partial_hazard(self.X_test_scaled)
        test_c_index = concordance_index(self.time_test, -risk_scores, self.y_test)
        print(f"Concordance Index (test): {test_c_index:.3f}")
        
        self.results['cox'] = {
            'c_index_train': c_index,
            'c_index_test': test_c_index,
            'hazard_ratios': hazard_ratios,
            'risk_scores': risk_scores
        }
        
    def fit_decision_tree(self):
        """
        Fit Decision Tree Classifier.
        """
        print("\n=== Decision Tree Classifier ===")
        
        # Fit decision tree (using original features, not scaled)
        self.dt_model = DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
        self.dt_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = self.dt_model.predict(self.X_train)
        y_pred_test = self.dt_model.predict(self.X_test)
        y_proba_test = self.dt_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        test_auc = roc_auc_score(self.y_test, y_proba_test)
        
        print(f"Training Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Test AUC: {test_auc:.3f}")
        
        print("\nFeature Importances:")
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.dt_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']:15}: {row['importance']:.3f}")
        
        self.results['dt'] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'feature_importance': feature_importance,
            'y_pred': y_pred_test,
            'y_proba': y_proba_test
        }
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations for both models.
        """
        print("\n=== Creating Visualizations ===")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Kaplan-Meier Survival Curves
        plt.subplot(3, 3, 1)
        kmf = KaplanMeierFitter()
        
        # Overall survival curve
        kmf.fit(self.time_train, self.y_train, label='Overall')
        kmf.plot_survival_function()
        
        # By smoking status
        for smoking_status, label in [(0, 'Non-smoker'), (1, 'Smoker')]:
            mask = self.X_train['currentSmoker'] == smoking_status
            if mask.sum() > 0:
                kmf.fit(self.time_train[mask], self.y_train[mask], label=label)
                kmf.plot_survival_function()
        
        plt.title('Kaplan-Meier Survival Curves by Smoking Status')
        plt.ylabel('Survival Probability')
        plt.xlabel('Time (years)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Cox Model Hazard Ratios
        plt.subplot(3, 3, 2)
        hr_data = self.results['cox']['hazard_ratios'].sort_values()
        colors = ['red' if x > 1 else 'blue' for x in hr_data.values]
        bars = plt.barh(range(len(hr_data)), hr_data.values, color=colors, alpha=0.7)
        plt.yticks(range(len(hr_data)), hr_data.index, rotation=45)
        plt.xlabel('Hazard Ratio')
        plt.title('Cox Model: Hazard Ratios')
        plt.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(hr_data.values):
            plt.text(v + 0.02 if v > 1 else v - 0.02, i, f'{v:.2f}', 
                    va='center', ha='left' if v > 1 else 'right')
        
        # 3. Decision Tree Visualization (simplified)
        plt.subplot(3, 3, 3)
        plot_tree(self.dt_model, max_depth=3, feature_names=list(self.X_train.columns),
                 class_names=['No CHD', 'CHD'], filled=True, rounded=True, fontsize=8)
        plt.title('Decision Tree (Depth=3)')
        
        # 4. Feature Importance Comparison
        plt.subplot(3, 3, 4)
        dt_importance = self.results['dt']['feature_importance'].set_index('feature')['importance']
        cox_importance = np.abs(self.cox_model.params_)  # Use absolute coefficients
        
        # Normalize both to [0, 1] for comparison
        dt_importance_norm = dt_importance / dt_importance.max()
        cox_importance_norm = cox_importance / cox_importance.max()
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Decision Tree': dt_importance_norm,
            'Cox Model': cox_importance_norm
        }).fillna(0)
        
        comparison_df.plot(kind='bar', ax=plt.gca())
        plt.title('Feature Importance Comparison\n(Normalized)')
        plt.ylabel('Normalized Importance')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Model Performance Comparison
        plt.subplot(3, 3, 5)
        models = ['Cox Regression', 'Decision Tree']
        c_index = [self.results['cox']['c_index_test'], self.results['dt']['test_auc']]
        accuracy = [None, self.results['dt']['test_accuracy']]
        
        x_pos = np.arange(len(models))
        plt.bar(x_pos - 0.2, c_index, 0.4, label='C-index/AUC', alpha=0.8)
        plt.bar(x_pos[1:] + 0.2, [accuracy[1]], 0.4, label='Accuracy', alpha=0.8)
        
        plt.xlabel('Model')
        plt.ylabel('Performance Metric')
        plt.title('Model Performance Comparison')
        plt.xticks(x_pos, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(c_index):
            if v is not None:
                plt.text(i - 0.2, v + 0.01, f'{v:.3f}', ha='center')
        plt.text(1 + 0.2, accuracy[1] + 0.01, f'{accuracy[1]:.3f}', ha='center')
        
        # 6. Risk Score Distribution (Cox Model)
        plt.subplot(3, 3, 6)
        risk_scores = self.results['cox']['risk_scores']
        plt.hist(risk_scores[self.y_test == 0], bins=30, alpha=0.7, label='No CHD', density=True)
        plt.hist(risk_scores[self.y_test == 1], bins=30, alpha=0.7, label='CHD', density=True)
        plt.xlabel('Risk Score')
        plt.ylabel('Density')
        plt.title('Cox Model: Risk Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Confusion Matrix (Decision Tree)
        plt.subplot(3, 3, 7)
        cm = confusion_matrix(self.y_test, self.results['dt']['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No CHD', 'CHD'], yticklabels=['No CHD', 'CHD'])
        plt.title('Decision Tree: Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 8. Age vs Risk (Cox Model)
        plt.subplot(3, 3, 8)
        scatter_colors = ['red' if event == 1 else 'blue' for event in self.y_test]
        plt.scatter(self.X_test['age'], risk_scores, c=scatter_colors, alpha=0.6)
        plt.xlabel('Age')
        plt.ylabel('Risk Score')
        plt.title('Cox Model: Age vs Risk Score')
        
        # Add trend line
        z = np.polyfit(self.X_test['age'], risk_scores, 1)
        p = np.poly1d(z)
        plt.plot(self.X_test['age'], p(self.X_test['age']), "r--", alpha=0.8)
        plt.grid(True, alpha=0.3)
        
        # 9. Survival by Risk Quartiles
        plt.subplot(3, 3, 9)
        risk_quartiles = pd.qcut(risk_scores, 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        
        kmf = KaplanMeierFitter()
        for quartile in risk_quartiles.cat.categories:
            mask = risk_quartiles == quartile
            if mask.sum() > 0:
                kmf.fit(self.time_test[mask], self.y_test[mask], label=f'{quartile}')
                kmf.plot_survival_function()
        
        plt.title('Survival Curves by Risk Quartiles')
        plt.ylabel('Survival Probability')
        plt.xlabel('Time (years)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cardiovascular_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self):
        """
        Generate a comprehensive analysis report.
        """
        print("\n" + "="*80)
        print("CARDIOVASCULAR RISK PREDICTION: COX REGRESSION VS DECISION TREE")
        print("="*80)
        
        print("\n1. DATASET OVERVIEW")
        print("-" * 40)
        print(f"Total samples: {len(self.data):,}")
        print(f"Features: {len(self.X_train.columns)}")
        print(f"Event rate: {self.data['TenYearCHD'].mean():.2%}")
        print(f"Follow-up time: 10.0 years (fixed)")        
        print("\n2. MODEL PERFORMANCE COMPARISON")
        print("-" * 40)
        print(f"{'Metric':<25} {'Cox Regression':<15} {'Decision Tree':<15}")
        print("-" * 55)
        print(f"{'C-index/AUC':<25} {self.results['cox']['c_index_test']:<15.3f} {self.results['dt']['test_auc']:<15.3f}")
        print(f"{'Accuracy (N/A for Cox)':<25} {'N/A':<15} {self.results['dt']['test_accuracy']:<15.3f}")
        
        print("\n3. INTERPRETABILITY ANALYSIS")
        print("-" * 40)
        
        print("\nCox Model - Top Risk Factors (Hazard Ratios):")
        hr_sorted = self.results['cox']['hazard_ratios'].sort_values(ascending=False)
        for feature, hr in hr_sorted.head(5).items():
            interpretation = "increases" if hr > 1 else "decreases"
            print(f"  {feature:15}: {hr:.3f} ({interpretation} risk)")
        
        print("\nDecision Tree - Top Features (Importance):")
        for _, row in self.results['dt']['feature_importance'].head(5).iterrows():
            print(f"  {row['feature']:15}: {row['importance']:.3f}")
        
        print("\n4. CLINICAL INSIGHTS")
        print("-" * 40)
        
        # Identify key risk factors
        high_risk_factors_cox = hr_sorted[hr_sorted > 1.2].index.tolist()
        high_risk_factors_dt = self.results['dt']['feature_importance'].head(3)['feature'].tolist()
        
        common_factors = set(high_risk_factors_cox).intersection(set(high_risk_factors_dt))
        
        print("Key findings:")
        print(f"- Both models identify these as important risk factors: {', '.join(common_factors)}")
        print(f"- Cox model identifies {len(high_risk_factors_cox)} significant risk factors")
        print(f"- Decision tree uses {len(high_risk_factors_dt)} primary splitting features")
        
        print("\n5. RECOMMENDATIONS")
        print("-" * 40)
        print("Model Selection Guidance:")
        
        if self.results['cox']['c_index_test'] > self.results['dt']['test_auc']:
            print("✓ Cox regression shows better discrimination (higher C-index)")
        else:
            print("✓ Decision tree shows better discrimination (higher AUC)")
        
        print("✓ Cox regression provides:")
        print("  - Quantitative risk estimates (hazard ratios)")
        print("  - Handles time-to-event data naturally")
        print("  - Established clinical interpretation")
        
        print("✓ Decision tree provides:")
        print("  - Simple decision rules")
        print("  - Easy to explain to patients")
        print("  - Handles non-linear relationships")
        
        print("\nRecommendation: Use Cox regression for clinical risk assessment")
        print("and decision trees for patient education and simple screening.")

def create_individual_plots(analysis):
    """
    Create individual plots one by one for screenshots.
    Each plot optimized for 1920x1080 display (16:9 aspect ratio).
    """
    
    # Set up matplotlib for high-quality screenshots
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    # Calculate figure size for 16:9 aspect ratio (width=16, height=9 inches)
    fig_width, fig_height = 16, 9
    
    print("Creating individual plots for screenshots...")
    print("Close each plot window to proceed to the next one.")
    print("=" * 60)
    
    # Plot 1: Kaplan-Meier Survival Curves
    print("Plot 1: Kaplan-Meier Survival Curves by Smoking Status")
    plt.figure(figsize=(fig_width, fig_height))
    
    kmf = KaplanMeierFitter()
    
    # Overall survival curve
    kmf.fit(analysis.time_train, analysis.y_train, label='Overall')
    ax = kmf.plot_survival_function(ci_show=False, linewidth=3)
    
    # By smoking status
    for smoking_status, label, color in [(0, 'Non-smoker', 'blue'), (1, 'Smoker', 'red')]:
        mask = analysis.X_train['currentSmoker'] == smoking_status
        if mask.sum() > 0:
            kmf.fit(analysis.time_train[mask], analysis.y_train[mask], label=label)
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color=color)
    
    plt.title('Kaplan-Meier Survival Curves by Smoking Status', fontweight='bold', pad=20)
    plt.ylabel('Survival Probability (No CHD)', fontweight='bold')
    plt.xlabel('Time (years)', fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True, loc='lower left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Cox Model Hazard Ratios
    print("\nPlot 2: Cox Model Hazard Ratios")
    plt.figure(figsize=(fig_width, fig_height))
    
    hr_data = analysis.results['cox']['hazard_ratios'].sort_values()
    colors = ['#FF6B6B' if x > 1 else '#4ECDC4' for x in hr_data.values]
    
    bars = plt.barh(range(len(hr_data)), hr_data.values, color=colors, alpha=0.8, edgecolor='black')
    plt.yticks(range(len(hr_data)), [name.replace('_', ' ').title() for name in hr_data.index])
    plt.xlabel('Hazard Ratio', fontweight='bold')
    plt.title('Cox Proportional Hazards Model: Risk Factor Analysis', fontweight='bold', pad=20)
    plt.axvline(x=1, color='black', linestyle='--', alpha=0.7, linewidth=2)
    plt.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels on bars
    for i, (bar, v) in enumerate(zip(bars, hr_data.values)):
        plt.text(v + 0.03 if v > 1 else v - 0.03, i, f'{v:.2f}', 
                va='center', ha='left' if v > 1 else 'right', fontweight='bold')
    
    # Add interpretation text
    plt.text(0.02, 0.98, 'HR > 1: Increased Risk\nHR < 1: Decreased Risk', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Decision Tree Visualization
    print("\nPlot 3: Decision Tree Structure")
    plt.figure(figsize=(fig_width, fig_height))
    
    plot_tree(analysis.dt_model, max_depth=3, 
             feature_names=[name.replace('_', ' ').title() for name in analysis.X_train.columns],
             class_names=['No CHD', 'CHD'], 
             filled=True, rounded=True, fontsize=12,
             impurity=True, proportion=True)
    
    plt.title('Decision Tree for 10-Year CHD Risk Prediction\n(Max Depth = 3)', 
              fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Plot 4: Feature Importance Comparison
    print("\nPlot 4: Feature Importance Comparison")
    plt.figure(figsize=(fig_width, fig_height))
    
    dt_importance = analysis.results['dt']['feature_importance'].set_index('feature')['importance']
    cox_importance = np.abs(analysis.cox_model.params_)
    
    # Normalize both to [0, 1] for comparison
    dt_importance_norm = dt_importance / dt_importance.max()
    cox_importance_norm = cox_importance / cox_importance.max()
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Decision Tree': dt_importance_norm,
        'Cox Regression': cox_importance_norm
    }).fillna(0)
    
    # Create the plot
    ax = comparison_df.plot(kind='bar', width=0.8, 
                           color=['#FF9999', '#66B2FF'], alpha=0.8, edgecolor='black')
    plt.title('Feature Importance Comparison Between Models', fontweight='bold', pad=20)
    plt.ylabel('Normalized Importance Score', fontweight='bold')
    plt.xlabel('Risk Factors', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', rotation=90, padding=3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 5: Model Performance Comparison
    print("\nPlot 5: Model Performance Metrics")
    plt.figure(figsize=(fig_width, fig_height))
    
    models = ['Cox Regression', 'Decision Tree']
    c_index_auc = [analysis.results['cox']['c_index_test'], analysis.results['dt']['test_auc']]
    accuracy = [None, analysis.results['dt']['test_accuracy']]
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    bars1 = plt.bar(x_pos - width/2, c_index_auc, width, label='C-index/AUC', 
                    color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = plt.bar([x_pos[1] + width/2], [accuracy[1]], width, label='Accuracy', 
                    color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Performance Score', fontweight='bold')
    plt.title('Model Performance Comparison', fontweight='bold', pad=20)
    plt.xticks(x_pos, models)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(c_index_auc):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    plt.text(1 + width/2, accuracy[1] + 0.02, f'{accuracy[1]:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Plot 6: Risk Score Distribution
    print("\nPlot 6: Cox Model Risk Score Distribution")
    plt.figure(figsize=(fig_width, fig_height))
    
    risk_scores = analysis.results['cox']['risk_scores']
    
    plt.hist(risk_scores[analysis.y_test == 0], bins=30, alpha=0.7, 
             label='No CHD', color='#4ECDC4', edgecolor='black', density=True)
    plt.hist(risk_scores[analysis.y_test == 1], bins=30, alpha=0.7, 
             label='CHD', color='#FF6B6B', edgecolor='black', density=True)
    
    plt.xlabel('Relative Risk Score', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    plt.title('Distribution of Cox Model Risk Scores by Outcome', fontweight='bold', pad=20)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add vertical lines for means
    mean_no_chd = risk_scores[analysis.y_test == 0].mean()
    mean_chd = risk_scores[analysis.y_test == 1].mean()
    plt.axvline(mean_no_chd, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8)
    plt.axvline(mean_chd, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 7: Confusion Matrix
    print("\nPlot 7: Decision Tree Confusion Matrix")
    plt.figure(figsize=(fig_width, fig_height))
    
    cm = confusion_matrix(analysis.y_test, analysis.results['dt']['y_pred'])
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No CHD', 'CHD'], yticklabels=['No CHD', 'CHD'],
                square=True, cbar_kws={'shrink': 0.8})
    
    plt.title('Decision Tree Classification Results\nConfusion Matrix', fontweight='bold', pad=20)
    plt.ylabel('Actual Diagnosis', fontweight='bold')
    plt.xlabel('Predicted Diagnosis', fontweight='bold')
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.02, 0.98, f'Overall Accuracy: {accuracy:.3f}', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Plot 8: Age vs Risk Relationship
    print("\nPlot 8: Age vs Risk Score Relationship")
    plt.figure(figsize=(fig_width, fig_height))
    
    scatter_colors = ['#FF6B6B' if event == 1 else '#4ECDC4' for event in analysis.y_test]
    scatter_labels = ['CHD' if event == 1 else 'No CHD' for event in analysis.y_test]
    
    # Create scatter plot
    for outcome, color, label in [(0, '#4ECDC4', 'No CHD'), (1, '#FF6B6B', 'CHD')]:
        mask = analysis.y_test == outcome
        plt.scatter(analysis.X_test['age'][mask], risk_scores[mask], 
                   c=color, alpha=0.6, s=60, label=label, edgecolors='black')
    
    plt.xlabel('Age (years)', fontweight='bold')
    plt.ylabel('Cox Model Risk Score', fontweight='bold')
    plt.title('Relationship Between Age and Cardiovascular Risk', fontweight='bold', pad=20)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add trend line
    z = np.polyfit(analysis.X_test['age'], risk_scores, 1)
    p = np.poly1d(z)
    age_range = np.linspace(analysis.X_test['age'].min(), analysis.X_test['age'].max(), 100)
    plt.plot(age_range, p(age_range), "r--", alpha=0.8, linewidth=3, label='Trend Line')
    
    plt.tight_layout()
    plt.show()
    
    # Plot 9: Survival by Risk Quartiles
    print("\nPlot 9: Survival Curves by Risk Quartiles")
    plt.figure(figsize=(fig_width, fig_height))
    
    risk_quartiles = pd.qcut(risk_scores, 4, labels=['Q1 (Lowest Risk)', 'Q2 (Low Risk)', 
                                                    'Q3 (High Risk)', 'Q4 (Highest Risk)'])
    
    colors = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']  # Green to Red gradient
    
    kmf = KaplanMeierFitter()
    for i, (quartile, color) in enumerate(zip(risk_quartiles.cat.categories, colors)):
        mask = risk_quartiles == quartile
        if mask.sum() > 0:
            kmf.fit(analysis.time_test[mask], analysis.y_test[mask], label=f'{quartile}')
            ax = kmf.plot_survival_function(ci_show=False, linewidth=4, color=color)
    
    plt.title('Survival Probability by Cox Model Risk Quartiles', fontweight='bold', pad=20)
    plt.ylabel('Survival Probability (No CHD)', fontweight='bold')
    plt.xlabel('Time (years)', fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True, loc='lower left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 1.05)
    
    # Add summary statistics
    event_rates = []
    for quartile in risk_quartiles.cat.categories:
        mask = risk_quartiles == quartile
        event_rate = analysis.y_test[mask].mean()
        event_rates.append(event_rate)
    
    summary_text = "Event Rates by Quartile:\n" + "\n".join([f"{q}: {r:.1%}" 
                                                             for q, r in zip(['Q1', 'Q2', 'Q3', 'Q4'], event_rates)])
    plt.text(0.98, 0.98, summary_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("\nAll plots have been displayed!")
    print("Screenshots should be taken at 1920x1080 resolution for best quality.")





def main():
    """
    Main execution function for the cardiovascular risk analysis.
    """
    print("Starting Cardiovascular Risk Prediction Analysis")
    print("=" * 60)
    
    # Initialize analysis
    analysis = CardiovascularRiskAnalysis()
    
    # Step 1: Load and preprocess data
    analysis.load_sample_data()
    analysis.preprocess_data()
    
    # Step 2: Fit models
    analysis.fit_cox_regression()
    analysis.fit_decision_tree()
    
    # Step 3: Create visualizations
    create_individual_plots(analysis)
    
    # Step 4: Generate report
    analysis.generate_report()
    
    print("\nAnalysis completed successfully!")
    print("Visualizations saved as 'cardiovascular_risk_analysis.png'")

if __name__ == "__main__":
    main()

# Additional utility functions for extended analysis

def perform_model_validation():
    """
    Perform cross-validation and bootstrap analysis.
    """
    print("\n=== Extended Model Validation ===")
    # This could include k-fold cross-validation, bootstrap confidence intervals, etc.
    pass

def sensitivity_analysis():
    """
    Perform sensitivity analysis for model assumptions.
    """
    print("\n=== Sensitivity Analysis ===")
    # This could test proportional hazards assumption, feature stability, etc.
    pass

def create_risk_calculator():
    """
    Create a simple risk calculator based on the fitted models.
    """
    print("\n=== Risk Calculator ===")
    # This could create a simple interface for risk calculation
    pass

# To run this analysis, simply execute:
# python cardiovascular_analysis.py

# Expected outputs:
# 1. Comprehensive comparison of both models
# 2. Multiple visualizations showing model performance and insights
# 3. Detailed report with clinical interpretations
# 4. Recommendations for model selection in clinical practice