import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CardiovascularRiskAnalysis:
    def __init__(self, csv_path=None, save_plots=True):
        self.data = None
        self.X_train = None; self.X_test = None
        self.y_train = None; self.y_test = None
        self.time_train = None; self.time_test = None
        self.scaler = StandardScaler()
        self.cox_model = None
        self.dt_model = None
        self.results = {}
        self.csv_path = csv_path
        self.save_plots = save_plots

    def load_sample_data(self):
        # If CSV exists, try to load it; otherwise generate synthetic dataset
        if self.csv_path and os.path.exists(self.csv_path):
            print(f"Loading dataset from {self.csv_path}")
            self.data = pd.read_csv(self.csv_path)
        else:
            print("No framingham.csv found — generating synthetic dataset.")
            np.random.seed(42)
            n_samples = 4000
            data = {
                'age': np.random.normal(50, 15, n_samples).clip(25, 80),
                'sex': np.random.choice([0, 1], n_samples, p=[0.52, 0.48]),  # 0=female,1=male
                'education': np.random.choice([1,2,3,4], n_samples, p=[0.2,0.3,0.3,0.2]),
                'currentSmoker': np.random.choice([0,1], n_samples, p=[0.7,0.3]),
                'cigsPerDay': np.random.poisson(5, n_samples).clip(0,40),
                'BPMeds': np.random.choice([0,1], n_samples, p=[0.85,0.15]),
                'prevalentStroke': np.random.choice([0,1], n_samples, p=[0.95,0.05]),
                'prevalentHyp': np.random.choice([0,1], n_samples, p=[0.65,0.35]),
                'diabetes': np.random.choice([0,1], n_samples, p=[0.88,0.12]),
                'totChol': np.random.normal(240, 40, n_samples).clip(150,400),
                'sysBP': np.random.normal(130, 20, n_samples).clip(90,200),
                'diaBP': np.random.normal(85, 15, n_samples).clip(60,120),
                'BMI': np.random.normal(26, 4, n_samples).clip(18,40),
                'heartRate': np.random.normal(75, 12, n_samples).clip(50,120),
                'glucose': np.random.normal(85, 25, n_samples).clip(60,200)
            }
            # Build risk score and survival times
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
            baseline_hazard = 0.01
            hazard = baseline_hazard * np.exp(risk_score - np.mean(risk_score))
            survival_time = np.random.exponential(1 / hazard)
            follow_up_time = np.minimum(survival_time, 10)
            event = (survival_time <= 10).astype(int)

            data['time'] = follow_up_time
            data['TenYearCHD'] = event
            # Convert to DataFrame
            self.data = pd.DataFrame(data)

        # Basic checks / renames
        if 'sex' in self.data.columns and 'male' not in self.data.columns:
            # keep numeric sex as "male" for compatibility with earlier code
            self.data['male'] = self.data['sex']

        # If time/event columns absent, try to construct or raise informative error
        if 'time' not in self.data.columns or 'TenYearCHD' not in self.data.columns:
            raise ValueError("Dataset must contain 'time' (duration) and 'TenYearCHD' (event) columns." )

        print(f"Dataset loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        print(f"Event rate: {self.data['TenYearCHD'].mean():.2%}")
        return self.data

    def preprocess_data(self):
        print("=== Data Preprocessing ===")
        # Fill numeric missing with median, categorical with mode
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col].fillna(self.data[col].median(), inplace=True)
            else:
                self.data[col].fillna(self.data[col].mode().iat[0], inplace=True)

        feature_cols = [
            'age','male','currentSmoker','cigsPerDay','BPMeds',
            'prevalentStroke','prevalentHyp','diabetes','totChol',
            'sysBP','diaBP','BMI','heartRate','glucose'
        ]

        # Verify features exist
        missing_features = [c for c in feature_cols if c not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing expected feature columns: {missing_features}")

        X = self.data[feature_cols]
        T = self.data['time']
        E = self.data['TenYearCHD']

        # Train-test split: keep time and event paired
        self.X_train, self.X_test, self.time_train, self.time_test, self.y_train, self.y_test = \
            train_test_split(X, T, E, test_size=0.3, random_state=42, stratify=E)

        # Scale features for Cox
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train), columns=self.X_train.columns, index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test), columns=self.X_test.columns, index=self.X_test.index
        )

        print(f"Training size: {self.X_train.shape[0]}, Test size: {self.X_test.shape[0]}")
        print("Unique times in train (sample):", np.unique(self.time_train)[:10])
        print("Events in train/test:", int(self.y_train.sum()), int(self.y_test.sum()))

    def fit_cox_regression(self):
        print("\n=== Cox Proportional Hazards Model ===")
        cox_data = self.X_train_scaled.copy()
        cox_data['time'] = self.time_train.values
        cox_data['event'] = self.y_train.values.astype(int)

        self.cox_model = CoxPHFitter()
        self.cox_model.fit(cox_data, duration_col='time', event_col='event', show_progress=False)

        hazard_ratios = np.exp(self.cox_model.params_)
        self.results['cox'] = {
            'hazard_ratios': hazard_ratios,
            'cox_summary': self.cox_model.summary
        }

        # Concordance indices
        c_index_train = self.cox_model.concordance_index_
        risk_scores = self.cox_model.predict_partial_hazard(self.X_test_scaled)
        c_index_test = concordance_index(self.time_test, -risk_scores, self.y_test)
        self.results['cox'].update({'c_index_train': c_index_train, 'c_index_test': c_index_test, 'risk_scores': risk_scores})

        print(f"Cox C-index (train): {c_index_train:.3f}")
        print(f"Cox C-index (test) : {c_index_test:.3f}")

    def fit_decision_tree(self):
        print("\n=== Decision Tree Classifier ===")
        self.dt_model = DecisionTreeClassifier(max_depth=6, min_samples_split=50, min_samples_leaf=20, random_state=42)
        self.dt_model.fit(self.X_train, self.y_train)

        y_pred_train = self.dt_model.predict(self.X_train)
        y_pred_test = self.dt_model.predict(self.X_test)
        y_proba_test = self.dt_model.predict_proba(self.X_test)[:,1]

        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        test_auc = roc_auc_score(self.y_test, y_proba_test)

        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.dt_model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        self.results['dt'] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'feature_importance': feature_importance,
            'y_pred': y_pred_test,
            'y_proba': y_proba_test
        }

        print(f"Decision Tree - Train Acc: {train_accuracy:.3f}, Test Acc: {test_accuracy:.3f}, Test AUC: {test_auc:.3f}")

    def create_and_save_plots(self):
        print("\n=== Creating and Saving Plots ===")
        saved_files = []

        # Plot 1: Kaplan-Meier by smoking status
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10,6))
        kmf.fit(self.time_train, event_observed=self.y_train, label='Overall')
        ax = kmf.plot_survival_function(ci_show=False)
        for smoking_status, label in [(0,'Non-smoker'), (1,'Smoker')]:
            mask = self.X_train['currentSmoker'] == smoking_status
            if mask.sum() > 0:
                kmf.fit(self.time_train[mask], event_observed=self.y_train[mask], label=label)
                kmf.plot_survival_function(ax=ax, ci_show=False)
        plt.title('Kaplan-Meier Survival Curves by Smoking Status')
        plt.xlabel('Time (years)'); plt.ylabel('Survival Probability')
        plt.grid(True, alpha=0.3)
        fname = 'km_smoking_status.png'; plt.savefig(fname, dpi=300, bbox_inches='tight'); saved_files.append(fname)
        plt.close()

        # Plot 2: Cox hazard ratios
        hr_data = self.results['cox']['hazard_ratios'].sort_values()
        plt.figure(figsize=(8,6))
        colors = ['#FF6B6B' if x>1 else '#4ECDC4' for x in hr_data.values]
        bars = plt.barh(range(len(hr_data)), hr_data.values, color=colors, alpha=0.8, edgecolor='black')
        plt.yticks(range(len(hr_data)), hr_data.index)
        plt.xlabel('Hazard Ratio'); plt.title('Cox Model: Hazard Ratios')
        plt.axvline(1, color='k', linestyle='--', alpha=0.6)
        for i, v in enumerate(hr_data.values):
            plt.text(v + 0.02 if v>1 else v - 0.02, i, f'{v:.2f}', va='center', ha='left' if v>1 else 'right')
        fname = 'cox_hazard_ratios.png'; plt.savefig(fname, dpi=300, bbox_inches='tight'); saved_files.append(fname)
        plt.close()

        # Plot 3: Decision tree (depth 3)
        plt.figure(figsize=(12,8))
        plot_tree(self.dt_model, max_depth=3, feature_names=list(self.X_train.columns),
                  class_names=['No CHD','CHD'], filled=True, rounded=True, fontsize=10)
        plt.title('Decision Tree (Max Depth=3)')
        fname = 'decision_tree_depth3.png'; plt.savefig(fname, dpi=300, bbox_inches='tight'); saved_files.append(fname)
        plt.close()

        # Plot 4: Feature importance comparison
        dt_importance = self.results['dt']['feature_importance'].set_index('feature')['importance']
        cox_importance = np.abs(self.cox_model.params_)
        dt_norm = dt_importance / dt_importance.max() if dt_importance.max()>0 else dt_importance
        cox_norm = cox_importance / cox_importance.max() if cox_importance.max()>0 else cox_importance
        comparison_df = pd.DataFrame({'Decision Tree': dt_norm, 'Cox Model': cox_norm}).fillna(0)
        ax = comparison_df.plot(kind='bar', figsize=(12,6), width=0.8)
        plt.title('Feature Importance Comparison (Normalized)'); plt.ylabel('Normalized Importance')
        plt.xticks(rotation=45); plt.tight_layout()
        fname = 'feature_importance_comparison.png'; plt.savefig(fname, dpi=300, bbox_inches='tight'); saved_files.append(fname)
        plt.close()

        # Plot 5: Model performance comparison
        plt.figure(figsize=(8,6))
        models = ['Cox Regression', 'Decision Tree']
        c_index = [self.results['cox']['c_index_test'], self.results['dt']['test_auc']]
        acc = [None, self.results['dt']['test_accuracy']]
        x = np.arange(len(models))
        plt.bar(x - 0.15, c_index, width=0.3, label='C-index / AUC', color='#FF6B6B')
        plt.bar(x[1] + 0.15, acc[1], width=0.3, label='Accuracy', color='#4ECDC4')
        plt.xticks(x, models)
        plt.ylim(0,1); plt.legend(); plt.title('Model Performance Comparison')
        for i, v in enumerate(c_index):
            plt.text(i - 0.15, v + 0.02, f'{v:.3f}', ha='center')
        plt.text(1 + 0.15, acc[1] + 0.02, f'{acc[1]:.3f}', ha='center')
        fname = 'model_performance_comparison.png'; plt.savefig(fname, dpi=300, bbox_inches='tight'); saved_files.append(fname)
        plt.close()

        # Plot 6: Risk score distribution (Cox)
        risk_scores = self.results['cox']['risk_scores']
        plt.figure(figsize=(8,6))
        plt.hist(risk_scores[self.y_test==0], bins=30, alpha=0.7, label='No CHD', density=True)
        plt.hist(risk_scores[self.y_test==1], bins=30, alpha=0.7, label='CHD', density=True)
        plt.legend(); plt.title('Cox Model: Risk Score Distribution'); plt.xlabel('Risk Score'); plt.ylabel('Density')
        fname = 'cox_risk_score_distribution.png'; plt.savefig(fname, dpi=300, bbox_inches='tight'); saved_files.append(fname)
        plt.close()

        # Plot 7: Confusion matrix (Decision Tree)
        cm = confusion_matrix(self.y_test, self.results['dt']['y_pred'])
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CHD','CHD'], yticklabels=['No CHD','CHD'])
        plt.title('Decision Tree: Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual')
        fname = 'decision_tree_confusion_matrix.png'; plt.savefig(fname, dpi=300, bbox_inches='tight'); saved_files.append(fname)
        plt.close()

        # Plot 8: Age vs Risk score (scatter)
        plt.figure(figsize=(8,6))
        risk_scores_arr = np.asarray(risk_scores).flatten()
        plt.scatter(self.X_test['age'], risk_scores_arr, c=self.y_test, cmap='bwr', alpha=0.6)
        z = np.polyfit(self.X_test['age'], risk_scores_arr, 1)
        p = np.poly1d(z)
        age_range = np.linspace(self.X_test['age'].min(), self.X_test['age'].max(), 100)
        plt.plot(age_range, p(age_range), 'k--', alpha=0.7)
        plt.title('Age vs Cox Model Risk Score'); plt.xlabel('Age'); plt.ylabel('Risk Score')
        fname = 'age_vs_risk_score.png'; plt.savefig(fname, dpi=300, bbox_inches='tight'); saved_files.append(fname)
        plt.close()

        # Plot 9: Survival by risk quartiles
        risk_quartiles = pd.qcut(risk_scores, 4, labels=['Q1','Q2','Q3','Q4'])
        plt.figure(figsize=(10,6))
        kmf = KaplanMeierFitter()
        for quart in risk_quartiles.cat.categories:
            mask = risk_quartiles == quart
            if mask.sum()>0:
                kmf.fit(self.time_test[mask], event_observed=self.y_test[mask], label=str(quart))
                kmf.plot_survival_function(ci_show=False)
        plt.title('Survival by Risk Quartiles'); plt.xlabel('Time (years)'); plt.ylabel('Survival Probability')
        fname = 'survival_by_risk_quartiles.png'; plt.savefig(fname, dpi=300, bbox_inches='tight'); saved_files.append(fname)
        plt.close()

        print(f"Saved {len(saved_files)} plot files.")
        return saved_files

    def generate_report(self):
        # Simple textual report printed to console
        print("\n=== REPORT SUMMARY ===")
        print(f"Total samples: {len(self.data)}")
        print(f"Features used: {len(self.X_train.columns)}")
        print(f"Event rate: {self.data['TenYearCHD'].mean():.2%}")
        print(f"Cox test C-index: {self.results['cox']['c_index_test']:.3f}")
        print(f"Decision tree test AUC: {self.results['dt']['test_auc']:.3f}")


def main():
    analysis = CardiovascularRiskAnalysis(csv_path='framingham.csv', save_plots=True)
    analysis.load_sample_data()
    analysis.preprocess_data()
    analysis.fit_cox_regression()
    analysis.fit_decision_tree()
    saved = analysis.create_and_save_plots()
    analysis.generate_report()

    # Create zip: main.py plus saved plots
    files_to_zip = ['main.py'] + saved
    zip_name = 'cardiovascular_analysis_bundle.zip'
    import zipfile
    with zipfile.ZipFile(zip_name, 'w') as zf:
        for f in files_to_zip:
            if os.path.exists(f):
                zf.write(f)
    print(f"Created ZIP: {zip_name}")


if __name__ == '__main__':
    main()
