
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
import joblib
import requests
import os
import matplotlib
# Try interactive backend first, fallback to Agg if not available
try:
    matplotlib.use('TkAgg')  # Interactive backend
except:
    try:
        matplotlib.use('Qt5Agg')  # Alternative interactive backend
    except:
        matplotlib.use('Agg')  # Non-interactive fallback
import matplotlib.pyplot as plt
import seaborn as sns

FDA_API_KEY = ""

# Configuration Constants
RANDOM_SEED = 42
DEFAULT_N_PATIENTS = 35000
DEFAULT_EPOCHS = 100
BATCH_SIZE = 128
PREDICTION_THRESHOLD = 0.7
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 8
MIN_LEARNING_RATE = 1e-7
LR_REDUCTION_FACTOR = 0.5

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    DEEP_LEARNING_AVAILABLE = True
    print("Deep learning libraries loaded successfully")
    # Set random seeds for reproducibility
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    print(f"Deep learning libraries not available: {e}")
    print("Install: pip install tensorflow")


class LungCancerMLPPredictor:
    """Deep Learning MLP model for predicting lung cancer chemotherapy treatment response."""
    
    def __init__(self):
        """Initialize the predictor with model paths and cache."""
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.drug_info_cache = {}
        self.model_path = 'lung_cancer_mlp_model.h5'
        self.scaler_path = 'lung_cancer_scaler.joblib'
        self.encoders_path = 'lung_cancer_encoders.joblib'
        self.prediction_threshold = PREDICTION_THRESHOLD  # Conservative threshold to reduce false positives
        self.feature_names = None
        self._load_model_if_exists()

    def _load_model_if_exists(self):
        """Load previously trained model if it exists."""
        try:
            if (os.path.exists(self.model_path) and 
                os.path.exists(self.scaler_path) and 
                os.path.exists(self.encoders_path) and 
                DEEP_LEARNING_AVAILABLE):
                self.model = keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.label_encoders = joblib.load(self.encoders_path)
                print("Loaded existing MLP model")
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False

    def fetch_drug_info(self, drug_name):
        """Fetch drug information from OpenFDA API."""
        
        if drug_name in self.drug_info_cache:
            return self.drug_info_cache[drug_name]

        try:
            base_url = "https://api.fda.gov/drug/label.json"
            
            search_queries = [
                f'openfda.generic_name:"{drug_name}"',
                f'openfda.brand_name:"{drug_name}"',
                f'{drug_name}'
            ]
            
            print(f"Fetching drug information for {drug_name}...")
            
            for search_query in search_queries:
                params = {
                    "api_key": FDA_API_KEY,
                    "search": search_query,
                    "limit": 1
                }
                
                response = requests.get(base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "results" in data and len(data["results"]) > 0:
                        result = data["results"][0]
                        drug_info = {
                            "name": drug_name,
                            "warnings": result.get("warnings", []),
                            "indications_and_usage": result.get("indications_and_usage", []),
                            "contraindications": result.get("contraindications", []),
                            "description": result.get("description", []),
                            "adverse_reactions": result.get("adverse_reactions", [])
                        }
                        self.drug_info_cache[drug_name] = drug_info
                        return drug_info
            
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching drug info: {e}")
            return None

    def generate_comprehensive_dummy_data(self, n_patients=DEFAULT_N_PATIENTS):
        """Generate comprehensive dummy lung cancer dataset suitable for deep learning."""
        print(f"Generating comprehensive dummy dataset with {n_patients:,} patients...")
        np.random.seed(RANDOM_SEED)
        
        # Define comprehensive feature set
        data = {}
        
        # 1. Demographics (5 features)
        data['age'] = np.random.normal(65, 12, n_patients).clip(18, 90)
        data['sex'] = np.random.choice(['Male', 'Female'], n_patients, p=[0.62, 0.38])
        data['race'] = np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], 
                                       n_patients, p=[0.70, 0.15, 0.08, 0.05, 0.02])
        data['bmi'] = np.random.normal(26, 4.5, n_patients).clip(15, 45)
        data['insurance'] = np.random.choice(['Private', 'Medicare', 'Medicaid', 'Uninsured'], 
                                           n_patients, p=[0.45, 0.35, 0.15, 0.05])
        
        # 2. Clinical Characteristics (10 features)
        data['histology_type'] = np.random.choice(['Adenocarcinoma', 'Squamous', 'Small_cell', 'Large_cell', 'Other'], 
                                                 n_patients, p=[0.45, 0.25, 0.15, 0.10, 0.05])
        data['cancer_stage'] = np.random.choice(['IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IV'], 
                                              n_patients, p=[0.08, 0.12, 0.08, 0.12, 0.15, 0.20, 0.25])
        data['ecog_performance'] = np.random.choice([0, 1, 2, 3, 4], n_patients, p=[0.25, 0.50, 0.18, 0.06, 0.01])
        data['tumor_size'] = np.random.lognormal(1.2, 0.8, n_patients).clip(0.5, 15)
        data['metastatic_sites'] = np.random.poisson(1.5, n_patients).clip(0, 6)
        data['tumor_location'] = np.random.choice(['Upper_lobe', 'Lower_lobe', 'Multiple', 'Central'], 
                                                n_patients, p=[0.45, 0.30, 0.15, 0.10])
        data['pleural_effusion'] = np.random.choice([0, 1], n_patients, p=[0.70, 0.30])
        data['weight_loss_pct'] = np.random.exponential(3, n_patients).clip(0, 30)
        data['time_since_diagnosis'] = np.random.exponential(8, n_patients).clip(0, 60)
        data['karnofsky_score'] = np.random.normal(75, 15, n_patients).clip(30, 100)
        
        # 3. Laboratory Values (15 features) - realistic ranges
        data['wbc_count'] = np.random.lognormal(2.2, 0.3, n_patients).clip(2, 20)
        data['hemoglobin'] = np.random.normal(12.5, 2.1, n_patients).clip(6, 18)
        data['platelet_count'] = np.random.normal(280, 80, n_patients).clip(50, 600)
        data['albumin'] = np.random.normal(3.8, 0.6, n_patients).clip(2.0, 5.0)
        data['ldh'] = np.random.lognormal(6.1, 0.4, n_patients).clip(100, 2000)
        data['creatinine'] = np.random.lognormal(0.1, 0.3, n_patients).clip(0.5, 4.0)
        data['bilirubin'] = np.random.lognormal(-0.5, 0.5, n_patients).clip(0.2, 5.0)
        data['alt'] = np.random.lognormal(3.2, 0.5, n_patients).clip(10, 200)
        data['ast'] = np.random.lognormal(3.3, 0.5, n_patients).clip(15, 250)
        data['crp'] = np.random.lognormal(1.5, 1.2, n_patients).clip(0.1, 100)
        data['cea_level'] = np.random.lognormal(1.8, 1.5, n_patients).clip(0.5, 500)
        data['calcium'] = np.random.normal(9.8, 0.6, n_patients).clip(7.5, 12.5)
        data['glucose'] = np.random.normal(110, 25, n_patients).clip(70, 300)
        data['neutrophil_pct'] = np.random.normal(65, 12, n_patients).clip(40, 90)
        data['lymphocyte_pct'] = np.random.normal(25, 8, n_patients).clip(5, 50)
        
        # 4. Genetic/Molecular Markers (8 features)
        data['egfr_mutation'] = np.random.choice(['Wild_type', 'Exon19del', 'L858R', 'Other'], 
                                               n_patients, p=[0.65, 0.15, 0.12, 0.08])
        data['kras_mutation'] = np.random.choice([0, 1], n_patients, p=[0.75, 0.25])
        data['alk_rearrangement'] = np.random.choice([0, 1], n_patients, p=[0.95, 0.05])
        data['pdl1_expression'] = np.random.beta(2, 5, n_patients) * 100
        data['tp53_mutation'] = np.random.choice([0, 1], n_patients, p=[0.45, 0.55])
        data['ros1_rearrangement'] = np.random.choice([0, 1], n_patients, p=[0.98, 0.02])
        data['braf_mutation'] = np.random.choice([0, 1], n_patients, p=[0.92, 0.08])
        data['pik3ca_mutation'] = np.random.choice([0, 1], n_patients, p=[0.85, 0.15])
        
        # 5. Treatment History (6 features)
        data['previous_chemo_lines'] = np.random.poisson(1.2, n_patients).clip(0, 5)
        data['previous_radiation'] = np.random.choice([0, 1], n_patients, p=[0.60, 0.40])
        data['previous_immunotherapy'] = np.random.choice([0, 1], n_patients, p=[0.70, 0.30])
        data['previous_surgery'] = np.random.choice(['None', 'Biopsy', 'Lobectomy', 'Pneumonectomy'], 
                                                  n_patients, p=[0.30, 0.35, 0.25, 0.10])
        data['concurrent_medications'] = np.random.poisson(5, n_patients).clip(0, 20)
        data['treatment_delay_days'] = np.random.exponential(15, n_patients).clip(0, 180)
        
        # 6. Comorbidities & Risk Factors (8 features)
        data['smoking_pack_years'] = np.random.exponential(25, n_patients).clip(0, 120)
        data['smoking_status'] = np.random.choice(['Never', 'Former_lt5yr', 'Former_gt5yr', 'Current'], 
                                                n_patients, p=[0.15, 0.20, 0.35, 0.30])
        data['copd_severity'] = np.random.choice(['None', 'Mild', 'Moderate', 'Severe'], 
                                               n_patients, p=[0.40, 0.30, 0.20, 0.10])
        data['cardiovascular_disease'] = np.random.choice([0, 1], n_patients, p=[0.65, 0.35])
        data['diabetes_type'] = np.random.choice(['None', 'Type1', 'Type2'], n_patients, p=[0.70, 0.05, 0.25])
        data['other_cancer_history'] = np.random.choice([0, 1], n_patients, p=[0.80, 0.20])
        data['family_cancer_history'] = np.random.choice([0, 1], n_patients, p=[0.55, 0.45])
        data['alcohol_use'] = np.random.choice(['None', 'Moderate', 'Heavy'], n_patients, p=[0.35, 0.50, 0.15])
        
        # Generate realistic treatment response based on clinical factors
        response_prob = self._calculate_response_probability(data, n_patients)
        data['treatment_response'] = np.random.binomial(1, response_prob, n_patients)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Store feature names for later use
        self.feature_names = [col for col in df.columns if col != 'treatment_response']
        
        print(f"Generated dataset: {len(df):,} patients, {len(self.feature_names)} features")
        print(f"Response rate: {df['treatment_response'].mean():.1%}")
        print(f"Feature categories: Demographics(5), Clinical(10), Labs(15), Genetics(8), History(6), Risk_factors(8)")
        
        return df
    
    def _calculate_response_probability(self, data, n_patients):
        """Calculate enhanced realistic response probability with improved clinical relationships."""
        # Start with age-stratified base probability (more realistic)
        age_array = np.array(data['age'])
        prob = np.where(age_array < 50, 0.55,  # Young: better baseline
                np.where(age_array < 70, 0.45,    # Middle-aged: standard
                         0.35))                   # Elderly: lower baseline
        
        # MAJOR PROGNOSTIC FACTORS (stronger relationships)
        # Cancer stage - most important factor
        stage_series = pd.Series(data['cancer_stage'])
        prob += np.where(stage_series.isin(['IA', 'IB']), 0.30,        # Early stage: major benefit
                np.where(stage_series.isin(['IIA', 'IIB']), 0.20,     # Intermediate: good benefit  
                np.where(stage_series.isin(['IIIA', 'IIIB']), 0.05,   # Advanced: small benefit
                         -0.25)))                                      # Stage IV: poor prognosis
        
        # EGFR mutation status - critical for targeted therapy
        egfr_series = pd.Series(data['egfr_mutation'])
        prob += np.where(egfr_series == 'Exon19del', 0.35,           # Best EGFR mutation
                np.where(egfr_series == 'L858R', 0.30,               # Good EGFR mutation
                np.where(egfr_series == 'Other', 0.20, 0)))          # Other mutations
        
        # Performance status - functional capacity
        ecog_array = np.array(data['ecog_performance'])
        prob += np.where(ecog_array == 0, 0.20,          # Perfect performance
                np.where(ecog_array == 1, 0.10,          # Good performance  
                np.where(ecog_array == 2, -0.05,         # Limited performance
                         -0.25)))                         # Poor performance (3-4)
        
        # LABORATORY MARKERS (more nuanced relationships)
        # Hemoglobin - oxygen carrying capacity
        hgb_array = np.array(data['hemoglobin'])
        prob += np.where(hgb_array >= 12, 0.10,          # Normal hemoglobin
                np.where(hgb_array >= 10, 0.05,          # Mild anemia
                         -0.15))                          # Severe anemia
        
        # LDH - tumor burden marker (more graduated response)
        ldh_array = np.array(data['ldh'])
        prob += np.where(ldh_array <= 250, 0.12,         # Normal LDH
                np.where(ldh_array <= 400, 0.05,         # Mildly elevated
                np.where(ldh_array <= 600, -0.08,        # Moderately elevated
                         -0.20)))                         # Highly elevated
        
        # Albumin - nutritional status
        alb_array = np.array(data['albumin'])
        prob += np.where(alb_array >= 3.5, 0.08,         # Good nutrition
                np.where(alb_array >= 3.0, 0.02,         # Mild malnutrition
                         -0.12))                          # Poor nutrition
        
        # GENETIC MARKERS (additive effects)
        prob += np.where(np.array(data['kras_mutation']) == 1, -0.08, 0)    # KRAS resistance
        prob += np.where(np.array(data['tp53_mutation']) == 1, -0.06, 0)    # TP53 poor prognosis
        prob += np.where(np.array(data['alk_rearrangement']) == 1, 0.25, 0) # ALK targeted therapy
        
        # DISEASE BURDEN
        prob -= np.where(np.array(data['metastatic_sites']) >= 3, 0.18, 0)  # Multiple mets
        prob -= np.where(np.array(data['tumor_size']) > 7, 0.10, 0)         # Large tumors
        prob -= np.where(np.array(data['weight_loss_pct']) > 15, 0.18, 0)   # Severe weight loss
        
        # TREATMENT HISTORY (resistance patterns)
        chemo_lines = np.array(data['previous_chemo_lines'])
        prob -= np.where(chemo_lines >= 3, 0.30,         # Heavily pretreated
                np.where(chemo_lines >= 1, 0.12, 0))     # Some prior treatment
        
        # COMORBIDITIES
        prob -= np.where(np.array(data['cardiovascular_disease']) == 1, 0.08, 0)
        prob -= np.where(pd.Series(data['copd_severity']).isin(['Moderate', 'Severe']), 0.10, 0)
        
        # More realistic probability bounds for medical outcomes
        return np.clip(prob, 0.10, 0.85)
    
    def create_mlp_architecture(self, input_dim):
        """Create MLP neural network architecture for lung cancer prediction."""
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("TensorFlow not available for deep learning")
            
        model = models.Sequential([
            # Input layer
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(96, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model with improved optimizer
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def preprocess_features(self, df, training=True):
        """Preprocess features for MLP training."""
        print("Preprocessing features for deep learning...")
        
        # Separate features and target
        if 'treatment_response' in df.columns:
            X = df.drop('treatment_response', axis=1).copy()
            y = df['treatment_response'].values
        else:
            X = df.copy()
            y = None
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        if training:
            # Fit and transform during training
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        else:
            # Only transform during prediction
            for col in categorical_columns:
                if col in self.label_encoders:
                    # Handle unseen categories
                    X[col] = X[col].astype(str)
                    mask = X[col].isin(self.label_encoders[col].classes_)
                    X.loc[~mask, col] = 'Unknown'
                    # Add unknown to encoder if needed
                    if 'Unknown' not in self.label_encoders[col].classes_:
                        classes = list(self.label_encoders[col].classes_) + ['Unknown']
                        self.label_encoders[col].classes_ = np.array(classes)
                    X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale features
        if training:
            if self.scaler is None:
                self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Please train the model first.")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y

    def train(self, n_patients=DEFAULT_N_PATIENTS, epochs=DEFAULT_EPOCHS):
        """Train the MLP model on comprehensive dummy lung cancer dataset.
        
        Args:
            n_patients (int): Number of patients to generate for training
            epochs (int): Number of training epochs
        """
        print("\n" + "="*50)
        print("TRAINING MLP MODEL")
        print("="*50)
        
        if not DEEP_LEARNING_AVAILABLE:
            print("ERROR: TensorFlow not available. Please install: pip install tensorflow")
            return False
        
        # Generate comprehensive dummy dataset
        print("GENERATING: Creating comprehensive dummy dataset...")
        df = self.generate_comprehensive_dummy_data(n_patients)
        
        print(f"STATS: Dataset created successfully: {len(df):,} patients, {len(self.feature_names)} features")
        print(f"SOURCE: Comprehensive dummy clinical trial dataset")
        
        # Prepare data for MLP training
        self.scaler = StandardScaler()
        X_processed, y = self.preprocess_features(df, training=True)
        
        # Split the data (70% train, 15% val, 15% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_processed, y, test_size=0.15, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
        )
        
        print(f"Data split: Train={len(X_train):,}, Validation={len(X_val):,}, Test={len(X_test):,}")
        
        # Create MLP architecture
        print("ARCHITECTURE: Building MLP neural network...")
        input_dim = X_processed.shape[1]
        self.model = self.create_mlp_architecture(input_dim)
        
        print(f"Model architecture: {input_dim} inputs -> 128 -> 96 -> 64 -> 32 -> 16 -> 1 output")
        print(f"Total parameters: {self.model.count_params():,}")
        
        # Setup enhanced callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_auc',  # Monitor AUC instead of loss for better medical relevance
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_auc',
                factor=LR_REDUCTION_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LEARNING_RATE,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Train the model
        print("TRAINING: Starting MLP training...")
        print(f"Configuration: {epochs} epochs, early stopping enabled, learning rate scheduling")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEVALUATION: Testing final model performance...")
        test_loss, test_accuracy, test_precision, test_recall, test_auc = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"Final Test Results:")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall: {test_recall:.4f}")
        
        # Save model and preprocessors
        print("\nSAVING: Saving model and preprocessors...")
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoders, self.encoders_path)
        print("MLP model saved successfully")
        
        # Save test data and history for visualization
        self.X_test_scaled = X_test
        self.y_test = y_test
        self.training_history = history.history
        
        # Create visualizations
        self.visualize_mlp_results(X_test, y_test, history, show=False)
        
        return True

    def visualize_mlp_results(self, X_test, y_test, history, show=True):
        """Generate and save MLP training visualization with training history and performance metrics."""
        try:
            # Make predictions
            y_test_proba = self.model.predict(X_test).flatten()
            y_test_pred = (y_test_proba > 0.5).astype(int)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Lung Cancer MLP Deep Learning Model - Training Results', fontsize=16, fontweight='bold')
            
            # 1. Training History - Loss
            axes[0, 0].plot(history.history['loss'], label='Training Loss', color='blue')
            axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', color='red')
            axes[0, 0].set_title('Model Loss During Training')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Training History - Accuracy
            axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
            axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
            axes[0, 1].set_title('Model Accuracy During Training')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Training History - AUC
            if 'auc' in history.history:
                axes[0, 2].plot(history.history['auc'], label='Training AUC', color='blue')
                axes[0, 2].plot(history.history['val_auc'], label='Validation AUC', color='red')
                axes[0, 2].set_title('Model AUC During Training')
                axes[0, 2].set_xlabel('Epoch')
                axes[0, 2].set_ylabel('AUC')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Confusion Matrix
            cm = confusion_matrix(y_test, y_test_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_title('Confusion Matrix (Test Set)')
            axes[1, 0].set_ylabel('Actual')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_xticklabels(['No Response', 'Response'])
            axes[1, 0].set_yticklabels(['No Response', 'Response'])
            
            # 5. ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_auc = auc(fpr, tpr)
            axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, 1].set_xlim([0.0, 1.0])
            axes[1, 1].set_ylim([0.0, 1.05])
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title('ROC Curve')
            axes[1, 1].legend(loc="lower right")
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Performance Metrics Bar Chart
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            
            # Calculate specificity
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'F1-Score': f1,
                'AUC': roc_auc
            }
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            bars = axes[1, 2].bar(range(len(metrics)), list(metrics.values()), color=colors)
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].set_title('Test Set Performance Metrics')
            axes[1, 2].set_ylim([0, 1])
            axes[1, 2].set_xticks(range(len(metrics)))
            axes[1, 2].set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
            axes[1, 2].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (key, value) in enumerate(metrics.items()):
                axes[1, 2].text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Print comprehensive metrics
            print("\n" + "="*60)
            print("MLP DEEP LEARNING MODEL - COMPREHENSIVE RESULTS")
            print("="*60)
            print(f"Test Sample Size: {len(y_test)} patients")
            print(f"Positive Cases: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
            print(f"Negative Cases: {len(y_test)-sum(y_test)} ({(len(y_test)-sum(y_test))/len(y_test)*100:.1f}%)")
            
            print(f"\nConfusion Matrix:")
            print(f"True Negatives:  {tn:4d}    False Positives: {fp:4d}")
            print(f"False Negatives: {fn:4d}    True Positives:  {tp:4d}")
            
            print(f"\nFINAL PERFORMANCE METRICS:")
            for metric, value in metrics.items():
                print(f"{metric:12s}: {value:.4f}")
            
            print(f"\nTraining Epochs Completed: {len(history.history['loss'])}")
            print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
            print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
            print("="*60)
            
            # Save visualization
            filename = 'mlp_training_results.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"MLP training visualization saved as '{filename}'")
            
            if show:
                try:
                    plt.show()
                except:
                    # If interactive display fails, try to open the saved file
                    print("Interactive display not available. Opening saved image...")
                    try:
                        import subprocess
                        import sys
                        if sys.platform == "win32":
                            os.startfile(filename)
                        elif sys.platform == "darwin":
                            subprocess.run(["open", filename])
                        else:
                            subprocess.run(["xdg-open", filename])
                    except:
                        print(f"Please manually open '{filename}' to view the visualization.")
            else:
                plt.close()
                
        except Exception as e:
            print(f"Warning: Could not generate MLP visualization - {e}")

    def predict(self, patient_data, drug_name):
        """Make a prediction for a single patient using comprehensive clinical features."""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        if not DEEP_LEARNING_AVAILABLE:
            print("ERROR: TensorFlow not available for MLP prediction")
            return None

        # Create a single-row DataFrame with the patient's data
        # Use the same feature structure as our comprehensive dummy data
        patient_df = pd.DataFrame([{
            # Demographics
            'age': patient_data.get('age', 60),
            'sex': patient_data.get('sex', 'Male'),
            'race': patient_data.get('race', 'White'),
            'bmi': patient_data.get('bmi', 25.0),
            'insurance': patient_data.get('insurance', 'Private'),
            
            # Clinical Characteristics
            'histology_type': patient_data.get('histology_type', 'Adenocarcinoma'),
            'cancer_stage': patient_data.get('cancer_stage', 'IIIA'),
            'ecog_performance': patient_data.get('ecog_performance', 1),
            'tumor_size': patient_data.get('tumor_size', 4.0),
            'metastatic_sites': patient_data.get('metastatic_sites', 2),
            'tumor_location': patient_data.get('tumor_location', 'Upper_lobe'),
            'pleural_effusion': patient_data.get('pleural_effusion', 0),
            'weight_loss_pct': patient_data.get('weight_loss_pct', 5.0),
            'time_since_diagnosis': patient_data.get('time_since_diagnosis', 30),
            'karnofsky_score': patient_data.get('karnofsky_score', 80),
            
            # Laboratory Values
            'wbc_count': patient_data.get('wbc_count', 7.0),
            'hemoglobin': patient_data.get('hemoglobin', 12.0),
            'platelet_count': patient_data.get('platelet_count', 280),
            'albumin': patient_data.get('albumin', 3.5),
            'ldh': patient_data.get('ldh', 200),
            'creatinine': patient_data.get('creatinine', 1.0),
            'bilirubin': patient_data.get('bilirubin', 1.0),
            'alt': patient_data.get('alt', 30),
            'ast': patient_data.get('ast', 35),
            'crp': patient_data.get('crp', 5.0),
            'cea_level': patient_data.get('cea_level', 3.0),
            'calcium': patient_data.get('calcium', 9.8),
            'glucose': patient_data.get('glucose', 110),
            'neutrophil_pct': patient_data.get('neutrophil_pct', 65),
            'lymphocyte_pct': patient_data.get('lymphocyte_pct', 25),
            
            # Genetic/Molecular Markers
            'egfr_mutation': patient_data.get('egfr_mutation', 'Wild_type'),
            'kras_mutation': patient_data.get('kras_mutation', 0),
            'alk_rearrangement': patient_data.get('alk_rearrangement', 0),
            'pdl1_expression': patient_data.get('pdl1_expression', 10.0),
            'tp53_mutation': patient_data.get('tp53_mutation', 0),
            'ros1_rearrangement': patient_data.get('ros1_rearrangement', 0),
            'braf_mutation': patient_data.get('braf_mutation', 0),
            'pik3ca_mutation': patient_data.get('pik3ca_mutation', 0),
            
            # Treatment History
            'previous_chemo_lines': patient_data.get('previous_chemo_lines', 1),
            'previous_radiation': patient_data.get('previous_radiation', 0),
            'previous_immunotherapy': patient_data.get('previous_immunotherapy', 0),
            'previous_surgery': patient_data.get('previous_surgery', 'Biopsy'),
            'concurrent_medications': patient_data.get('concurrent_medications', 5),
            'treatment_delay_days': patient_data.get('treatment_delay_days', 15),
            
            # Risk Factors
            'smoking_pack_years': patient_data.get('smoking_pack_years', 30),
            'smoking_status': patient_data.get('smoking_status', 'Former_gt5yr'),
            'copd_severity': patient_data.get('copd_severity', 'None'),
            'cardiovascular_disease': patient_data.get('cardiovascular_disease', 0),
            'diabetes_type': patient_data.get('diabetes_type', 'None'),
            'other_cancer_history': patient_data.get('other_cancer_history', 0),
            'family_cancer_history': patient_data.get('family_cancer_history', 0),
            'alcohol_use': patient_data.get('alcohol_use', 'Moderate')
        }])

        # Preprocess using the same pipeline as training
        try:
            X_processed, _ = self.preprocess_features(patient_df, training=False)
            
            # Make prediction
            probability = self.model.predict(X_processed)[0][0]  # MLP returns single probability
            
            # Apply conservative threshold for medical safety
            conservative_prediction = 1 if probability >= self.prediction_threshold else 0
            
            # Skip drug info fetch to prevent hanging (handled by Gradio interface)
            drug_info = {'name': drug_name, 'status': 'Available via web interface'}
            
            # Determine confidence level based on probability
            if probability >= 0.8 or probability <= 0.2:
                confidence_level = "High"
            elif probability >= 0.7 or probability <= 0.3:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            return {
                'prediction': 'Positive response expected' if conservative_prediction == 1 else 'Negative response expected',
                'confidence': f"{probability*100:.1f}%",
                'confidence_level': confidence_level,
                'threshold_used': f"{self.prediction_threshold:.1f}",
                'probability_positive': probability,
                'probability_negative': 1 - probability,
                'conservative_prediction': conservative_prediction,
                'drug_info': drug_info,
                'model_type': 'MLP Deep Learning'
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'prediction': 'Error in prediction',
                'confidence': '0.0%',
                'confidence_level': 'Low',
                'error': str(e)
            }


def main():
    """Main program loop with interactive menu for MLP Deep Learning model."""
    predictor = LungCancerMLPPredictor()
    
    while True:
        print("\n" + "="*50)
        print("LUNG CANCER MLP PREDICTOR")
        print("="*50)
        print("1. Train MLP model")
        print("2. Make prediction")
        print("3. Get drug information")
        print("4. View training & performance visualization") 
        print("5. Exit")
        print("="*50)
        print("Info: Uses deep learning MLP with 35K patients, 52 features")
        print("User Input: Only 13 essential features required (39 use defaults)")
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
        except EOFError:
            print("Automated run completed!")
            break
        
        if choice == '1':
            # Train MLP model
            print("\nTraining MLP model with comprehensive dataset...")
            print(f"Configuration: {DEFAULT_N_PATIENTS:,} patients, 52 features, {DEFAULT_EPOCHS} epochs")
            success = predictor.train(n_patients=DEFAULT_N_PATIENTS, epochs=DEFAULT_EPOCHS)
            if success:
                print("\nTraining completed!")
            
        elif choice == '2':
            # Make prediction
            if predictor.model is None:
                print("\nPlease train the MLP model first!")
                continue
                
            print("\n" + "="*50)
            print("PATIENT DATA INPUT")
            print("="*50)
            print("Enter patient data (press Enter for default values):")
            
            try:
                patient_data = {}
                
                # Demographics
                print("\nDemographics:")
                patient_data['age'] = float(input("Age [60]: ") or "60")
                patient_data['sex'] = input("Sex (Male/Female) [Male]: ") or "Male"
                patient_data['race'] = input("Race (White/Black/Asian/Hispanic/Other) [White]: ") or "White"
                patient_data['bmi'] = float(input("BMI [25.0]: ") or "25.0")
                
                # Clinical characteristics
                print("\nClinical Characteristics:")
                patient_data['histology_type'] = input("Histology (Adenocarcinoma/Squamous/Small_cell/Large_cell/Other) [Adenocarcinoma]: ") or "Adenocarcinoma"
                patient_data['cancer_stage'] = input("Cancer Stage (IA/IB/IIA/IIB/IIIA/IIIB/IV) [IIIA]: ") or "IIIA"
                patient_data['ecog_performance'] = int(input("ECOG Performance (0-4) [1]: ") or "1")
                patient_data['tumor_size'] = float(input("Tumor size (cm) [4.0]: ") or "4.0")
                
                # Laboratory values
                print("\nKey Laboratory Values:")
                patient_data['hemoglobin'] = float(input("Hemoglobin (g/dL) [12.0]: ") or "12.0")  
                patient_data['ldh'] = float(input("LDH (U/L) [200]: ") or "200")
                patient_data['albumin'] = float(input("Albumin (g/dL) [3.5]: ") or "3.5")
                
                # Genetic markers
                print("\nGenetic Markers:")
                patient_data['egfr_mutation'] = input("EGFR Mutation (Wild_type/Exon19del/L858R/Other) [Wild_type]: ") or "Wild_type"
                patient_data['kras_mutation'] = int(input("KRAS Mutation (0/1) [0]: ") or "0")
                
                # Get drug name
                drug_name = input("\nChemotherapy drug name [Cisplatin]: ") or "Cisplatin"
                
                # Make prediction
                print("\nAnalyzing patient data with MLP model...")
                result = predictor.predict(patient_data, drug_name)
                
                if result and 'error' not in result:
                    print("\n" + "="*50)
                    print("MLP PREDICTION RESULTS")
                    print("="*50)
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']} ({result['confidence_level']} confidence)")
                    print(f"Decision Threshold: {result['threshold_used']}")
                    print(f"Model Type: {result.get('model_type', 'MLP Deep Learning')}")
                    print(f"Positive Response Probability: {result['probability_positive']:.3f}")
                    print(f"Negative Response Probability: {result['probability_negative']:.3f}")
                    
                    if 'drug_info' in result and result['drug_info']:
                        print(f"\nDrug Information:")
                        drug_info = result['drug_info']
                        print(f"   Name: {drug_info.get('name', 'N/A')}")
                        if drug_info.get('warnings'):
                            print(f"   Warnings: {drug_info['warnings'][0][:200] if drug_info['warnings'] else 'None'}...")
                else:
                    print(f"\nPrediction error: {result.get('error', 'Unknown error')}")
                    
            except ValueError as e:
                print(f"\nInvalid input: {e}")
            except Exception as e:
                print(f"\nError: {e}")
                
        elif choice == '3':
            # Get drug information
            drug_name = input("\nEnter drug name: ").strip()
            if drug_name:
                print(f"\nLooking up information for '{drug_name}'...")
                drug_info = predictor.fetch_drug_info(drug_name)
                
                if drug_info and drug_info.get('name'):
                    print("\n" + "="*50)
                    print("DRUG INFORMATION")
                    print("="*50)
                    print(f"Name: {drug_info['name']}")
                    if drug_info.get('indications_and_usage'):
                        print(f"Indications: {drug_info['indications_and_usage'][0][:200] if drug_info['indications_and_usage'] else 'N/A'}...")
                    if drug_info.get('warnings'):
                        print(f"Warnings: {drug_info['warnings'][0][:200] if drug_info['warnings'] else 'N/A'}...")
                else:
                    print("No information found for this drug.")
            else:
                print("Please enter a valid drug name.")
                
        elif choice == '4':
            # View training visualization
            if predictor.model is None:
                print("\nPlease train the MLP model first!")
                continue
                
            if hasattr(predictor, 'training_history') and predictor.training_history:
                print("\nDisplaying MLP training results and performance metrics...")
                predictor.visualize_mlp_results(
                    predictor.X_test_scaled, 
                    predictor.y_test, 
                    type('History', (), {'history': predictor.training_history})(), 
                    show=True
                )
            else:
                print("\nNo training history available. Please train the model first!")
                
        elif choice == '5':
            # Exit
            print("\nThank you for using the Lung Cancer MLP Predictor!")
            print("Model: Multi-Layer Perceptron with clinical features")
            print("Dataset: 35K patients, 52 features (13 inputs + 39 defaults)")
            print("Deep Learning: TensorFlow/Keras with preprocessing")
            break
            
        else:
            print("\nInvalid choice. Please enter 1-5.")
            
        try:
            input("\nPress Enter to continue...")
        except EOFError:
            print("Continuing...")
            continue


if __name__ == "__main__":
    """Program entry point."""
    print("\n" + "="*70)
    print("LUNG CANCER MLP PREDICTOR")
    print("Multi-Layer Perceptron with Comprehensive Clinical Features")  
    print("="*70)
    
    print("\n💡 This system uses:")
    print("   • Deep Learning: Multi-Layer Perceptron (MLP) neural network")
    print("   • Dataset: 35,000 dummy patients with 52 clinical features")
    print("   • Framework: TensorFlow/Keras with advanced preprocessing")
    print("   • Features: Demographics, clinical, lab, genetic, treatment history")
    
    if not DEEP_LEARNING_AVAILABLE:
        print("\n⚠️  WARNING: TensorFlow not available!")
        print("   Install with: pip install tensorflow")
        print("   Some features will be limited without TensorFlow\n")
    else:
        print("\n✅ TensorFlow available - Full MLP functionality enabled\n")
    
    main()
