# Lung Cancer Treatment Response Prediction - Deep Learning MLP

🧠 **Advanced Multi-Layer Perceptron (MLP) system for predicting lung cancer chemotherapy treatment response using deep learning.**

## 🎯 Project Overview

This project implements a sophisticated neural network model to predict whether lung cancer patients will respond positively to chemotherapy treatment. Built with TensorFlow/Keras, the system uses 13 essential clinical features and achieves 73-78% accuracy.

## 🚀 Features

- **Deep Learning MLP Architecture**: 52→128→96→64→32→16→1 neurons with advanced optimization
- **Professional Web Interface**: Clean Gradio interface for clinical users
- **FDA Drug Information**: Integrated drug lookup with safety information
- **Comprehensive Documentation**: Detailed technical and user guides
- **Synthetic Data Generation**: Creates realistic medical correlations for training
- **Conservative Predictions**: Optimized for medical safety with threshold-based predictions

## 📊 Model Performance

- **Accuracy**: 73-78%
- **Architecture**: 6-layer MLP with batch normalization and dropout
- **Optimizer**: AdamW with learning rate scheduling
- **Training Data**: 35,000 synthetic patients with medical correlations
- **Features**: 13 essential clinical parameters

## 🏥 Clinical Features Used

1. **Patient Demographics**: Age, Gender, BMI
2. **Disease Characteristics**: Tumor Size, Cancer Stage, Histology Type
3. **Treatment History**: Previous Treatments, Radiation History
4. **Health Status**: Performance Status, Comorbidity Index
5. **Laboratory Values**: White Blood Cell Count, Hemoglobin Level
6. **Lifestyle Factors**: Smoking Status, Years of Smoking

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/AryamanAnand123/lung-cancer-mlp-prediction.git
   cd lung-cancer-mlp-prediction
   ```

2. **Install dependencies**:

   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn gradio requests joblib
   ```

3. **Run the web interface**:

   ```bash
   python gradio_interface.py
   ```

4. **Access the application**: Open http://127.0.0.1:7860 in your browser

## 💻 Usage

### Web Interface

- Launch `gradio_interface.py` for the professional clinical interface
- Input patient parameters using the form fields
- Get treatment response predictions with confidence scores
- View FDA drug information and safety data

### Command Line

```python
from CancerMLP import LungCancerMLPPredictor

# Initialize predictor
predictor = LungCancerMLPPredictor()

# Train the model (optional - pre-trained model included)
predictor.train_model()

# Make predictions
patient_data = {
    'age': 65,
    'gender': 'Male',
    'bmi': 24.5,
    # ... other features
}
result = predictor.predict_treatment_response(patient_data)
```

## 📁 Project Structure

```
├── CancerMLP.py                     # Main MLP deep learning system
├── gradio_interface.py              # Professional web interface
├── lung_cancer_mlp_model.h5         # Pre-trained neural network model
├── lung_cancer_encoders.joblib      # Feature encoders
├── lung_cancer_scaler.joblib        # Data preprocessing scaler
├── MLP_Code_Documentation.txt       # Technical implementation guide
├── Gradio_Code_Documentation.txt    # Web interface documentation
├── Essential_13_Features_Guide.txt  # Clinical features reference
├── Research_Paper_Draft_IEEE.tex    # IEEE format research paper
└── mlp_research_metrics_report.txt  # Performance analysis
```

## 🔬 Technical Details

### Neural Network Architecture

- **Input Layer**: 52 features (13 essential + 39 defaults)
- **Hidden Layers**: 128→96→64→32→16 neurons
- **Output Layer**: 1 neuron (treatment response probability)
- **Activation Functions**: ReLU (hidden), Sigmoid (output)
- **Regularization**: Batch normalization, Dropout (0.3-0.5)

### Training Configuration

- **Optimizer**: AdamW (adaptive learning rate)
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Early Stopping**: Patience=15, monitor validation loss
- **Batch Size**: 128, Epochs: 100 (with early stopping)

## 📈 Research Applications

This system is designed for:

- Clinical decision support research
- Treatment response prediction studies
- Medical AI algorithm development
- Healthcare optimization research

**⚠️ Important**: This tool is for research purposes only and should not be used for actual medical diagnosis or treatment decisions without proper clinical validation.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Aryaman Anand** - [AryamanAnand123](https://github.com/AryamanAnand123)

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Gradio team for the web interface framework
- FDA OpenFDA API for drug information integration
- Medical research community for clinical feature insights

---

**📧 Contact**: Aryamananand24@gmail.com

**🌟 If this project helps your research, please star this repository!**
