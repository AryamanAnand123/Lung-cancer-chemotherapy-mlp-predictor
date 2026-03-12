import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CancerMLP import LungCancerMLPPredictor

# Initialize predictor
print("Loading MLP model...")
predictor = LungCancerMLPPredictor()

def get_drug_info_safe(drug_name):
    """Safely fetch drug information with timeout and error handling."""
    try:
        import requests
        
        # FDA API configuration
        FDA_API_KEY = "XnBhtWS9u7fMNOTqDdHUQQUzLkdafHTiDvRfOqcn"
        base_url = "https://api.fda.gov/drug/label.json"
        
        params = {
            'api_key': FDA_API_KEY,
            'search': f'openfda.generic_name:"{drug_name.lower()}"',
            'limit': 1
        }
        
        # Quick timeout to prevent hanging
        response = requests.get(base_url, params=params, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                result = data['results'][0]
                return {
                    'name': result.get('openfda', {}).get('generic_name', [drug_name])[0] if result.get('openfda', {}).get('generic_name') else drug_name,
                    'brand_names': result.get('openfda', {}).get('brand_name', []),
                    'manufacturer': result.get('openfda', {}).get('manufacturer_name', []),
                    'indications': result.get('indications_and_usage', [])[:1] if result.get('indications_and_usage') else [],
                    'warnings': result.get('boxed_warning', [])[:1] if result.get('boxed_warning') else []
                }
        return None
        
    except Exception as e:
        print(f"Drug API error: {e}")
        return None

def format_drug_info(drug_name, drug_info):
    """Format drug information for display."""
    if not drug_info:
        return f"Drug Information ({drug_name}): API temporarily unavailable"
    
    lines = [f"Drug Information ({drug_info['name'].upper()}):"]
    
    # Brand names
    if drug_info.get('brand_names'):
        brands = ', '.join(drug_info['brand_names'][:3])  # Limit to 3 brands
        lines.append(f"• Brand Names: {brands}")
    
    # Manufacturer
    if drug_info.get('manufacturer'):
        manufacturer = drug_info['manufacturer'][0] if drug_info['manufacturer'] else 'N/A'
        lines.append(f"• Manufacturer: {manufacturer}")
    
    # Indications (truncated)
    if drug_info.get('indications'):
        indication = drug_info['indications'][0][:200] + "..." if len(drug_info['indications'][0]) > 200 else drug_info['indications'][0]
        lines.append(f"• Indications: {indication}")
    
    # Warnings (truncated)  
    if drug_info.get('warnings'):
        warning = drug_info['warnings'][0][:150] + "..." if len(drug_info['warnings'][0]) > 150 else drug_info['warnings'][0]
        lines.append(f"• Warning: {warning}")
    
    return '\n'.join(lines)

def predict_cancer_response(age, sex, race, bmi, histology, stage, ecog, tumor_size, 
                           hemoglobin, ldh, albumin, egfr_mutation, kras_mutation, drug_name):
    """Make prediction using the MLP model with the 13 essential features."""
    try:
        # Prepare patient data with the 13 essential features
        patient_data = {
            # Demographics (4 features)
            'age': age,
            'sex': sex,
            'race': race,
            'bmi': bmi,
            
            # Clinical Characteristics (4 features)
            'histology_type': histology,
            'cancer_stage': stage,
            'ecog_performance': ecog,
            'tumor_size': tumor_size,
            
            # Laboratory Values (3 features)
            'hemoglobin': hemoglobin,
            'ldh': ldh,
            'albumin': albumin,
            
            # Genetic Markers (2 features)
            'egfr_mutation': egfr_mutation,
            'kras_mutation': kras_mutation
        }
        
        # Get drug information first (non-blocking)
        drug_info = get_drug_info_safe(drug_name)
        
        # Make prediction
        result = predictor.predict(patient_data, drug_name)
        
        if result:
            # Format results using correct keys from CancerMLP
            probability = result.get('probability_positive', 0.0)  # Use correct key
            prediction = result.get('prediction', 'Unknown')
            confidence = result.get('confidence', '0.0%')
            confidence_level = result.get('confidence_level', 'Low')
            
            # Determine response status
            response_status = "Likely to Respond" if "Positive" in prediction else "Unlikely to Respond"
            
            # Format drug information
            drug_section = format_drug_info(drug_name, drug_info)
            
            # Create formatted output
            output = f"""
PREDICTION RESULTS

Response Probability: {probability:.1%}
Prediction: {response_status}
Confidence: {confidence} ({confidence_level})
Threshold Used: {result.get('threshold_used', '0.7')}

Patient Summary:
• Demographics: {age}y {sex} {race}, BMI {bmi}
• Cancer: {histology}, Stage {stage}, ECOG {ecog}
• Labs: Hgb {hemoglobin}, LDH {ldh}, Albumin {albumin}
• Genetics: EGFR {egfr_mutation}, KRAS {kras_mutation}
• Treatment: {drug_name}

{drug_section}

Model: {result.get('model_type', 'MLP Deep Learning')}

Disclaimer: This prediction is for research purposes only.
Consult qualified oncologists for treatment decisions.
            """
            return output
        else:
            return "Error: Could not make prediction. Please check your inputs and ensure the model is properly trained."
            
    except Exception as e:
        error_msg = str(e)
        return f"""
PREDICTION ERROR

Error Details: {error_msg}

Troubleshooting:
• Ensure the MLP model is trained
• Check that all input values are within valid ranges
• Verify TensorFlow is properly installed

Support: Check the console for detailed error information
        """

# Create Gradio interface
with gr.Blocks(title="Lung Cancer MLP Predictor") as interface:
    
    gr.Markdown("""
    # Lung Cancer MLP Predictor
    ### Deep Learning Model for Chemotherapy Response Prediction
    
    This system uses a Multi-Layer Perceptron neural network trained on clinical data 
    to predict treatment response probability. **13 Essential Features** are required:
    
    **Demographics (4):** Age, Sex, Race, BMI  
    **Clinical (4):** Histology, Stage, ECOG, Tumor Size  
    **Laboratory (3):** Hemoglobin, LDH, Albumin  
    **Genetic (2):** EGFR Mutation, KRAS Mutation  
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Patient Demographics")
            age = gr.Slider(
                minimum=18, maximum=90, value=65, step=1,
                label="Age (years)",
                info="Patient age at diagnosis"
            )
            sex = gr.Dropdown(
                choices=["Male", "Female"], value="Male",
                label="Sex"
            )
            race = gr.Dropdown(
                choices=["White", "Black", "Asian", "Hispanic", "Other"], 
                value="White",
                label="Race/Ethnicity"
            )
            bmi = gr.Slider(
                minimum=15.0, maximum=45.0, value=25.0, step=0.5,
                label="BMI (Body Mass Index)",
                info="kg/m²"
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### Clinical Information")
            histology = gr.Dropdown(
                choices=["Adenocarcinoma", "Squamous", "Small_cell", "Large_cell", "Other"],
                value="Adenocarcinoma",
                label="Histology Type",
                info="Type of lung cancer"
            )
            stage = gr.Dropdown(
                choices=["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"],
                value="IIIA",
                label="Cancer Stage",
                info="TNM staging system"
            )
            ecog = gr.Dropdown(
                choices=[0, 1, 2, 3, 4], value=1,
                label="ECOG Performance Status",
                info="0=Fully active, 1=Restricted, 2=Ambulatory 50%+, 3=Limited, 4=Bedridden"
            )
            tumor_size = gr.Slider(
                minimum=0.5, maximum=15.0, value=4.0, step=0.1,
                label="Tumor Size (cm)",
                info="Largest diameter"
            )
            
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### � Laboratory Values")
            hemoglobin = gr.Slider(
                minimum=6.0, maximum=18.0, value=12.0, step=0.1,
                label="Hemoglobin (g/dL)",
                info="Normal: 12-16 g/dL"
            )
            ldh = gr.Slider(
                minimum=100, maximum=2000, value=200, step=10,
                label="LDH Lactate Dehydrogenase (U/L)",
                info="Normal: 140-280 U/L, Elevated >300 indicates aggressive disease"
            )
            albumin = gr.Slider(
                minimum=2.0, maximum=5.0, value=3.5, step=0.1,
                label="Albumin (g/dL)",
                info="Normal: 3.5-5.0 g/dL, Low <3.0 indicates poor prognosis"
            )
            
            gr.Markdown("### Genetic Markers")
            egfr_mutation = gr.Dropdown(
                choices=["Wild_type", "Exon19del", "L858R", "Other"],
                value="Wild_type",
                label="EGFR Mutation Status",
                info="Critical for targeted therapy selection"
            )
            kras_mutation = gr.Dropdown(
                choices=[0, 1], value=0,
                label="KRAS Mutation Status",
                info="0=Wild-type, 1=Mutated (resistance marker)"
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### Treatment")
            drug_name = gr.Textbox(
                value="carboplatin",
                label="Drug Name",
                info="Chemotherapy drug being evaluated"
            )
            
            gr.Markdown("### Generate Prediction")
            predict_btn = gr.Button(
                "Predict Treatment Response",
                variant="primary",
                size="lg"
            )
    
    # Output section
    gr.Markdown("### Results")
    output = gr.Textbox(
        label="Prediction Results",
        lines=15,
        max_lines=20
    )
    
    # Connect the prediction function
    predict_btn.click(
        fn=predict_cancer_response,
        inputs=[age, sex, race, bmi, histology, stage, ecog, tumor_size,
                hemoglobin, ldh, albumin, egfr_mutation, kras_mutation, drug_name],
        outputs=output
    )
    
    # Example cases
    gr.Markdown("""
    ### Example Cases
    Try these sample patients to see how the model works:
    """)
    
    with gr.Row():
        good_prognosis_btn = gr.Button("Good Prognosis Patient", size="sm")
        poor_prognosis_btn = gr.Button("Poor Prognosis Patient", size="sm")
        
    def set_good_prognosis():
        return (45, "Female", "Asian", 22.0, "Adenocarcinoma", "IA", 0, 2.0, 14.0, 180, 4.0, "Exon19del", 0, "erlotinib")
        
    def set_poor_prognosis():
        return (75, "Male", "White", 28.0, "Small_cell", "IV", 3, 8.0, 9.5, 450, 2.8, "Wild_type", 1, "carboplatin")
    
    good_prognosis_btn.click(
        fn=set_good_prognosis,
        outputs=[age, sex, race, bmi, histology, stage, ecog, tumor_size,
                hemoglobin, ldh, albumin, egfr_mutation, kras_mutation, drug_name]
    )
    
    poor_prognosis_btn.click(
        fn=set_poor_prognosis,
        outputs=[age, sex, race, bmi, histology, stage, ecog, tumor_size,
                hemoglobin, ldh, albumin, egfr_mutation, kras_mutation, drug_name]
    )

if __name__ == "__main__":
    print("Starting Gradio interface...")
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False
    )