import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# -----------------------------------------------------------------
# 1. 3-CLASS MULTI-ARMED MODEL ARCHITECTURE
# -----------------------------------------------------------------
class MultiClassMultiArmedPredictor(nn.Module):
    def __init__(self, tabular_features_dim=5, num_classes=3):
        super(MultiClassMultiArmedPredictor, self).__init__()

        # Arm 1: 1D CNN Sequence Branch
        self.seq_net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Arm 2: Dense MLP Tabular Branch
        self.tab_net = nn.Sequential(
            nn.Linear(tabular_features_dim, 16),
            nn.ReLU()
        )

        # Fusion Head -> Outputs 3 Raw Logits
        self.fc = nn.Sequential(
            nn.Linear(32 + 16, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x_seq, x_tab):
        x_seq = x_seq.reshape(-1, 4, 2)
        x_seq = self.seq_net(x_seq)
        x_tab = self.tab_net(x_tab)
        x_combined = torch.cat((x_seq, x_tab), dim=1)
        return self.fc(x_combined)

# Setup device & model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiClassMultiArmedPredictor(tabular_features_dim=5, num_classes=3).to(device)

# Load your 3-class weights file
WEIGHTS_PATH = '3class_classifier.pth'  # or 'arm1:MultiArm.pth'
try:
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f" Successfully loaded 3-Class PyTorch model weights from: {WEIGHTS_PATH}")
except Exception as e:
    print(f"⚠️ Warning: Could not load weight file ({e}). Running demo pass.")

model.eval()

# Class Map
CLASS_NAMES = {0: "Benign", 1: "Pathogenic", 2: "VUS (Uncertain Significance)"}

# -----------------------------------------------------------------
# 2. BARE-BONES MINIMAL UI
# -----------------------------------------------------------------
BASIC_UI = """
<!DOCTYPE html>
<html>
<body>
    <h3>VEP 3-Class Model Tester</h3>
    <p>Paste JSON into the box:</p>
    <textarea id="box" rows="8" cols="45">{
  "ref_allele": "G",
  "alt_allele": "C",
  "number_submitters": 25,
  "origin_simple": "germline",
  "review_stars": 4.0,
  "has_somatic_impact": false,
  "gene_vulnerability": 0.95
}</textarea><br><br>
    <button onclick="test()">Predict</button>
    <pre id="out"></pre>

    <script>
        async function test() {
            const raw = document.getElementById('box').value;
            const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: raw
            });
            const data = await res.json();
            document.getElementById('out').innerText = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
"""

# -----------------------------------------------------------------
# 3. ENDPOINTS
# -----------------------------------------------------------------
@app.route('/')
def home():
    return render_template_string(BASIC_UI)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid JSON"}), 400

    data = request.get_json()

    # Preprocess Sequence Arm (8 features)
    ref_allele = str(data.get('ref_allele', 'A')).upper()
    alt_allele = str(data.get('alt_allele', 'G')).upper()
    bases = ['A', 'C', 'G', 'T']
    
    # FIXED: Group by base channel so reshape(-1, 4, 2) gets [[Ref_A, Alt_A], [Ref_C, Alt_C], ...]
    seq_vector = []
    for base in bases:
        seq_vector.append(1.0 if ref_allele == base else 0.0)
        seq_vector.append(1.0 if alt_allele == base else 0.0)

    # Preprocess Tabular Arm (5 features)
    num_submitters = int(data.get('number_submitters', 1))
    origin_simple = str(data.get('origin_simple', 'germline')).lower()
    review_stars = float(data.get('review_stars', 1.0))
    has_somatic = bool(data.get('has_somatic_impact', False))
    gene_vulnerability = float(data.get('gene_vulnerability', 0.5))

    scaled_submitters = float(np.log1p(num_submitters))
    origin_num = 1.0 if 'germline' in origin_simple else (2.0 if 'somatic' in origin_simple else 0.0)
    somatic_impact = 1.0 if has_somatic else 0.0

    tab_vector = [scaled_submitters, origin_num, review_stars, somatic_impact, gene_vulnerability]

    # Convert to PyTorch Tensors
    x_seq_tensor = torch.tensor([seq_vector], dtype=torch.float32).to(device)
    x_tab_tensor = torch.tensor([tab_vector], dtype=torch.float32).to(device)

    # Run Model Inference
    with torch.no_grad():
        logits = model(x_seq_tensor, x_tab_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_class_id = int(np.argmax(probs))
    predicted_label = CLASS_NAMES[predicted_class_id]

    # PRINT IN TERMINAL / CMD DIRECTLY
    print("\n" + "="*50)
    print(" INCOMING INFERENCE REQUEST")
    print(f" Input: Ref={ref_allele}, Alt={alt_allele}, Stars={review_stars}, Vuln={gene_vulnerability}")
    print(f" PREDICTED CLASS: {predicted_class_id} ({predicted_label})")
    print(f" Class Probabilities: Benign={probs[0]*100:.2f}%, Pathogenic={probs[1]*100:.2f}%, VUS={probs[2]*100:.2f}%")
    print("="*50 + "\n")

    return jsonify({
        "predicted_class_id": predicted_class_id,
        "predicted_class": predicted_label,
        # "class_probabilities": {
        #     "0_Benign": f"{round(float(probs[0]) * 100, 2)}%",
        #     "1_Pathogenic": f"{round(float(probs[1]) * 100, 2)}%",
        #     "2_VUS": f"{round(float(probs[2]) * 100, 2)}%"
        # }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)