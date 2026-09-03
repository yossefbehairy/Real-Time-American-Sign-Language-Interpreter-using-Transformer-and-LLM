import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import json
import math

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=135):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# === Transformer Model ===
class Model(nn.Module):
    def __init__(self, num_embed, d_model, max_len, n_heads, num_encoders, num_classes, dropout, activation, batch_first=True):
        super(Model, self).__init__()
        self.embed = nn.ModuleList()
        self.embed.append(nn.Linear(d_model, d_model * 2))
        self.embed.append(nn.LayerNorm(d_model * 2))
        self.embed.append(nn.ReLU(inplace=True))
        for _ in range(num_embed - 2):
            self.embed.append(nn.Linear(d_model * 2, d_model * 2))
            self.embed.append(nn.LayerNorm(d_model * 2))
            self.embed.append(nn.ReLU(inplace=True))
        self.embed.append(nn.Linear(d_model * 2, d_model))
        self.embed.append(nn.LayerNorm(d_model))
        self.embed.append(nn.ReLU(inplace=True))

        self.positionalEncoder = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.cls_embedding = nn.Parameter(torch.zeros((1, d_model)))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*2,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.output = nn.Linear(d_model, num_classes)

    def forward(self, x):
        for layer in self.embed:
            x = layer(x)
        x = self.positionalEncoder(x)
        x = x + self.cls_embedding
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.output(x)
        return x

# === Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(
    num_embed=4,
    d_model=176,
    max_len=135,
    n_heads=4,
    num_encoders=2,
    num_classes=250,
    dropout=0.11073790254354612,
    activation='relu',
    batch_first=True
).to(device)

state_dict = torch.load("best_model_final_epoch400.pth", map_location=device)
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if k in model.state_dict() and model.state_dict()[k].shape == v.shape
}
model.load_state_dict(filtered_state_dict, strict=False)
#model.load_state_dict(state_dict, strict=False)
model.eval()

input_proj = nn.Linear(543 * 3, 135 * 176).to(device)

# Load label mapping
with open("sign_to_prediction_index_map.json", "r") as f:
    word_to_class = json.load(f)
class_to_word = {v: k for k, v in word_to_class.items()}

# === MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    refine_face_landmarks=True
)

def extract_landmarks(results):
    def get_landmarks(landmarks, count):
        return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark] if landmarks else [[0, 0, 0]] * count
    all_landmarks = []
    all_landmarks += get_landmarks(results.face_landmarks, 468)
    all_landmarks += get_landmarks(results.pose_landmarks, 33)
    all_landmarks += get_landmarks(results.left_hand_landmarks, 21)
    all_landmarks += get_landmarks(results.right_hand_landmarks, 21)
    return np.array(all_landmarks[:543], dtype=np.float32)

# === Main Real-time Loop ===
cap = cv2.VideoCapture(0)
collected_words = []
max_signs = 5

while cap.isOpened() and len(collected_words) < max_signs:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # Draw MediaPipe Landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Display instructions
    instruction = "Press SPACE to capture sign, Q to quit"
    cv2.putText(frame, instruction, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 250), 2)

    # Show captured words
    for i, word in enumerate(collected_words):
        cv2.putText(frame, f"{i+1}: {word}", (10, 70 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    cv2.imshow("Real-Time GSLR Inference (Press SPACE)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\n🛑 Exiting - User pressed Q.")
        break

    if key == 32:  # SPACE key pressed
        if (results.left_hand_landmarks or results.right_hand_landmarks):
            landmarks = extract_landmarks(results)

            if landmarks.shape == (543, 3):
                landmarks_flat = landmarks.flatten()
                landmarks_tensor = torch.tensor(landmarks_flat, dtype=torch.float32).unsqueeze(0).to(device)
                projected_input = input_proj(landmarks_tensor)
                projected_input = projected_input.view(1, 135, 176)

                with torch.no_grad():
                    output = model(projected_input)
                    predicted_class = output.argmax(dim=1).item()
                word = class_to_word.get(predicted_class, "Unknown")

                if word != "Unknown":
                    collected_words.append(word)
                    print(f"✅ Word captured: {word}")
        else:
            print("⚠️ No hands detected. Show hands before pressing SPACE.")

cap.release()
cv2.destroyAllWindows()

print("\n📋 Final Predicted Words:")
print(collected_words)
