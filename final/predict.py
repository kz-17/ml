import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import requests
import time

FLASK_URL = "http://127.0.0.1:5000/update"


class HandGestureMLP(nn.Module):
    def __init__(self, input_size=63, hidden1=128, hidden2=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_model(model_path="hand_gesture_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandGestureMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def to_row_norm(landmarks, w, h):
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
    wrist = pts[0]
    pts[:, :2] -= wrist[:2]
    pts[:, 2] -= np.mean(pts[:, 2])
    return pts.flatten().tolist()


def predict():
    model, device = load_model()
    label_map = {0: "open", 1: "close"}

    mp_hands = mp.solutions.hands
    draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    last_sent = 0
    cooldown = 0.5

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            h, w, _ = frame.shape

            current_label = "none"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                row = to_row_norm(hand.landmark, w, h)
                input_tensor = torch.tensor([row], dtype=torch.float32).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    _, pred = torch.max(output, 1)
                    current_label = label_map[pred.item()]

                color = (0, 255, 0) if current_label == "open" else (0, 0, 255)
                cv2.putText(frame, f"Gesture: {current_label}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                now = time.time()
                if now - last_sent > cooldown:
                    try:
                        requests.post(FLASK_URL, json={"gesture": current_label}, timeout=0.1)
                        last_sent = now
                    except requests.exceptions.RequestException:
                        pass

            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict()
