import cv2
import mediapipe as mp
import csv
import os
import numpy as np

SAVE = "dataset"
os.makedirs(SAVE, exist_ok=True)
labels = ["open", "close"]

mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils

def to_row_norm(landmarks, w, h):
    pts = np.array([[lm.x*w, lm.y*h, lm.z*w] for lm in landmarks], dtype=np.float32)
    wrist = pts[0]
    pts[:, :2] -= wrist[:2]
    pts[:, 2] -= np.mean(pts[:, 2])
    return pts.flatten().tolist()

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    cap = cv2.VideoCapture(0)
    current_label = None
    recording = False
    file = None
    writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w, _ = frame.shape
        label_text = f"Label: {current_label or '-'} | Rec: {recording}"
        cv2.putText(frame, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if recording else (0, 0, 255), 2)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            row = to_row_norm(hand.landmark, w, h)
            if recording and current_label and row is not None:
                row.append(current_label)
                writer.writerow(row)

        cv2.imshow("1", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            if file:
                file.close()
            break
        elif k == ord('o'):
            current_label = "open"
            file = open(os.path.join(SAVE, f"{current_label}.csv"), "a", newline="")
            writer = csv.writer(file)
            recording = True
        elif k == ord('c'):
            current_label = "close"
            file = open(os.path.join(SAVE, f"{current_label}.csv"), "a", newline="")
            writer = csv.writer(file)
            recording = True
        elif k == ord(' '):
            recording = not recording

cap.release()
cv2.destroyAllWindows()

files = glob.glob("dataset/*.csv")
dfs = [pd.read_csv(f, header=None) for f in files]
df = pd.concat(dfs, ignore_index=True)
df = df.dropna()
df.to_csv("dataset.csv", index=False)
