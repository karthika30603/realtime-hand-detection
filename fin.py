import cv2
import mediapipe as mp

mp_hand = mp.solutions.hands
hands = mp_hand.Hands()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Extract 3D coordinates
            landmarks_3d = []
            for landmark in hand_landmarks.landmark:
                x, y, z = landmark.x * w, landmark.y * h, landmark.z
                landmarks_3d.append((int(x), int(y), z))

            # Print 3D coordinates
            for i, (x, y, z) in enumerate(landmarks_3d):
                print(f"Landmark {i + 1}: X={x}, Y={y}, Z={z}")

            # Draw lines between landmarks
            connections = mp_hand.HAND_CONNECTIONS
            for connection in connections:
                start_x, start_y, _ = landmarks_3d[connection[0]]
                end_x, end_y, _ = landmarks_3d[connection[1]]
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255,0,255), 2)

            # Draw landmarks on the frame
            for x, y, _ in landmarks_3d:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()