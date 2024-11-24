# IMPORTS ----------------------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np
import os
from scipy import stats
from tensorflow.keras.models import load_model
from gtts import gTTS


# CONSTANTS ----------------------------------------------------------------
THRESHOLD = 0.90
ACTIONS = np.array(["hola", "gracias", "te quiero"])
MODEL_PATH = "gesture_recognition_model.h5"

# VARIABLES ----------------------------------------------------------------
sequence = []
predictions = []
detected_action = ""
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
lastAction = 1
colors = [
    (117, 16, 245),
    (255, 255, 0),
    (0, 255, 255),
]
model = load_model(MODEL_PATH)


# Functions --------------------------------------------------------
def prob_viz(res, ACTIONS, input_frame, colors):

    output_frame = input_frame.copy()

    for num, prob in enumerate(res):
        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (int(prob * 100), 90 + num * 40),
            colors[num],
            -1,
        )
        cv2.putText(
            output_frame,
            ACTIONS[num],
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output_frame


def mediapipe_detection(image, holistic_model, face_mesh_model=None):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert RGB to RGB
    image.flags.writeable = False  # Make the image non-writeable for performance
    holistic_results = holistic_model.process(image_rgb)  # Make predictions

    # Draw landmarks of the face
    face_results = None
    if face_mesh_model:
        face_results = face_mesh_model.process(image_rgb)

    image.flags.writeable = True  # Image is now writable
    image_bgr = cv2.cvtColor(
        image_rgb, cv2.COLOR_RGB2BGR
    )  # Convert back to BGR for OpenCV

    return image_bgr, holistic_results, face_results


def draw_styled_landmarks(image, holistic_results, face_results):
    # Draw landmarks of the body
    if holistic_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            holistic_results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

    # Draw landmarks of the hands
    if holistic_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            holistic_results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if holistic_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            holistic_results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

    # Draw landmarks of the face with FaceMesh
    if face_results and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(
                    color=(80, 110, 10), thickness=1, circle_radius=1
                ),
                mp_drawing.DrawingSpec(
                    color=(80, 256, 121), thickness=1, circle_radius=1
                ),
            )


def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def text2voice(text):
    tts = gTTS(text=text, lang="es", slow=False)
    tts.save("output.mp3")
    os.system("mpg123 output.mp3")


# MAIN ----------------------------------------------------------------

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results, face_results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results, face_results)

        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        if hand_detected:
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(ACTIONS[np.argmax(res)])

                predictions.append(np.argmax(res))

                # Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > THRESHOLD:

                        # text2voice
                        if ACTIONS[np.argmax(res)] != lastAction:
                            detected_action = ACTIONS[np.argmax(res)]

                            text2voice(detected_action)
                            lastAction = detected_action

                # Viz probabilities
                image = prob_viz(res, ACTIONS, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                image,
                detected_action,
                (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Show to screen
        cv2.imshow("Sing language Detection IA", image)

        # Break with q or Q
        if cv2.waitKey(10) & 0xFF == ord("q") or cv2.waitKey(10) & 0xFF == ord("Q"):
            break

    cap.release()
    cv2.destroyAllWindows()
