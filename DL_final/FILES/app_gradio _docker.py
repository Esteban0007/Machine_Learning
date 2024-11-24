# IMPORTS ----------------------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import gradio as gr

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
lastAction = ""
colors = [
    (117, 16, 245),
    (255, 255, 0),
    (0, 255, 255),
]
model = load_model(MODEL_PATH)

# FUNCTIONS ----------------------------------------------------------------


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


def mediapipe_detection(image, holistic_model):
    if image is None:
        print("Error: La imagen de entrada es None.")
        return None, None

    # Convertir la imagen a una matriz NumPy si no lo es
    try:
        image = np.asarray(image, dtype=np.uint8)
    except Exception as e:
        print(f"Error al convertir la imagen a NumPy: {e}")
        return None, None

    try:
        # Convertir a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error al convertir la imagen a RGB: {e}")
        return None, None

    # Asegurar que la imagen sea modificable
    try:
        image.flags.writeable = (
            False  # Hacer la imagen no modificable (para rendimiento)
        )
    except ValueError:
        print("Advertencia: No se pudo cambiar el flag writeable a False.")

    # Procesar con Mediapipe
    results = holistic_model.process(image_rgb)

    try:
        image.flags.writeable = True  # Restaurar la modificabilidad de la imagen
    except ValueError:
        print("Advertencia: No se pudo cambiar el flag writeable a True.")

    # Convertir de nuevo a BGR para mostrar
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results


def draw_styled_landmarks(image, holistic_results):
    if holistic_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            holistic_results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

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


# MAIN ----------------------------------------------------------------
def process_frame(frame):
    global sequence, predictions, detected_action, lastAction

    # Set mediapipe model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        if results is None:
            return frame

        # Draw landmarks
        draw_styled_landmarks(image, results)

        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        if hand_detected:
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                predictions.append(np.argmax(res))

                # Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > THRESHOLD:
                        detected_action = ACTIONS[np.argmax(res)]

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

        return image


# Gradio Interface --------------------------------------------------------
demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(source="webcamp", streaming=True),
    outputs="image",
    live=True,
)

if __name__ == "__main__":
    demo.launch()
