# IMPORTS ----------------------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np
import os

# CONSTANTS ----------------------------------------------------------------
DATA_PATH = os.path.join("MP_Data")

# VARIABLES ----------------------------------------------------------------
actions = np.array(["hello", "thanks", "iloveyou"])

no_sequences = 30
sequence_length = 30
start_folder = 1

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_face_mesh = mp.solutions.face_mesh  # FaceMesh model



# FUNCTIONS ----------------------------------------------------------------
def mediapipe_detection(image, holistic_model, face_mesh_model=None):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Make the image non-writeable for performance
    holistic_results = holistic_model.process(image_rgb)

    # Si estás usando FaceMesh para la cara, también se debe procesar la imagen
    face_results = None
    if face_mesh_model:
        face_results = face_mesh_model.process(image_rgb)

    image.flags.writeable = True
    image_bgr = cv2.cvtColor(
        image_rgb, cv2.COLOR_RGB2BGR
    )  # Convert back to BGR for OpenCV
    return image_bgr, holistic_results, face_results


def draw_styled_landmarks(image, holistic_results, face_results):
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


# MAIN ----------------------------------------------------------------

cap = cv2.VideoCapture(0)

# Start MediaPipe models
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:

    # Config of FaceMesh with use of CPU
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        
    ) as face_mesh_model:

        # Get data for every action
        for action in actions:
            # Video
            for sequence in range(start_folder, start_folder + no_sequences):
                # Frames
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()

                    # Detections
                    image, holistic_results, face_results = mediapipe_detection(
                        frame, holistic, face_mesh_model
                    )

                    # Draw landmarks
                    draw_styled_landmarks(image, holistic_results, face_results)

                    # Data colection logic
                    if frame_num == 0:
                        cv2.putText(
                            image,
                            "STARTING COLLECTION",
                            (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            4,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            image,
                            f"Collecting frames for {action} Video Number {sequence}",
                            (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("OpenCV Feed", image)
                        cv2.waitKey(500)
                    else:
                        cv2.putText(
                            image,
                            f"Collecting frames for {action} Video Number {sequence}",
                            (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("OpenCV Feed", image)

                    # Export key points
                    keypoints = extract_keypoints(holistic_results)
                    npy_path = os.path.join(
                        DATA_PATH, action, str(sequence), f"{frame_num}.npy"
                    )
                    np.save(npy_path, keypoints)

                    # Go out with "q"
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        break

cap.release()
cv2.destroyAllWindows()
