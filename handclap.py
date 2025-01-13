import mediapipe as mp
import numpy as np
import cv2
from functions import *

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def handclap_detection(flip = True, video_path = 0):
    cap = cv2.VideoCapture(video_path)

    with mp_hands.Hands(
        static_image_mode=False,  # False로 설정하면 동영상에서도 작동
        max_num_hands=2,  # 감지할 손의 최대 개수
        min_detection_confidence=0.5,  # 감지 임계값
        min_tracking_confidence=0.5  # 추적 임계값
    ) as hands:
        print("AA")
        while cap.isOpened():
            # total frame number
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            # Read a frame
            ret, frame = cap.read()

            # If reading the frame fails
            if not ret:
                print("Error: frame cout not be read.")
                break

            # Flip the frame horizontally
            if flip:
                frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            image, left_hand, right_hand = hand_mediapipe_detection(frame, hands)
            if right_hand is None or left_hand is None:
                print("NOOOOOOOOOOOOO")
                continue

            left_wrist_ndarray, right_wrist_ndarray, left_index_mcp_ndarry, right_index_mcp_ndarry, left_pinky_mcp_ndarry, right_pinky_mcp_ndarry = hand_width_height_scailing_ndarray(height, width, left_hand, right_hand)

            left_wrist_coords, right_wrist_coords, left_index_mcp_coords, right_index_mcp_coords, left_pinky_mcp_coords, right_pinky_mcp_coords = hand_width_height_scailing(height, width, left_hand, right_hand)

            # if right_hand is None or left_hand is None:
            #     continue
            
            left_wrist_to_index_mcp_vector = left_index_mcp_ndarry - left_wrist_ndarray

            left_wrist_to_pinky_mcp_vector = right_index_mcp_ndarry - left_wrist_ndarray

            right_wrist_to_index_mcp_vector = left_pinky_mcp_ndarry - right_wrist_ndarray

            right_wrist_to_pinky_mcp_vector = right_pinky_mcp_ndarry - right_wrist_ndarray

            left_normal_vector = np.cross(left_wrist_to_index_mcp_vector, left_wrist_to_pinky_mcp_vector)

            right_normal_vector = np.cross(right_wrist_to_index_mcp_vector, right_wrist_to_pinky_mcp_vector)
            
            if np.linalg.norm(left_normal_vector) != 0 and np.linalg.norm(right_normal_vector) != 0:
                print("AAAAAAAADFADSFADSFADSFADSFSDFADSFADSDASAFSAFDSDSAFDSAFAAFA")
                # Normalize the original normal_vector by its norm
                left_normal_vector = left_normal_vector / np.linalg.norm(left_normal_vector)

                right_normal_vector = right_normal_vector / np.linalg.norm(right_normal_vector)

                draw_normal_vector(image, left_wrist_coords, left_normal_vector)

                draw_normal_vector(image, right_wrist_coords, right_normal_vector)

            else:
                print("normal vector error")

            draw_vector(image, left_wrist_coords, left_index_mcp_coords)
            draw_vector(image, left_wrist_coords, left_pinky_mcp_coords)
            draw_vector(image, right_wrist_coords, right_index_mcp_coords)
            draw_vector(image, right_wrist_coords, right_pinky_mcp_coords)
            
            # print(left_hand[0][0]> left_hand[5][0])
            
            # mp_drawing.draw_landmarks(
            #     frame, right_hand, mp_hands.HAND_CONNECTIONS,
            #     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            #     mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            # )


            cv2.imshow("Handclap Detection", image)


            # Exit when the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # Release video capture and window resources
    cap.release()
    cv2.destroyAllWindows()

handclap_detection()