import mediapipe as mp
import numpy as np
import cv2
import time
from functions import *

RED = "\033[31m"
RESET = "\033[0m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# diff = []
def handclap_detection(flip = True, video_path = 0):
    cap = cv2.VideoCapture(video_path)

    first_constraint = False

    second_constraint = False

    clap_count = 0
    hand_count = 0

    with mp_hands.Hands(
        static_image_mode=False,  # False로 설정하면 동영상에서도 작동
        max_num_hands=2,  # 감지할 손의 최대 개수
        min_detection_confidence=0.5,  # 감지 임계값
        min_tracking_confidence=0.5  # 추적 임계값
    ) as hands:
        current_frame_count = 0
        first_constraint = False
        current_time = 0
        
        while cap.isOpened():
            # total frame number
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Read a frame
            ret, frame = cap.read()

            # If reading the frame fails
            if not ret:
                print("Error: frame cout not be read.")
                break

            current_frame_count += 1

            # Flip the frame horizontally
            if flip:
                frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            image, left_hand, right_hand = hand_mediapipe_detection(frame, hands)
            
            if right_hand is not None and left_hand is not None:
                hand_count+= 1


                x_left_min, x_left_max, y_left_min, y_left_max, x_right_min, x_right_max, y_right_min, y_right_max = calculate_bounding_box(height, width, left_hand, right_hand)
                # calculate_bounding_box(height, width, left_hand, right_hand)
                # print(x_left_min, x_left_max, y_left_min, y_left_max, x_right_min, x_right_max, y_right_min, y_right_max)
                # print(type(x_left_max))

                left_wrist_ndarray, right_wrist_ndarray, left_index_mcp_ndarry, right_index_mcp_ndarry, left_pinky_mcp_ndarry, right_pinky_mcp_ndarry = hand_width_height_scailing_ndarray(height, width, left_hand, right_hand)
                
                left_wrist_coords, right_wrist_coords, left_index_mcp_coords, right_index_mcp_coords, left_pinky_mcp_coords, right_pinky_mcp_coords = hand_width_height_scailing(height, width, left_hand, right_hand)
                
                left_wrist_to_index_mcp_vector = left_index_mcp_ndarry - left_wrist_ndarray

                left_wrist_to_pinky_mcp_vector = right_index_mcp_ndarry - left_wrist_ndarray

                right_wrist_to_index_mcp_vector = left_pinky_mcp_ndarry - right_wrist_ndarray

                right_wrist_to_pinky_mcp_vector = right_pinky_mcp_ndarry - right_wrist_ndarray

                left_normal_vector = np.cross(left_wrist_to_index_mcp_vector, left_wrist_to_pinky_mcp_vector)

                right_normal_vector = np.cross(right_wrist_to_index_mcp_vector, right_wrist_to_pinky_mcp_vector)
                
                if np.linalg.norm(left_normal_vector) != 0 and np.linalg.norm(right_normal_vector) != 0:
                    # Normalize the original normal_vector by its norm
                    left_normal_vector = left_normal_vector / np.linalg.norm(left_normal_vector)

                    right_normal_vector = right_normal_vector / np.linalg.norm(right_normal_vector)

                    draw_normal_vector(image, left_wrist_coords, left_normal_vector)

                    draw_normal_vector(image, right_wrist_coords, right_normal_vector)

                else:
                    print("normal vector error")

                draw_bounding_boxes(image, height, width,
                        x_left_min, x_left_max, y_left_min, y_left_max, 
                        x_right_min, x_right_max, y_right_min, y_right_max)
                draw_vector(image, left_wrist_coords, left_index_mcp_coords)
                draw_vector(image, left_wrist_coords, left_pinky_mcp_coords)
                draw_vector(image, right_wrist_coords, right_index_mcp_coords)
                draw_vector(image, right_wrist_coords, right_pinky_mcp_coords)

                wrist_distance = abs(left_wrist_ndarray[0] - right_wrist_ndarray[0])
                
                ratio, wrist_distance, avg_diagonal = calculate_diagonal_distance_ratio(x_left_min, x_left_max, y_left_min, y_left_max, x_right_min, x_right_max, y_right_min, y_right_max, wrist_distance)

                print(ratio)
                # current_time = time.time()

                # # 만약 모든 조건이 False즉 초기 상태일때 두 손의 거리가 30 이하라면
                # if first_constraint is False and second_constraint is False and wrist_distance < 0.02:
                #     print(wrist_distance)
                #     print(f"{GREEN}FIRSTTTTTTTTTTTTTTT{RESET}")
                #     first_constraint = True
                #     starting_time = time.time()
                
                # # 만약 처음 두 손이 30거리 안에 있어서 첫 조건이 시작되고 시간이 흘렀지만
                # # 두 손벽이 n + m 만큼의 거리로 벌려지지 않고 계속 그대로 있고 시간 임계값이 지날경우 조건 초기화
                # elif first_constraint and (current_time - starting_time) > 6:
                #     print(f"{RED}CANCELING CONSTRAINTT!!!!{RESET}")
                #     first_constraint = False
                #     second_constraint = False

                # current_time = time.time()
            
                # # 첫 조건(거리가 30 이하) 가 달성 됐고, 2번째 조건이 미달성이며, 손목사이의 거리가 60 이상이고 첫번째 조건 달성 시점부터 3초가 지나지 않았다면 2번째 조건 True
                # if first_constraint is True and second_constraint is False and wrist_distance > 0.03 and (current_time - starting_time) < 3:
                #     print(f"{GREEN}SECONDDDDDDDDDDDDD{RESET}")
                #     second_constraint = True

                # # second_constraint 이 true인거를 체크할 필요는 없고, 만약 위에 조건이 false로 나왔을때 (거리가 이상하거나 시간이 넘어갔거나 했을때) 조건 다시 초기화 처음부터 재시작.
                # elif second_constraint and (current_time - starting_time) > 3:
                #     print(f"{RED}CANCELING CONSTRAINTT!!!!{RESET}")
                #     first_constraint = False
                #     second_constraint = False
                
                # current_time = time.time()
                # if first_constraint is True and second_constraint is True and (current_time - starting_time) < 3 and wrist_distance < 0.20:
                #     print(f"{BLUE}CLAPPPPPPPPPPPPPPPP{RESET}")
                #     clap_count += 1
                #     print(f"{YELLOW}{clap_count} times clapped{RESET}")
                #     first_constraint = False
                #     second_constraint = False


            elif right_hand is None:
                
                pass
            elif left_hand is None:
                pass
            # print(current_frame_count)
            # print(hand_count)
            cv2.namedWindow("Handclap Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Handclap Detection", 1280, 720)
            cv2.imshow("Handclap Detection", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # Release video capture and window resources
    cap.release()
    cv2.destroyAllWindows()

handclap_detection()