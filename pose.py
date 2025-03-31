import mediapipe as mp
import numpy as np
import cv2
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_pose_connections = mp_pose.POSE_CONNECTIONS

LEFT_WRIST_INDEX = 15
RIGHT_WRIST_INDEX = 16
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12

RED = "\033[31m"
RESET = "\033[0m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"

def calculate_2d_distance(lm1, lm2, width, height):
    x1, y1 = lm1.x * width, lm1.y * height
    x2, y2 = lm2.x * width, lm2.y * height
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def pose_estimation(video_path, flip=True):
    cap = cv2.VideoCapture(video_path)

    first_constraint = False

    second_constraint = False

    clap_count = 0

    with mp_pose.Pose(
        static_image_mode=False,  
        model_complexity=1,       
        enable_segmentation=False, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame could not be read.")
                break
            
            if flip:
                frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_wrist = landmarks[LEFT_WRIST_INDEX]
                right_wrist = landmarks[RIGHT_WRIST_INDEX]
                left_shoulder = landmarks[LEFT_SHOULDER_INDEX]
                right_shoulder = landmarks[RIGHT_SHOULDER_INDEX]

                shoulder_width = calculate_2d_distance(left_shoulder, right_shoulder, width, height)

                wrist_distance = calculate_2d_distance(left_wrist, right_wrist, width, height)

                # 손목 거리 정규화 (어깨 너비 기준)
                if shoulder_width > 0:
                    wrist_ratio = wrist_distance / shoulder_width
                else:
                    wrist_ratio = 0

                # print(f"손목 간 2D 거리: {wrist_distance:.4f} px")
                # print(f"어깨 너비(2D): {shoulder_width:.4f} px")
                # print(f"정규화된 손목 거리 비율: {wrist_ratio:.4f}")


                current_time = time.time()
                # 만약 모든 조건이 False즉 초기 상태일때 두 손의 거리가 30 이하라면
                if first_constraint is False and second_constraint is False and wrist_ratio < 1.0:
                    print(wrist_distance)
                    print(f"{GREEN}FIRSTTTTTTTTTTTTTTT{RESET}")
                    first_constraint = True
                    starting_time = time.time()
                
                # 만약 처음 두 손이 30거리 안에 있어서 첫 조건이 시작되고 시간이 흘렀지만
                # 두 손벽이 n + m 만큼의 거리로 벌려지지 않고 계속 그대로 있고 시간 임계값이 지날경우 조건 초기화
                elif first_constraint and (current_time - starting_time) > 6:
                    print(f"{RED}CANCELING CONSTRAINTT!!!!{RESET}")
                    first_constraint = False
                    second_constraint = False

                current_time = time.time()
            
                # 첫 조건(거리가 30 이하) 가 달성 됐고, 2번째 조건이 미달성이며, 손목사이의 거리가 60 이상이고 첫번째 조건 달성 시점부터 3초가 지나지 않았다면 2번째 조건 True
                if first_constraint is True and second_constraint is False and wrist_ratio > 1.1 and (current_time - starting_time) < 5:
                    print(f"{GREEN}SECONDDDDDDDDDDDDD{RESET}")
                    second_constraint = True

                # second_constraint 이 true인거를 체크할 필요는 없고, 만약 위에 조건이 false로 나왔을때 (거리가 이상하거나 시간이 넘어갔거나 했을때) 조건 다시 초기화 처음부터 재시작.
                elif second_constraint and (current_time - starting_time) > 5:
                    print(f"{RED}CANCELING CONSTRAINTT!!!!{RESET}")
                    first_constraint = False
                    second_constraint = False
                
                current_time = time.time()
                if first_constraint is True and second_constraint is True and (current_time - starting_time) < 5 and wrist_ratio < 1.0:
                    print(f"{BLUE}CLAPPPPPPPPPPPPPPPP{RESET}")
                    clap_count += 1
                    print(f"{YELLOW}{clap_count} times clapped{RESET}")
                    first_constraint = False
                    second_constraint = False

                cv2.circle(image, 
                           (int(left_wrist.x * width), int(left_wrist.y * height)), 
                           10, (0, 255, 0), -1)
                cv2.circle(image, 
                           (int(right_wrist.x * width), int(right_wrist.y * height)), 
                           10, (0, 255, 0), -1)
                cv2.circle(image, 
                           (int(left_shoulder.x * width), int(left_shoulder.y * height)), 
                           10, (255, 0, 0), -1)
                cv2.circle(image, 
                           (int(right_shoulder.x * width), int(right_shoulder.y * height)), 
                           10, (255, 0, 0), -1)

                text = f"Wrist Ratio: {wrist_ratio:.2f}"
                cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.namedWindow("Handclap Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Handclap Detection", 1280, 720)
            cv2.imshow("Handclap Detection", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

pose_estimation(0)
