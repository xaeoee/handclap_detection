import cv2
import numpy as np

def head_mediapipe_detection(frame, model):
    """
    Detect facial landmarks using MediaPipe FaceMesh model and extract key points.
    
    Args:
        frame (np.array): Input image frame in BGR format
        model: MediaPipe FaceMesh model instance
    
    Returns:
        tuple: Contains:
            - frame (np.array): Processed image frame
            - left_eye_coords (np.array or None): Coordinates of left eye landmark (x,y,z)
            - right_eye_coords (np.array or None): Coordinates of right eye landmark (x,y,z)
            - left_lip_coords (np.array or None): Coordinates of left lip corner landmark (x,y,z)
            - right_lip_coords (np.array or None): Coordinates of right lip corner landmark (x,y,z)
            - center_forehead_coords (np.array or None): Coordinates of center forehead point (x,y,z)
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = model.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if not results.multi_face_landmarks:  # Check if no face landmarks are detected
            return frame, None, None, None, None, None
    for face_landmarks in results.multi_face_landmarks:
        # Extract landmarks as a NumPy array
        landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks.landmark])
        left_eye_coords = landmarks[133]
        right_eye_coords = landmarks[362]
        left_lip_coords = landmarks[61]
        right_lip_coords = landmarks[291]
        center_forehead_coords = (right_eye_coords + left_eye_coords) / 2

    return frame, left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords, center_forehead_coords

def head_width_height_scailing(height, width, left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords, center_forehead_coords):
    """
    Scale facial landmark coordinates based on image dimensions.
    
    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels
        left_eye_coords (np.array): Left eye coordinates (x,y,z)
        right_eye_coords (np.array): Right eye coordinates (x,y,z)
        left_lip_coords (np.array): Left lip corner coordinates (x,y,z)
        right_lip_coords (np.array): Right lip corner coordinates (x,y,z)
        center_forehead_coords (np.array): Center forehead coordinates (x,y,z)
    
    Returns:
        tuple: Scaled coordinates for all facial landmarks in pixel values
    """
    left_eye_coords = (int(left_eye_coords[0] * width), int(left_eye_coords[1] * height), int(left_eye_coords[2]))
    right_eye_coords = (int(right_eye_coords[0] * width), int(right_eye_coords[1] * height), int(right_eye_coords[2]))
    left_lip_coords = (int(left_lip_coords[0] * width), int(left_lip_coords[1] * height), int(left_lip_coords[2]))
    right_lip_coords = (int(right_lip_coords[0] * width), int(right_lip_coords[1] * height), int(right_lip_coords[2]))
    center_forehead_coords = (int(center_forehead_coords[0] * width), int(center_forehead_coords[1] * height), int(center_forehead_coords[2]))

    return left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords, center_forehead_coords

def head_coords_from_3d_to_2d(left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords, center_forehead_coords):
    """
    Convert 3D coordinates to 2D by removing the z-component.
    
    Args:
        left_eye_coords (tuple): 3D coordinates of left eye
        right_eye_coords (tuple): 3D coordinates of right eye
        left_lip_coords (tuple): 3D coordinates of left lip corner
        right_lip_coords (tuple): 3D coordinates of right lip corner
        center_forehead_coords (tuple): 3D coordinates of center forehead
    
    Returns:
        tuple: 2D coordinates (x,y) for all facial landmarks
    """
    left_eye_coords = (left_eye_coords[0], left_eye_coords[1])
    right_eye_coords = (right_eye_coords[0], right_eye_coords[1])
    left_lip_coords = (left_lip_coords[0], left_lip_coords[1])
    right_lip_coords = (right_lip_coords[0], right_lip_coords[1])
    center_forehead_coords = (center_forehead_coords[0], center_forehead_coords[1])

    return left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords, center_forehead_coords

def head_plot_circle(image, center_forehead_coords, left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords):
    """
    Draw circles at facial landmark positions on the image.
    
    Args:
        image (np.array): Input image
        center_forehead_coords (tuple): Center forehead coordinates (x,y)
        left_eye_coords (tuple): Left eye coordinates (x,y)
        right_eye_coords (tuple): Right eye coordinates (x,y)
        left_lip_coords (tuple): Left lip corner coordinates (x,y)
        right_lip_coords (tuple): Right lip corner coordinates (x,y)
    """
    cv2.circle(image, center_forehead_coords, radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, left_eye_coords, radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, right_eye_coords, radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, left_lip_coords, radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, right_lip_coords, radius=3, color=(0, 255, 0), thickness=-1)

def draw_vector(image, start_point, end_point, color=(0, 255, 0), thickness=2):
    """
    Draw a line vector between two points on the image.
    
    Args:
        image (np.array): Input image
        start_point (tuple): Starting coordinates (x,y)
        end_point (tuple): Ending coordinates (x,y)
        color (tuple): RGB color values, defaults to green (0,255,0)
        thickness (int): Line thickness in pixels, defaults to 2
    """
    print("drawing")
    cv2.line(image, start_point, end_point, color, thickness)

def draw_normal_vector(image, start_point, normal_vector, scale=120):
    """
    Draw an arrow representing the normal vector on the image.
    
    Args:
        image (np.array): Input image
        start_point (tuple): Starting coordinates (x,y)
        normal_vector (np.array): Normal vector components [x,y]
        scale (int): Scale factor for vector visualization, defaults to 60
    """
    end_point = (int(start_point[0] + normal_vector[0] * scale), 
                    int(start_point[1] - normal_vector[1] * scale))
    cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), 2, tipLength=0.3)

def head_angle_calculation(previous_normalised_vector, current_normalised_vector):
    """
    Calculate the angle between two normalized vectors.
    
    Args:
        previous_normalised_vector (np.array): Previous frame's normalized vector
        current_normalised_vector (np.array): Current frame's normalized vector
    
    Returns:
        float: Angle in degrees between the two vectors
    """
    cosine_similarity = np.dot(previous_normalised_vector, current_normalised_vector)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    angle_radians = np.arccos(cosine_similarity)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def head_movement_direction(previous_normal_vector, current_normal_vector):
    """
    Determine the direction of head movement based on vector changes.
    
    Args:
        previous_normal_vector (np.array): Normal vector from previous frame
        current_normal_vector (np.array): Normal vector from current frame
    
    Returns:
        str: Direction of movement ('left', 'right', 'up', 'down', or 'stationary')
    """
    vector_difference = current_normal_vector - previous_normal_vector

    if abs(vector_difference[0]) > abs(vector_difference[1]):
        if vector_difference[0] > 0:
            return "right"
        else:
            return "left"
    elif abs(vector_difference[0]) < abs(vector_difference[1]):
        if vector_difference[1] > 0:
            return "down"
        else:
            return "up"
    else:
        return "stationary"
    

# -------------------------- handclap detection function below this line -------------------------- #



def hand_mediapipe_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = model.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # print(len(results.multi_hand_landmarks)) number of hands
    if results.multi_hand_landmarks:
        hands_count = len(results.multi_hand_landmarks)
        if hands_count == 2:
            # 왼손, 오른손 분류
            # results.multi_hand_landmarks[0] or [1] 은 첫번째 손 두번째손 을 나타낸다.
            # .landmark[0] 은 손목을 나타낸다. 그래서 두 손의 손목 x좌표를 비교해서 큰쪽 손을 왼손으로 설정한다.
            # 사람의 왼손이 화면상 왼쪽에 나와서이다.
            if results.multi_hand_landmarks[0].landmark[0].x > results.multi_hand_landmarks[1].landmark[0].x:
                left_hand = np.array([(landmark.x, landmark.y, landmark.z) for landmark in results.multi_hand_landmarks[1].landmark])
                right_hand = np.array([(landmark.x, landmark.y, landmark.z) for landmark in results.multi_hand_landmarks[0].landmark])
            else:
                left_hand = np.array([(landmark.x, landmark.y, landmark.z) for landmark in results.multi_hand_landmarks[0].landmark])
                right_hand = np.array([(landmark.x, landmark.y, landmark.z) for landmark in results.multi_hand_landmarks[1].landmark])
            print("Both")

            return frame, left_hand, right_hand

        elif hands_count == 1:
            # 한 손만 감지된 경우
            # .landmark[4] 는 엄지 손가락의 끝부분.
            # .landmark[20] 은 새끼손가락 끝 부분.
            # 만약 새끼 손까락이 엄지손가락보다 오른쪽에 위치해있을경우
            # 그건 오른손이다.
            if results.multi_hand_landmarks[0].landmark[4].x < results.multi_hand_landmarks[0].landmark[20].x:
                print("Right")
                return frame, None, np.array([(landmark.x, landmark.y, landmark.z) for landmark in results.multi_hand_landmarks[0].landmark])
            else:
                # 새끼 손가락이 엄지손가락보다 화면상 좌측에 있을경우 오른손이다.
                print("Left")
                return frame, np.array([(landmark.x, landmark.y, landmark.z) for landmark in results.multi_hand_landmarks[0].landmark]), None
        else:
            return frame, None, None
    else:
        return frame, None, None


def hand_width_height_scailing(height, width, left_hand, right_hand):
    left_wrist_coords = (int(left_hand[0][0] * width), int(left_hand[0][1] * height))
    right_wrist_coords = (int(right_hand[0][0] * width), int(right_hand[0][1] * height))
    left_index_mcp_coords = (int(left_hand[5][0] * width), int(left_hand[5][1] * height))
    right_index_mcp_coords = (int(right_hand[5][0] * width), int(right_hand[5][1] * height))
    left_pinky_mcp_coords = (int(left_hand[17][0] * width), int(left_hand[17][1] * height))
    right_pinky_mcp_coords = (int(right_hand[17][0] * width), int(right_hand[17][1] * height))

    return left_wrist_coords, right_wrist_coords, left_index_mcp_coords, right_index_mcp_coords, left_pinky_mcp_coords, right_pinky_mcp_coords

def hand_width_height_scailing_ndarray(height, width, left_hand, right_hand):
    left_wrist_ndarray = left_hand[0]
    right_wrist_ndarray = right_hand[0]
    left_index_mcp_ndarry = left_hand[5]
    right_index_mcp_ndarry = right_hand[5]
    left_pinky_mcp_ndarry = left_hand[17]
    right_pinky_mcp_ndarry = right_hand[17]
    
    return left_wrist_ndarray, right_wrist_ndarray, left_index_mcp_ndarry, right_index_mcp_ndarry, left_pinky_mcp_ndarry, right_pinky_mcp_ndarry