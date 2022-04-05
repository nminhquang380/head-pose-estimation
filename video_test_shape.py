import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
import pyautogui as pg


cwd = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]]

H, W = 140, 35

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)
    
    reprojectdst = tuple(map(tuple, (reprojectdst.reshape(8, 2))))


    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle

def center_vector(list):
    x_start = int((list[1][0] + list[5][0] +list[2][0] +list[6][0]) / 4)
    y_start = int((list[1][1] + list[5][1] +list[2][1] +list[6][1]) / 4)
    start = (x_start, y_start)
    x_end = int((list[0][0] + list[4][0] +list[3][0] +list[7][0]) / 4)
    y_end = int((list[0][1] + list[4][1] +list[3][1] +list[7][1]) / 4)
    end = (x_end, y_end)
    return start, end

'''
define output to control mouse
0: center
1: top
2: bot
3: left
4: right
'''
def control_mouse(start, end, step):
    vector = (end[0]-start[0], end[1]-start[1])
    h = end[1]-start[1]
    w = end[0]-start[0]
    # center do nothing
    if ((w/4)**2 + h**2) <= 1225:
        pass
    # top move up
    elif h-0.25*w >= 0 and h+0.25*w >= 0:
        pg.move(0, step)
    # bot move down
    elif h-0.25*w <= 0 and h+0.25*w <= 0:
        pg.move(0, -step)
    # left move left
    elif h-0.25*w > 0 and h+0.25*w < 0:
        pg.move(step, 0)
    # right move right
    elif h-0.25*w < 0 and h+0.25*w > 0:
        pg.move(-step, 0)

def main():
    # return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)

                # Visual landmark points of face  
                # for (x, y) in shape:
                #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                # Visual tư thế đầu bằng hình khối
                # for start, end in line_pairs:
                    # start_point = tuple(map(int, element.item()) for element in reprojectdst[start])
                    # end_point = tuple(map(int, element.item()) for element in reprojectdst[end])
                    # start_point = tuple([int(ele) for ele in reprojectdst[start]])
                    # end_point = tuple([int(ele) for ele in reprojectdst[end]])
                    # cv2.line(frame, start_point, end_point, (0, 0, 255))
                    # cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                # Visual bằng đường thẳng
                start, end = center_vector(reprojectdst)
                cv2.line(frame, start, end, (0, 0, 255))
                cv2.putText(frame,"x = {:.2f}".format(end[0]-start[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                              0.75, (0, 0, 0))
                cv2.putText(frame,"y = {:.2f}".format(end[1]-start[1]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                              0.75, (0, 0, 0))

                #control mouse
                control_mouse(start, end, 25)
                # 3D vector
                # cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, (0, 0, 0), thickness=2)
                # cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, (0, 0, 0), thickness=2)
                # cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, (0, 0, 0), thickness=2)

            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
