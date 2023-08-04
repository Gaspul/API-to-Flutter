import cv2
import numpy as np
import time
from func import mp_pose
from func import trajectory_len
from func import detect_interval
from func import feature_params
from func import lk_params
from func import calculate_angle
from func import mp_drawing


def squat(input_path, output_path):

    # LK Params
    trajectories = []
    frame_idx = 0

    #To store angle data
    angle_min_rknee = []
    angle_min_lknee = []
    angle_min_hip = []

    # Load the image file
    cap = cv2.VideoCapture(input_path)

    cap.set(3, 640)
    cap.set(4, 480)

    # Create a VideoWriter to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # Curl counter variables
    counter = 0
    min_ang_hip = 0
    min_ang_rknee = 0
    stage = None
    sqlevel = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            # start time to calculate FPS
            start = time.time()

            suc, frame = cap.read()

            if not suc:
                print("Detection Finish ...")
                break  # Exit the loop or perform other necessary actions

            if frame is None:
                print("Empty frame, skipping...")
                continue

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = frame.copy()

            # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
            if len(trajectories) > 0:
                img0, img1 = prev_gray, frame_gray
                p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1

                new_trajectories = []

                # Get all the trajectories
                for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    trajectory.append((x, y))
                    if len(trajectory) > trajectory_len:
                        del trajectory[0]
                    new_trajectories.append(trajectory)
                    # Newest detected point
                    cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

                trajectories = new_trajectories

                # Draw all the trajectories
                cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
                # cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1,
                #             (0, 255, 0), 2)

            # Update interval - When to update and detect new features
            if frame_idx % detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255

                # Lastest point in latest trajectory
                for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
                    cv2.circle(mask, (x, y), 5, 0, -1)

                # Detect the good features to track
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    # If good features can be tracked - add that to the trajectories
                    for x, y in np.float32(p).reshape(-1, 2):
                        trajectories.append([(x, y)])

            frame_idx += 1
            prev_gray = frame_gray

            # Recolor image to RGB
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            overlay = image.copy()

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)  # Knee joint angle
                angle_r_knee = round(angle_r_knee, 2)
                angle_l_knee = calculate_angle(l_hip, l_knee, l_ankle)  # Knee joint angle
                angle_l_knee = round(angle_l_knee, 2)

                angle_r_hip = calculate_angle(r_shoulder, r_hip, r_knee)
                angle_r_hip = round(angle_r_hip, 2)
                angle_l_hip = calculate_angle(l_shoulder, l_hip, l_knee)
                angle_l_hip = round(angle_l_hip, 2)

                angle_min_rknee.append(angle_r_knee)
                angle_min_lknee.append(angle_r_knee)
                angle_min_hip.append(angle_r_hip)

                # Visualize angle
                cv2.putText(image, str(angle_r_knee),
                            tuple(np.multiply(r_knee, [1080, 1920]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                cv2.putText(image, str(angle_l_knee),
                            tuple(np.multiply(l_knee, [1080, 1920]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                cv2.putText(image, str(angle_r_hip),
                            tuple(np.multiply(r_hip, [1080, 1920]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                cv2.putText(image, str(angle_l_hip),
                            tuple(np.multiply(l_hip, [1080, 1920]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # reps counter logic
                if angle_r_knee > 150 and angle_r_hip > 150:
                    stage = "Down"
                if angle_r_knee <= 90:
                    if angle_r_knee <= 90 and stage == "Down" and 40 <= angle_r_hip <= 100:
                        stage = "Up"
                        sqlevel = "Full Depth Squat, nice"
                        counter += 1
                        min_ang_rknee = min(angle_min_rknee)
                        print(min_ang_rknee)
                        print(counter)
                        print("Full Squat")
                        angle_min_rknee = []
                        angle_min_hip = []

                    elif angle_r_knee <= 90 and stage == "Down" and angle_r_hip <= 40:
                        stage = "Up"
                        sqlevel = "Leaning too forward, straighten your body"
                    elif angle_r_knee <= 90 and stage == "Down" and angle_r_hip >= 100:
                        stage = "Up"
                        sqlevel = "Lean forward slightly for balance"

                elif 90 <= angle_r_knee <= 125:
                    min_ang = min(angle_min_rknee)
                    min_angl = min(angle_min_lknee)
                    print(min_ang)
                    print(min_angl)
                    min_ang_rknee = min(angle_min_rknee)
                    # min_ang_lknee = min(angle_min_lkne)
                    min_ang_hip = min(angle_min_hip)
                    max_ang = max(angle_min_rknee)

                    print(min(angle_min_rknee), " _ ", max(angle_min_rknee))
                    print(min(angle_min_hip), " _ ", max(angle_min_hip))

                    if 90 <= angle_r_knee <= 125 and stage == "Down":
                        stage = "Down"
                        sqlevel = "Half reps, you need to go deeper"
                        counter += 0
                        print(counter)
                        print("Half Squat")

            except:
                pass
            # Render squat counter and status box
            cv2.rectangle(image, (20, 20), (600, 320), (0, 0, 0), -1)
            # Rep data

            cv2.putText(image, "REPS : " + str(counter),
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.putText(image, "Hip-joint angle : " + str(min_ang_hip),
                        (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # knee angle:
            cv2.putText(image, "Knee-joint angle : " + str(min_ang_rknee),
                        (30, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.rectangle(image, (20, 1700), (1050, 1860), (0, 0, 0), -1)

            alpha = 0.4

            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            cv2.putText(image_new, "Squat Level : " + str(sqlevel),
                        (30, 1750),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.putText(image_new, "Instruction : " + str(stage),
                        (30, 1830),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image_new, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(203, 17, 17), thickness=2, circle_radius=2)
                                      )
            # End time
            end = time.time()
            # calculate the FPS for current frame detection
            fps = 1 / (end - start)

            cv2.putText(image_new, f"{fps:.2f} FPS", (950, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write the processed frame to the output video file
            out.write(image_new)

        cap.release()
        out.release()