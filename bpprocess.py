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

def benchpress(input_path, output_path):

    # LK Params
    trajectories = []
    frame_idx = 0

    # To store angle data
    angle_min_lelbow = []

    # Load the image file
    cap = cv2.VideoCapture(input_path)

    cap.set(3, 640)
    cap.set(4, 480)

    # Create a VideoWriter to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    min_ang_lelbow = 0
    counter = 0
    stage = None
    bplevel = None

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
                cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 2)

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
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                # Calculate angle
                angle_r_shoulder = calculate_angle(l_shoulder, r_shoulder, r_elbow)
                angle_r_shoulder = round(angle_r_shoulder, 2)

                angle_l_shoulder = calculate_angle(r_shoulder, l_shoulder, l_elbow)
                angle_l_shoulder = round(angle_l_shoulder, 2)

                angle_r_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)
                angle_r_elbow = round(angle_r_elbow, 2)

                angle_l_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
                angle_l_elbow = round(angle_l_elbow, 2)

                angle_min_lelbow.append(angle_l_elbow)

                # Visualize angle
                """cv2.putText(image, str(angle), 
                               tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )"""

                cv2.putText(image, str(angle_r_shoulder),
                            tuple(np.multiply(r_shoulder, [1080, 1920]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                cv2.putText(image, str(angle_l_shoulder),
                            tuple(np.multiply(l_shoulder, [1080, 1920]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                cv2.putText(image, str(angle_r_elbow),
                            tuple(np.multiply(r_elbow, [1080, 1920]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                cv2.putText(image, str(angle_l_elbow),
                            tuple(np.multiply(l_elbow, [1080, 1920]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle_l_elbow > 140:
                    stage = "Down"
                    bplevel = "Straigth elbow, full reps position"
                if angle_l_elbow <= 90:
                    if angle_l_elbow <= 90 and stage == "Down":
                        stage = "Up"
                        bplevel = "Full depth press, good form"
                        counter += 1
                        print(counter)
                        min_ang_lelbow = min(angle_min_lelbow)
                        print(min_ang_lelbow)
                        angle_min_lelbow = []
                elif angle_l_elbow >= 90:
                    if angle_l_elbow >= 90 and stage == "Down":
                        stage = "Down"
                        bplevel = "Half reps, go more deeper for better form"
                        counter += 0
                        print(counter)
                        min_ang_lelbow = min(angle_min_lelbow)
                        print(min_ang_lelbow)
                        angle_min_lelbow = []

            except:
                pass

                # Setup status box reps counter
            cv2.rectangle(image, (20, 20), (435, 160), (0, 0, 0), -1)

            cv2.putText(image, "REPS : " + str(counter),
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.rectangle(overlay, (20, 620), (435, 700), (0, 0, 0), -1)

            alpha = 0.4

            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            cv2.putText(image_new, "Instruction : " + str(stage),
                        (30, 680),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.putText(image_new, "Bench Press Level : " + str(bplevel),
                        (30, 640),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image_new, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(203, 17, 17), thickness=2, circle_radius=2)
                                      )

            # End time
            end = time.time()
            # calculate the FPS for current frame detection
            fps = 1 / (end - start)

            # Show Results
            cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write the processed frame to the output video file
            out.write(image_new)

        cap.release()
        out.release()