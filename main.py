import numpy as np
import cv2
import time

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=7)

trajectory_len = 10  # Difference length lines between old point and new point
detect_interval = 1  # interval between detects first point and second point and so on
trajectories = []  # the old frame point and new frame point
frame_idx = 0

cap = cv2.VideoCapture("Sample Video/Sample1.mp4")

while True:
    # start time to calculate FPS
    start = time.time()

    suc, frame1 = cap.read()
    frame = cv2.resize(frame1, (0, 0), fx=1, fy=1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray # frames
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1,2)  # saving points on frames previous and current frames
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1  # check trajctory if it's good optical flow of objects

        new_trajectories = []

        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))  # good points of x,y and append them
            if len(trajectory) > trajectory_len:
                del trajectory[0]  # if reach the max trajectory then will delete for lines differences
            new_trajectories.append(trajectory)  # saving current tranjectory that for to manipulate with old point
            # Newest detected point (y > 150 and y < 410) or
            if (y > 150 and y < 410) or (x > 700 or x < 200 ):
                # solved
                # (y - trajectory[0][1] > 0) detects dwon(y axis) , (x - trajectory[0][0] < 0) detects left(x axis)
                # (y - trajectory[0][1] < 0) up(y axis) , (x - trajectory[0][0] > 0) right(x axis)

                if (y - trajectory[0][1] > -0.09) and (x - trajectory[0][0] < 0):
                    cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), 1)  # color points detects

                elif (y - trajectory[0][1] > -0.09) and (x - trajectory[0][0] > 0):
                    cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), 1)  # color points detects
                else:
                    cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), 1)







        trajectories = new_trajectories

        # Draw all the trajectories
        # cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False,
        #               (0, 255, 0))  # Detect differences trajectory between two points
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        # track count: %d that to count how many point is tracking

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in
                     trajectories]:  # takes last two good point on trajectory
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask,
                                    **feature_params)  # mask = mask to track latest point on trajectory, **feature_params will take parameter again
        if p is not None:  # final step of trajectory to be tracked points
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end - start)

    # Show Results
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
