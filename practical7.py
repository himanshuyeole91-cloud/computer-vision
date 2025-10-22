import os
import cv2
import numpy as np

def lucas_kanade_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        cap.release()
        return
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    if p0 is None or len(p0) == 0:
        print("No good features found to track.")
        cap.release()
        return
    mask = np.zeros_like(old_frame)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is None or st is None:
                break
            good_new = p1[st.flatten() == 1]
            good_old = p0[st.flatten() == 1]
            if len(good_new) == 0:
                break
            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
            img = cv2.add(frame, mask)
            cv2.imshow('Lucas-Kanade Optical Flow', img)
            if cv2.waitKey(30) & 0xFF == 27:
                break
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

def farneback_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        cap.release()
        return
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    try:
        while True:
            ret, frame2 = cap.read()
            if not ret:
                break
            frame_next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, frame_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('Farneback Optical Flow', rgb)
            if cv2.waitKey(30) & 0xFF == 27:
                break
            prvs = frame_next
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"c:\Users\Hp\Downloads\hi.mp4"
    if not os.path.isfile(video_path):
        print("Video file not found:", video_path)
    else:
        print("Select Optical Flow Method:")
        print("1. Lucas-Kanade (Sparse)")
        print("2. Farneback (Dense)")
        choice = input("Enter choice: ").strip()
        if choice == "1":
            lucas_kanade_optical_flow(video_path)
        else:
            farneback_optical_flow(video_path)
