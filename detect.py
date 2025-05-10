import torch
import cv2
import ctypes

model = torch.hub.load('ultralytics/yolov5','custom', path='C:/PycharmProjects/fire3/yolov5/runs/train/exp10/weights/best.pt')  # Замените путь на ваш
def detect_fire_on_video(video):
    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Подготовим объект для записи видео с аннотациями
    out = cv2.VideoWriter('output_video_with_fire_detection.avi', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(img_rgb)
        annotated_frame = results.render()[0]

        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Fire Detection', annotated_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
video = 'C:/PycharmProjects/fire3/IMG_7260.MP4'
video2 = 'C:/PycharmProjects/fire3/firee.MP4'
detect_fire_on_video(video)
detect_fire_on_video(video2)
