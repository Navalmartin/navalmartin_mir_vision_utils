import cv2
from typing import Callable


def play_video(camera: cv2.VideoCapture,
               video_name: str = 'preview',
               frame_processor: Callable = None) -> None:
    """Play the video from the given camera source

    Parameters
    ----------
    camera: The camera source
    video_name: The name of the window to show the video
    frame_processor: Callable to process the frame
    Returns
    -------
    None
    """
    while True:
        success, frame = camera.read()

        if frame_processor is not None:
            frame = frame_processor(frame)

        cv2.imshow(video_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
