from navalmartin_mir_vision_utils.mir_vision_config import WITH_CV2

if WITH_CV2:

    import cv2
else:
    raise NotImplementedError("Example requires OpenCV support but this was not detected.")

from navalmartin_mir_vision_utils.video_utils.utils import play_video


if __name__ == '__main__':

    # get the camera with id = 0
    camera = cv2.VideoCapture(0)

    play_video(camera=camera)
