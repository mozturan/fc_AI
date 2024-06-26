import cv2

def read_video(video_path):
    """
    Reads a video from the given video path and returns a list of frames.

    Parameters:
        video_path (str): The path to the video file.

    Returns:
        List[numpy.ndarray]: A list of frames extracted from the video.

    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path):
    """
    Saves a video from the given frames to the specified output path.

    Parameters:
        ouput_video_frames (List[numpy.ndarray]): List of video frames to save.
        output_video_path (str): The path where the video will be saved.

    Returns:
        None
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()