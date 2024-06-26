import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        """
        Initializes a CameraMovementEstimator object with the given frame.

        Args:
            frame (numpy.ndarray): The input frame to initialize the object with.

        Returns:
            None
        """
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )

    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        """
        Calculates the camera movement for each frame in the given list of frames.

        Args:
            frames (List[numpy.ndarray]): A list of frames to calculate camera movement for.
            read_from_stub (bool, optional): Whether to read the camera movement from a stub file. Defaults to False.
            stub_path (str, optional): The path to the stub file. Defaults to None.

        Returns:
            List[List[int]]: A list of camera movement coordinates for each frame.

        Raises:
            FileNotFoundError: If the stub file does not exist and read_from_stub is True.

        Description:
            This function calculates the camera movement for each frame in the given list of frames. It first checks if the 
            camera movement should be read from a stub file. If so, it reads the camera movement from the stub file and returns 
            it. Otherwise, it initializes the camera movement list and performs optical flow tracking on each frame. It calculates 
            the maximum distance between the tracked features in each frame and updates the camera movement coordinates if the 
            maximum distance exceeds the minimum distance threshold. Finally, it returns the camera movement list.
        """
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new,old) in enumerate(zip(new_features,old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point,old_features_point)
                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point ) 
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)

            old_gray = frame_gray.copy()

        return camera_movement
    
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames
    
    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        """
        Adds the adjusted positions to the tracks based on the camera movement.

        Parameters:
            tracks (dict): A dictionary containing the tracks. The keys are objects, and the values are lists of tracks for each object.
            camera_movement_per_frame (list): A list of camera movements per frame. Each element in the list represents the camera movement for a specific frame.

        Returns:
            None
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted