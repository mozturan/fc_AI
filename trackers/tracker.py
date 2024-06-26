from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf,tracks):
        """
            Adds the position of each object in the tracks dictionary to the corresponding track in the tracks dictionary.
            
            Parameters:
                - sekf: The second parameter is not used in this function. It is included in the function signature by mistake.
                - tracks (dict): A dictionary containing tracks of different objects. The keys are the object names, and the values are dictionaries of tracks for each object. Each track is a dictionary of track information, including the bounding box of the object.
            
            Returns:
                None
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def position_interpolation(self, ball_positions):
        """
        Interpolates missing values in the given list of ball positions.

        Parameters:
            ball_positions (list): A list of dictionaries, where each dictionary represents a ball position.
                Each ball position is a dictionary with the following structure:
                {
                    frame_number (int): The frame number of the ball position.
                    {
                        'bbox' (list): A list of four numbers representing the bounding box coordinates of the ball.
                    }
                }

        Returns:
            list: A list of dictionaries, where each dictionary represents a ball position with interpolated missing values.
                Each ball position is a dictionary with the following structure:
                {
                    frame_number (int): The frame number of the ball position.
                    {
                        'bbox' (list): A list of four numbers representing the bounding box coordinates of the ball.
                    }
                }
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self, frames):
        """
        Detects objects in a batch of frames using a YOLO model.

        Args:
            frames (List[numpy.ndarray]): A list of frames to detect objects in.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents the detection results for a frame.
                Each detection result is a dictionary with the following structure:
                {
                    'boxes': List[List[float]],
                    'scores': List[float],
                    'labels': List[int],
                    'xyxy': List[List[float]]
                }
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            results = self.model.predict(batch_frames,
                                         conf = 0.1)
            detections.extend(results)
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Generates object tracks for a given sequence of frames.

        Args:
            frames (List[np.ndarray]): A list of frames to process.
            read_from_stub (bool, optional): Whether to read tracks from a stub file. Defaults to False.
            stub_path (str, optional): The path to the stub file. Defaults to None.

        Returns:
            dict: A dictionary containing the object tracks. The keys are "players", "referees", and "ball", and the values are dictionaries mapping track IDs to bounding box coordinates.

        Description:
            This function takes a sequence of frames and detects objects in each frame using the YOLO model. The detected objects are then tracked using the supervision.ByteTrack tracker. The object tracks are stored in a dictionary with keys "players", "referees", and "ball". Each key maps to a list of dictionaries, where each dictionary represents the bounding box coordinates of an object at a specific frame. The track IDs are used as keys in the dictionaries, and the bounding box coordinates are stored under the key "bbox". 

            If the `read_from_stub` parameter is set to True and a valid `stub_path` is provided, the function will attempt to read the object tracks from the stub file instead of generating new tracks. The stub file is a binary file that contains serialized object tracks. If the stub file exists, its contents are loaded and returned as the object tracks.

            If `stub_path` is provided, the generated object tracks are also saved to the stub file for future use.
        """
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
                
        detections = self.detect_frames(frames)

        tracks={
            "players":[], # track_id : bbox
            "referees":[], # track_id : bbox
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to Player
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
                    
            #tracker

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id ==  cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox" : bbox}

                if cls_id ==  cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox" : bbox}

                if cls_id ==  cls_names_inv['ball']:
                    tracks["ball"][frame_num][track_id] = {"bbox" : bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id= None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(frame, 
                    center = (x_center, y2), 
                    axes = (int(width), 10), 
                    angle = 0, 
                    startAngle=-45,
                    endAngle=235,
                    color = color,
                    thickness = 1,
                    lineType = cv2.LINE_AA) 

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    
    def draw_traingle(self, frame, bbox, color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
                        
    def draw_annotations(self, frames, tracks, team_ball_control):
        """
        Draw annotations on frames based on the given tracks and team ball control.

        Args:
            frames (List[np.ndarray]): A list of frames to draw annotations on.
            tracks (Dict[str, Dict[int, Dict[str, Any]]]): A dictionary containing the tracks of players, balls, and referees.
                The dictionary has the following structure:
                {
                    'players': Dict[int, Dict[str, Any]],
                    'ball': Dict[int, Dict[str, Any]],
                    'referees': Dict[int, Dict[str, Any]]
                }
                Each track is represented as a dictionary with the following structure:
                {
                    'bbox': List[float],
                    'color': Tuple[int, int, int],
                    'has_ball': bool
                }
            team_ball_control (List[int]): A list representing the team ball control for each frame.

        Returns:
            List[np.ndarray]: A list of frames with annotations drawn on them.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #drawin players 
            for track_id, player in player_dict.items():
                color = player.get("color", (0,0,0))

                frame = self.draw_ellipse(frame, player["bbox"], 
                                          color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))
            #drawing referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], 
                                          (0,0,0))

            # #drawing ball
            # for track_id, ball in ball_dict.items():
            #     frame = self.draw_ellipse(frame, ball["bbox"], 
            #                               (0,255,0), track_id)

            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_frames.append(frame)

        return output_frames

