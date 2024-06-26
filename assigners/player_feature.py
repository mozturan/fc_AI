import cv2
import sys 
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class FeatureAssigner:
    def __init__(self) -> None:
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Adds the speed and distance information to the tracks dictionary.

        Args:
            tracks (dict): A dictionary containing the tracks information.

        Returns:
            None

        This function iterates over the objects in the tracks dictionary and calculates the speed and distance 
        information for each object and its corresponding tracks. The speed is calculated by dividing the distance 
        between the start and end positions of a track by the time elapsed between the start and end frames. The 
        speed is then converted from meters per second to kilometers per hour. The distance information is accumulated 
        for each object and track and stored in the total_distance dictionary. The speed and distance information 
        is then added to the corresponding tracks in the tracks dictionary.

        Note:
            - The function assumes that the tracks dictionary is structured as follows:
                - tracks[object][frame_num][track_id] = {'position_transformed': (x, y), ...}
            - The function assumes that the frame_window attribute is set to the desired frame window size.
            - The function assumes that the frame_rate attribute is set to the desired frame rate.
        """

        total_distance= {}
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue
            num_frames = len(object_tracks)
            for frame_num in range(0, num_frames,self.frame_window):
                last_frame = min(frame_num + self.frame_window, 
                                 num_frames-1)
                
                for track_id, track_info in object_tracks[frame_num].items():

                    #* if the player out of frame, skip
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start = object_tracks[frame_num][track_id]['position_transformed']
                    end = object_tracks[last_frame][track_id]['position_transformed']

                    if start is None or end is None:
                        continue

                    distance = measure_distance(start, end)
                    time_elapsed = (last_frame - frame_num)/ self.frame_rate
                    speed_mps = distance / time_elapsed
                    speed_kph = speed_mps * 3.6

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_kph
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       distance = track_info.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       position = list(position)
                       position[1]+=40

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            output_frames.append(frame)
        
        return output_frames
    
    