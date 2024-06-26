from utils import read_video, save_video
from trackers import Tracker
from assigners import ColorAssigner, BallAssigner, FeatureAssigner
from camera import CameraMovementEstimator
from perspective import Transformer
import cv2
import numpy as np

def main():

    #* Read and save a video
    video_path = "input_videos/08fd33_4.mp4"
    frames = read_video(video_path)

    #* init the tracker
    tracker = Tracker("models/best.pt")

    #* get the tracks
    tracks = tracker.get_object_tracks(frames,
                                      read_from_stub=True,
                                      stub_path="stubs/tracks.pkl")

    tracker.add_position_to_tracks(tracks)

    #* Camera Movement Estimator
    camera_estimator = CameraMovementEstimator(frames[0])
    camera_move_per_frame = camera_estimator.get_camera_movement(frames,
                                                                 read_from_stub=True,
                                                                 stub_path="stub/camera_move.pkl")
    camera_estimator.add_adjust_positions_to_tracks(tracks,
                                                    camera_move_per_frame)
    
    #* perspective transformation
    transformer = Transformer()
    transformer.add_transformed_position_to_tracks(tracks)

    #* ball position interpolation
    tracks["ball"] = tracker.position_interpolation(tracks["ball"])

    feature_estimator = FeatureAssigner()
    feature_estimator.add_speed_and_distance_to_tracks(tracks)

    #* Get the team for each player
    color_assigner = ColorAssigner()
    color_assigner.assign_color(frames[0], 
                                tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = color_assigner.get_teams(frames[frame_num], 
                                            track["bbox"], 
                                            player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['color'] = color_assigner.team_colors[team]

    
    #* Assigning ball
    ball_assigner = BallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = ball_assigner.assign_ball(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    #* draw output 
    output_frames = tracker.draw_annotations(frames, tracks, team_ball_control)

    #* Draw camera movement
    output_frames = camera_estimator.draw_camera_movement(output_frames, camera_move_per_frame)
    
    #* Draw player feature
    output_frames = feature_estimator.draw_speed_and_distance(output_frames, tracks)

    #* save the tracks
    save_video(output_frames, "output_videos/deneme.mp4")

if __name__ == "__main__":
    main()

    # for track_id, player in tracks["players"][0].items():
    #     bbox = player['bbox']
    #     frame = frames[0]

    #     #crop image
    #     cropped = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     #save image
    #     cv2.imwrite(f"output_videos/{track_id}.jpg", cropped)

    #     break
