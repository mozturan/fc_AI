from utils import read_video, save_video
from trackers import Tracker
def main():

    # Read and save a video
    video_path = "input_videos/08fd33_4.mp4"
    frames = read_video(video_path)

    # init the tracker
    tracker = Tracker("models/best.pt")

    # get the tracks
    tracks = tracker.get_object_tracks(frames,
                                      read_from_stub=True,
                                      stub_path="tracks.pkl")

    #draw output 
    output_frames = tracker.draw_annotations(frames, tracks)
    # # save the tracks
    # with open('tracks.pkl', 'wb') as f:
    #     pickle.dump(tracks, f)

    save_video(output_frames, "output_videos/deneme.mp4")

if __name__ == "__main__":
    main()