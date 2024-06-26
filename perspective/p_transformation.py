import numpy as np 
import cv2

class Transformer():
    def __init__(self):
        """
        Initializes a Transformer object.

        This method initializes a Transformer object with the necessary parameters for perspective transformation.
        It sets the width and length of the football court in meters, as well as the pixel vertices that define the
        perspective transformation. The pixel vertices are stored as a NumPy array with shape (4, 2).

        Parameters:
            None

        Returns:
            None
        """

        #* football court is 105x68 meters in real
        #* 105/2 = 52.5 => 52.5/9 =5.83 => x4 = 23.32

        court_width = 68
        court_length = 23.32

        #* pixel vertices
        self.pixel_vertices = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])
        
        self.target_vertices = np.array([
            [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform(self, point):
        """
        Transforms a point based on perspective transformation.

        Parameters:
            point (numpy.ndarray): The point to be transformed.

        Returns:
            numpy.ndarray: The transformed point.
        """
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None
        
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
        return tranform_point.reshape(-1,2)
    
    def add_transformed_position_to_tracks(self,tracks):
        """
        Adds the transformed position of each track to the given tracks dictionary.

        Parameters:
            tracks (dict): A dictionary containing tracks, where each track is represented as a dictionary.

        Returns:
            None

        This function iterates over each object, object_tracks pair in the tracks dictionary. For each track, it retrieves the 'position_adjusted' key from the track_info dictionary and converts it into a numpy array. It then calls the `transform` method to transform the position. If the transformed position is not None, it squeezes the transformed position and converts it to a list before assigning it to the 'position_transformed' key in the track_info dictionary.

        Example usage:
            tracks = {
                'object1': [
                    {
                        'track_id1': {
                            'position_adjusted': [1, 2],
                            # other track info
                        }
                    },
                    # other tracks
                ],
                # other objects
            }
            transformer = PerspectiveTransformer()
            transformer.add_transformed_position_to_tracks(tracks)
            print(tracks)
            # Output:
            # {
            #     'object1': [
            #         {
            #             'track_id1': {
            #                 'position_adjusted': [1, 2],
            #                 'position_transformed': [transformed_position],
            #                 # other track info
            #             }
            #         },
            #         # other tracks
            #     ],
            #     # other objects
            # }
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed