import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class BallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball(self, players, ball_bbox):
        """
        Assigns the ball to the player closest to it based on the players' bounding boxes.
        
        Parameters:
            players (dict): A dictionary containing player information with player IDs as keys and bounding boxes as values.
            ball_bbox (tuple): A tuple representing the bounding box of the ball.
        
        Returns:
            int: The ID of the player assigned to the ball.
        """

        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id
                    
        return assigned_player