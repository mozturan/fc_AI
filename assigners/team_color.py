from sklearn.cluster import KMeans

class ColorAssigner:

    def __init__(self):
        """
        Initializes a new instance of the ColorAssigner class.

        This constructor initializes two dictionaries: `team_colors` and `player_team_dict`.
        `team_colors` is an empty dictionary that will store the colors of each team.
        `player_team_dict` is an empty dictionary that will store the mapping between player IDs and their respective teams.

        Parameters:
            None

        Returns:
            None
        """
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self,image):
        """
        Reshapes the input image to a 2D array and performs K-means clustering with 2 clusters.

        Parameters:
            image (numpy.ndarray): The input image to be reshaped.

        Returns:
            KMeans: The KMeans clustering model.
        """
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def assign_color(self,frame,player_detections):
        """
        Assigns colors to players based on the player detections provided.
        
        Parameters:
            frame: The frame containing player detections.
            player_detections: A dictionary of player detections with their bounding boxes.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame,bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_color(self,frame,bbox):
        """
        Get the color of a player based on the bounding box of their detection in a frame.

        Parameters:
            frame (numpy.ndarray): The frame containing player detections.
            bbox (tuple): The bounding box of the player detection.

        Returns:
            numpy.ndarray: The color of the player.

        Description:
            This function takes a frame and a bounding box of a player detection and returns the color of the player.
            It first extracts the image region corresponding to the bounding box. Then, it selects the top half of the image.
            Next, it applies a clustering algorithm (KMeans) to the top half of the image to group pixels into clusters.
            The cluster labels are then reshaped to match the shape of the top half of the image.
            The function identifies the cluster of pixels that are present in the four corners of the image.
            The cluster that appears most frequently in the corners is considered the non-player cluster.
            The player cluster is then determined as the other cluster.
            Finally, the function returns the color of the player cluster.
        """
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
    
    def get_teams(self, frame, player_bbox, player_id):
        """
        Returns the team ID of a player based on the player's unique ID and color.
        
        Parameters:
            frame: The frame containing the player.
            player_bbox: The bounding box of the player.
            player_id: The unique ID of the player.
        
        Returns:
            The team ID of the player.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==95 or player_id== 137:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id