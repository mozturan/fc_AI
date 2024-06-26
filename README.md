# Football Match Analising Tool

Goal is to detect players, ball, referees and extract some features and values using **computer vision techniques** to analyse the match. What implemented so far is:

- Train **YOLOv5** with a proper football match dataset to find objects
- **Kmeans** for pixel segmentation and clustering to detect t-shirt color
- Using **optical flow** to estimate camera movement to improve analysis
- **Perspective transformation** to respresent scenes depth for better
- **Speed and distance calculation** for the players
- Different drawing styles than bbox

![11](https://github.com/mozturan/fc_AI/assets/89272933/8bf87054-c5ea-4a88-8b05-50cffa773b80)

NOTE: The software implemented on this [dataset on Kaggle](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data?select=clips) and **Yolov5** is trained with this [dataset on Roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)

