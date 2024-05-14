import pandas as pd

data = {
    'Name': 2,
    'Distance': 0.9441477402416214,
    'Direction': 1.8821175366317755,
    'Pose_estimate': [0, 1, 2, 3],  # Assuming this is a list
    'Error_covariance': [[4, 5, 6, 7], [0, 1, 2, 3]],  # Assuming this is a list
    'Time': 0.0
}

df = pd.DataFrame([data])
print(df)
