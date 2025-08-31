int32 track_id           # Unique ID for tracked object
string label             # Class label
float32 probability      # Confidence

# Bounding box (pixel coordinates)
float32 xmin
float32 ymin
float32 xmax
float32 ymax

# Center point (optional, derived from bbox)
float32 x
float32 y
float32 z

# Velocity estimates (optional, can be improved later)
float32 velocity_x
float32 velocity_y
float32 velocity_z