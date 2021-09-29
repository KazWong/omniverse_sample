# A 3D bounding box that can be positioned and rotated about its center (6 DOF)
# Dimensions of this box are in meters, and as such, it may be migrated to
#   another package, such as geometry_msgs, in the future.

#
string name
float64 confidence
# The 3D position and orientation of the bounding box center
geometry_msgs/Pose center

# The size of the bounding box, in meters, surrounding the object's center
#   pose.
geometry_msgs/Vector3 size
