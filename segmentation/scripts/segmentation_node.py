#!/usr/bin/env python3

import rospy
from rail_manipulation_msgs.srv import SegmentObjects, SegmentObjectsRequest, SegmentObjectsResponse
from segmentation.srv import Object_detection
from stretch_fetch_grasp_bridge.srv import StretchSegmentation, StretchSegmentationRequest, StretchSegmentationResponse
from sensor_msgs.msg import Image
import numpy as np

import tf2_ros
import tf2_geometry_msgs

from sensor_msgs.msg import CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge

class SegmentationNode(object):
    def __init__(self):
        rospy.init_node('segmentation_node')
        self.bridge = CvBridge() 

        # Creating subscriber
        # Subscription to image topic
        self.img_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.img_callback)
        # Subscription to point cloud topic
        self.point_sub = rospy.Subscriber('/camera/depth/color/points',PointCloud2,self.point_cloud_callback)
        # Subscription to camera info topic
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        # Creating publishers
        # Publishing segmented point cloud
        self.segmented_cloud_pub = rospy.Publisher('/segmented_cloud', PointCloud2, queue_size=10)

        # Create the service
        # Service for segmentation
        self.segmentation_service = rospy.Service('/stretch_segmentation/segment_objects', StretchSegmentation, self.segmentation_service_callback)
        
        # Creating clients
        # Client for rail segmentation
        rospy.loginfo("Waiting for service /rail_segmentation/segment_objects")
        rospy.wait_for_service('/rail_segmentation/segment_objects')
        self.segmentation_client = rospy.ServiceProxy('/rail_segmentation/segment_objects', SegmentObjects)
        rospy.loginfo("Got service /rail_segmentation/segment_objects")
        # Client for object detection from DETIC
        rospy.loginfo("Waiting for service /object_detection")
        rospy.wait_for_service('/object_detection')
        self.detic_client = rospy.ServiceProxy('object_detection',Object_detection)
        rospy.loginfo("Got service /object_detection")


    def camera_info_callback(self, data):
        self.camera_info = data

    def img_callback(self, data):
        self.img = data

    def point_cloud_callback(self,data):
        self.point_cloud = data

    def transform_point_to_frame(self, point, from_frame, to_frame, timeout=rospy.Duration(1.0)):
        """
        Used to transform point from base_link to camera_color_optical_frame
        Transform a point from one frame to another using tf2_ros.

        :param point: A PointStamped object representing the point in the original frame.
        :param from_frame: The frame in which the point is currently.
        :param to_frame: The target frame to which the point should be transformed.
        :param timeout: The time to wait for the transformation to be available.
        
        :return: A PointStamped object of the point in the target frame.
        """
        # Initialize a tf2 buffer and listener if not already done
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        # Wait for the transform to become available
        try:
            tf_buffer.can_transform(to_frame, from_frame, rospy.Time(0), timeout)
            point_Stamped = tf2_geometry_msgs.PointStamped()
            point_Stamped.header.frame_id = from_frame
            point_Stamped.point = point

            # Transform the point to the target frame
            point_transformed = tf_buffer.transform(point_Stamped, to_frame, timeout)
            return point_transformed

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr('Error transforming point from %s to %s: %s', from_frame, to_frame, e)
            return None
    
    def project_to_image_plane(self, point_3d, fx=905.608154296875, fy=903.4915771484375, cx=644.8792114257812, cy=361.4921569824219):
        """
        Project a 3D point in the camera frame to 2D pixel coordinates.

        :param point_3d: A tuple (X, Y, Z) representing the 3D point in camera frame.
        :param fx: Focal length in x direction.
        :param fy: Focal length in y direction.
        :param cx: Optical center x coordinate.
        :param cy: Optical center y coordinate.

        :return: A tuple (u, v) representing the 2D pixel coordinates.
        """
        X = point_3d.x
        Y = point_3d.y
        Z = point_3d.z
        if Z == 0:
            raise ValueError("Z coordinate is zero, cannot project to image plane.")
        u = (fx * X) / Z + cx
        v = (fy * Y) / Z + cy
        return int(round(u)), int(round(v))  # Pixel coordinates are typically expressed as integers

    def find_object_index(self, rail_objects,detic_detection):
        """
        Match segmentation mask from DETIC to segmented point cloud from RAIL segmentation

        :param rail_objects: Segmented point cloud from RAIL segmentation
        :param detic_detection: Segmentation mask from DETIC

        :return: Index of the object of interest in the segmented point cloud
        """
        # Bounding box of the object of interest from DETIC
        bounding_coords = np.array(detic_detection.bounding_box.data)
        # Calculate center of the bounding box
        center_ooi = np.array([np.mean(np.array([bounding_coords[0],bounding_coords[2]])),np.mean(np.array([bounding_coords[1],bounding_coords[3]]))])
        
        # Calculate the center of the segmented objects from RAIL segmentation
        obj_centers = np.zeros((len(rail_objects.objects),2))
        for i in range(len(rail_objects.objects)):
            obj = rail_objects.objects[i]
            # Trasform the point to the target frame
            point_transformed = self.transform_point_to_frame(obj.centroid, 'base_link', 'camera_color_optical_frame')
            pixel_coords = self.project_to_image_plane(point_transformed.point)
            obj_centers[i,:] = pixel_coords

        # Difference between the center of the bounding box and the center of rail segmented objects 
        self.diff = obj_centers - center_ooi[np.newaxis,:]
        self.diff = np.abs(self.diff)
        self.diff = np.sum(self.diff,axis=1)

        # Index of the object of interest in the segmented point cloud
        closest_index = np.argmin(self.diff) 
        return closest_index

    def segmentation_service_callback(self, req):
        """
        Service callback that returns the segmented point cloud of the object of interest

        :param req: Request for segmentation

        :return: Response with StretchSegmentationResponse.segemented_point_cloud and StretchSegmentationResponse.success
        """
        rospy.loginfo("RAIL seg + DETIC segmentation ")
        
        # Create the response
        resp = StretchSegmentationResponse()
        
        # Call the service
        objects = self.segmentation_client()
        objects = objects.segmented_objects
        rospy.loginfo("Got objects from RAIL segmentation")
        
        # Call the service
        rospy.loginfo("Calling detic service")
        detic_detection = self.detic_client(req.object_name, self.img)
        rospy.loginfo("Got Detic Detection")
        object_index = self.find_object_index(objects,detic_detection)
        rospy.loginfo("Object_index:"+str(object_index))

        # Add the segmented objects to the response
        resp.segmented_point_cloud =  objects.objects[object_index].point_cloud #segmented_cloud
        resp.success = True
        
        return resp

def main():
    rospy.loginfo("Segmentation node")
    node = SegmentationNode()
    rospy.spin()
    

if __name__ == '__main__':
    main()
