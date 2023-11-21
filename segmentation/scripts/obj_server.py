#!/usr/bin/env python3

import rospy
import sys
sys.path.append("ADD PATH TO DETIC HERE")
from demo_rerail import *
import numpy as np

from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from segmentation.srv import Object_detection, Object_detectionResponse
import cv2
from cv_bridge import CvBridge
bridge = CvBridge()

class segment():
    def __init__(self):

        # Declare variables
        self.mask = []
        self.classes = []
        self.box = []
        self.confidence = []
        self.object_name = ''

        #  Call server
        s = rospy.Service('object_detection',Object_detection,self.callback)
        s

    # Process images and get mask function
    def callback(self,data):

        self.object_name = data.obj_name
        
        # Get image and convert to cv2 format
        img = data.img
        filename = 'ros_test.png'
        cv_image = bridge.imgmsg_to_cv2(img,"bgr8")
        cv_image = cv2.rotate(cv_image,cv2.ROTATE_90_CLOCKWISE)
        cv_image = cv2.flip(cv_image,1)
        cv2.imwrite(filename, cv_image) 
        cv2.destroyAllWindows()

        # Use detic for object detection
        input = filename
        output = 'cv_2.png'
        boxes,confidences,masks_dets,classes = detect(input,output)

        # Log succes
        rospy.loginfo("I received an image and I'm searching for " + str(self.object_name))

        # Get classes
        self.classes = classes

        # Get index for object of interest
        idx = np.where(np.array(self.classes) == self.object_name)[0][0]

        # Get masks for class
        self.mask = masks_dets[idx]

        # Find region of image with object
        a = np.where(np.array(self.mask)==True)
        x = list(a[0])
        y = list(a[1])

        # Get box
        self.box = list(boxes[idx])

        # Get confidence
        self.confidence = confidences[idx]
        print(self.confidence)

        #  Define the messages
        srv_masks_x = Int32MultiArray(data=x)
        srv_masks_y = Int32MultiArray(data=y)
        srv_image_size = Int32MultiArray(data = [np.array(self.mask).shape[0],np.array(self.mask).shape[1]])
        srv_box = Float32MultiArray(data=self.box)
        srv_confidence = self.confidence

        srv_object_det = Object_detectionResponse(masks_x=srv_masks_x,masks_y=srv_masks_y,classes=self.classes,image_size=srv_image_size,
                                    bounding_box=srv_box,confidence=srv_confidence)
        

        #  Respond to query
        rospy.loginfo("Returning masks and corresponding classes")
        rate = rospy.Rate(10) # 10hz
        rate.sleep()
        return srv_object_det
        

def main():

    # Initialize node
    rospy.init_node('segment', anonymous=True)
    segment()
    rospy.spin()


if __name__ == '__main__':
    main()
