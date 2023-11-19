#!/usr/bin/env python3

import rospy
import sys
sys.path.append("/home/tofunmi/rerail_stretchit_segmentation/Detic")
from demo_rerail import *
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from segmentation.msg import masks_classes
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
        self.object_name = rospy.get_param('~object')

        # Subscribe to camera topic to get masks and object classes
        self.sub_cam = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.sub_cam

        # Publish masks and classes
        self.pub_object_det = rospy.Publisher('object_det',masks_classes, queue_size=10)

    # Process images and get mask function
    def callback(self,data):
        # Get image and convert to cv2 format
        img = data
        filename = 'ros_test.png'
        cv_image = bridge.imgmsg_to_cv2(img,"bgr8")
        cv_image = cv2.rotate(cv_image,cv2.ROTATE_90_CLOCKWISE)
        cv_image = cv2.flip(cv_image,1)
        cv2.imwrite(filename, cv_image) 
        cv2.imshow('image',cv_image)
        cv2.waitKey(3)

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

        #  Define the messages
        msg_masks_x = Int32MultiArray(data=x)
        msg_masks_y = Int32MultiArray(data=y)
        msg_image_size = Int32MultiArray(data = [np.array(self.mask).shape[0],np.array(self.mask).shape[1]])
        msg_box = Float32MultiArray(data=self.box)
        msg_confidence = self.confidence

        msg_object_det = masks_classes(masks_x=msg_masks_x,masks_y=msg_masks_y,classes=self.classes,image_size=msg_image_size,
                                    bounding_box=msg_box,confidence=msg_confidence)
        

        #  Publish messages
        rospy.loginfo("Publishing masks and corresponding classes")
        self.pub_object_det.publish(msg_object_det)
        rate = rospy.Rate(10) # 10hz
        rate.sleep()
        return
        

def main():

    # Initialize node
    rospy.init_node('segment', anonymous=True)
    segment()
    rospy.spin()


if __name__ == '__main__':
    main()
