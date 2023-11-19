#!/usr/bin/env python3

import rospy
from segmentation.srv import Object_detection
from sensor_msgs.msg import Image

class segment_resp():

    def __init__(self):

        self.object_name = rospy.get_param('~object')

        # Subscribe to camera topic to get masks and object classes
        self.sub_cam = rospy.Subscriber("/camera/color/image_raw", Image, self.client)
        self.sub_cam
    
    def client(self,img):
        # Call object detection service
        rospy.wait_for_service('object_detection')
        try:
            object_detection = rospy.ServiceProxy('object_detection',Object_detection)
            print(self.object_name)
            resp = object_detection(self.object_name,img)
            print(resp,'here')
            return 
        
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

def main():
    # Initialize node
    rospy.init_node('segment_resp', anonymous=True)
    segment_resp()
    rospy.spin()


if __name__ == '__main__':
    main()