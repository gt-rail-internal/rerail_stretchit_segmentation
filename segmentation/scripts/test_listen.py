#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from segmentation.msg import masks_classes

class segment_resp():

    def __init__(self):
        # Subscribe to object detection message
        self.sub = rospy.Subscriber("/object_det", masks_classes, self.callback)
        self.sub

    def callback(self,data):
        print(data)
        return

def main():
    # Initialize node
    rospy.init_node('segment_resp', anonymous=True)
    segment_resp()
    rospy.spin()

if __name__ == '__main__':
    main()