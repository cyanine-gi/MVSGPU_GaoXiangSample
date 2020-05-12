#encoding=utf-8
import sys 
import message_filters
from std_msgs.msg import Int32, Float32

import rospy
import rosbag
import cv2 
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import numpy as np
import pyquaternion
import se3

pose_topic = "/vins_estimator/odometry"
#pose_topic = '/vins_estimator/camera_pose'
image_topic = "/gi/forward/left/image_raw"




index = 0
bridge = CvBridge()
record_file = None
def callback(odom,img):
    # The callback processing the pairs of numbers that arrived at approximately the same time
    global index,record_file
    path = './dataset_output/images/'+str(index)+'.png'
    img_cv = bridge.imgmsg_to_cv2(img)
    cv2.imwrite(path,img_cv)
    t_ = odom.pose.pose.position
    q_ = odom.pose.pose.orientation

    pos = np.array([t_.x,t_.y,t_.z]).T
    quat = pyquaternion.Quaternion(q_.w,q_.x,q_.y,q_.z)
    #q = pyquaternion.Quaternion(axis = [0,1,0],degrees = 90)
    q = pyquaternion.Quaternion(axis = [0,1,0],degrees = 0)
    #new_q = quat*q
    #new_pos = (pos.T *q.rotation_matrix).T
    #new_q = quat*q
    #new_pos = q.rotate(pos)


    #prev_pose = quat.transformation_matrix
    #prev_pose[0:3, 3] = pos
    #prev_pose[0:3,0:3] = quat.rotation_matrix

    #new_pose = q.transformation_matrix
    #new_pose[0:3,0:3] = q.rotation_matrix
    #new_pose.setTranslation(np.array([0,0,0]))

    #transformed_pose = prev_pose*new_pose
    #new_q = pyquaternion.Quaternion(matrix = transformed_pose[0:3,0:3])
    new_q = quat*q
    new_pos = q.inverse.rotate(pos)
    print ('pos:',pos)
    print ('new_pos:',new_pos)
    part2 = ' '.join(map(str,[new_pos[0],new_pos[1],new_pos[2],new_q.x,new_q.y,new_q.z,new_q.w]))
    record_file.write(path+' '+part2+'\n')
    index+=1



def _main_():
    rospy.init_node("dataset_generator_node")
    global record_file
    record_file = open('dataset_output/record.txt','w+')
    odom_sub = message_filters.Subscriber(pose_topic,Odometry)
    img_sub = message_filters.Subscriber(image_topic,Image)
    
    ts = message_filters.ApproximateTimeSynchronizer([odom_sub, img_sub], 10, 0.04, allow_headerless=False)
    ts.registerCallback(callback)
    rospy.spin()



if __name__ == '__main__':
    _main_()
