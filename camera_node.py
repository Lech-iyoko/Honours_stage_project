#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


class FaceDetectionNode:
    def __init__(self):
        # Initialize node 
        rospy.init_node('detect_face_node', anonymous=True)
        
        # Create publisher and subscribe to the camera topic
        self.image_sub = rospy.Subscriber("/camera/images_raw", Image, self.image_callback)
        self.image_pub = rospy.Publisher("detected_faces", Image, queue_size=10)
        self.bridge = CvBridge()
        
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw bounding boxes around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)
            
    def preprocess_image(self, image):
        # Define image augmentation parameters
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Apply preprocessing steps
        preprocessed_image = datagen.apply_transform(image, {'brightness': 0.2})  # Example transformation
        
        return preprocessed_image
            
def main():
    # Get camera topic from ROS parameter server
    camera_topic = rospy.get_param("/camera_topic")
    
    # Create an instance for the detect face node 
    node = FaceDetectionNode()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        
if __name__ == '__main__':
    main()

