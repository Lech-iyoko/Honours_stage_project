#!/usr/bin/env python3

# import necessary libraries and modules for the script
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from keras.models import load_model
import numpy as np
from std_msgs.msg import String


class FaceRecognitionNode:
    def __init__(self):
        rospy.init_node('recognize_face_node', anonymous=True)
        self.image_sub = rospy.Subscriber("/detected_faces", Image, self.image_callback)
        self.result_pub = rospy.Publisher("/recognized_faces", String, queue_size=10)
        self.bridge = CvBridge()
        
        # Load the TensorFlow Lite model
        self.face_model = tf.lite.Interpreter(model_path='tf_converion.py')
        self.face_model.allocate_tensors()

        self.target_names = ['Lech', 'Alex', 'Imaan', 'Nick']
        
    def image_callback(self, data):
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        # Perform face recognition
        predicted_name = self.recognize_face(cv_image)
        
        # Print recognized name
        self.result_pub.publish(predicted_name)
        
    def recognize_face(self, cv_image):
        # preprocess the image
        face_resized = cv2.resize(cv_image, (160,160))
        face_resized = np.expand_dims(face_resized, axis=0) / 255.0
        
        # Perform face recognition using the loaded model
        input_details = self.face_model.get_input_details()
        output_details = self.face_model.get_output_details()
        self.face_model.set_tensor(input_details[0]['index'], face_resized)
        self.face_model.invoke()
        embedding = self.face_model.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(embedding)
        confidence = np.max(embedding)
        
        # Check if confidence is above the threshold
        if confidence > 0.8 and predicted_class < len(self.target_names):
            predicted_name = self.target_names[predicted_class]
        else:
            predicted_name = 'Unknown'
            
        return predicted_name

def main():
    # Get recognition result topic from ROS parameter server
    recognition_result_topic = rospy.get_param("/recognition_result_topic")
    
    # Create an instance of the recognize face node
    node = FaceRecognitionNode(recognition_result_topic)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        
if __name__ == '__main__':
    main()

