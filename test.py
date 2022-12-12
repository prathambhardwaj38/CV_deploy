import threading
from typing import Union
import av
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
import os 
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def main():
    class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
        
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            
            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            out_image = frame.to_ndarray(format="bgr24")

            with self.frame_lock:
                
                self.out_image = out_image
            return out_image

    # ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)
    ctx = webrtc_streamer(
    key="snapshot",
    video_frame_callback=VideoTransformer,
    rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
    model=load_model('model_file.h5')
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

    if ctx.video_transformer:

        snap = st.button("Snapshot")
        if snap:
            with ctx.video_transformer.frame_lock:
                out_image = ctx.video_transformer.out_image

            if out_image is not None:
                
                st.write("Output image:")
                st.image(out_image, channels="BGR")
                gray=cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
                faces= faceDetect.detectMultiScale(gray, 1.3, 3)
                for x,y,w,h in faces:
                    sub_face_img=gray[y:y+h, x:x+w]
                    resized=cv2.resize(sub_face_img,(48,48))
                    normalize=resized/255.0
                    reshaped=np.reshape(normalize, (1, 48, 48, 1))
                    result=model.predict(reshaped)
                    label=np.argmax(result, axis=1)[0]
                    print(label)
                    cv2.rectangle(out_image, (x,y), (x+w, y+h), (0,0,255), 1)
                    cv2.rectangle(out_image,(x,y),(x+w,y+h),(50,50,255),2)
                    cv2.rectangle(out_image,(x,y-40),(x+w,y),(50,50,255),-1)
                    cv2.putText(out_image, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                st.image(out_image)
                my_path = os.path.abspath(os.path.dirname(__file__))       
                # cv2.imwrite(os.path.join(my_path, "../Data/"+"filename.jpg"), out_image)

            else:
                st.warning("No frames available yet.")
main()