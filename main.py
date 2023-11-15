import base64
import cv2 as cv
import numpy as np
import NDIlib as ndi
import os
import pytextractor
import time

from configparser import ConfigParser
from obswebsocket import obsws, requests

class TVAdBlocker:
    def __init__(self):
        config = ConfigParser()
        config.read("config.ini")

        # datos websocket
        self.host = config["websocket"]["host"]
        self.port = config["websocket"]["port"]
        self.password = config["websocket"]["password"]
        self.ws = obsws(self.host, self.port, self.password)

        self.logo_path  = config["opencv"]["logo_path"]  # logo to detect
        self.tv_scene = config["obs"]["tv_scene"]  # OBS scene where the TV program is
        self.ad_scene = config["obs"]["ad_scene"]  # OBS scene to show during ads
        self.obs_item = config["obs"]["obs_item"]  # OBS scene to show during ads
        self.threshold = float(config["opencv"]["threshold"])  # threshold to detect the logo
        self.matchmode = config["opencv"]["matchmode"] # matchmode method algorithm
        self.sleep    = int(config["opencv"]["sleep"]) # sleep time to detect the logo
        
        self.directory = config['directory']['folder'] # IMAGE DIRECTORY
        self.imgFormat = config['directory']['format']

        #Datos
        self.path           = config["opencv"]['path']
        self.fileName       = config["opencv"]['fileName']
        self.filePath       = None
        self.fileSaveName   = config["opencv"]['fileSaveName']

        self.config         = config["tesseract"]['config']
        self.detect         = config["tesseract"]['detect']
    
    def main(self):
        try:
            self.ws.connect()

            logo = cv.imread(self.logo_path)
            directory = self.directory

            cv.startWindowThread()
            
            while True:
                scene = self.ws.call(requests.GetCurrentScene())
                sceneName = scene.getName()
                file_path = os.path.join(directory, f"{sceneName}.{self.imgFormat}")
 

                screenshot = self.ws.call(requests.TakeSourceScreenshot(sourceName=sceneName, embedPictureFormat=self.imgFormat, saveToFilePath=file_path))
                #print(screenshot.data()['sourceName'])
                
                #screenshot_data_64 = screenshot.datain['img']
                #header, encoded_64 = screenshot_data_64.split("base64,", 1)
                #screenshot_data_byte = base64.b64decode(encoded_64)
                
                screenshot_blob = cv.imread(file_path)

                # Convert the screenshot to a NumPy array
                #img = np.frombuffer(screenshot_data_byte, dtype=np.uint8)

                # Convert RGBA to BGR (OpenCV format)
                #frame = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
                #gray_frame = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
                frame = cv.cvtColor(screenshot_blob, cv.COLOR_RGBA2BGR)
                
                # Resize the logo image to be smaller than or equal to the frame image.
                logo_height, logo_width = logo.shape[:2]
                frame_height, frame_width = frame.shape[:2]

                # If the logo is larger than the frame, resize it to be smaller than or equal to the frame.
                if logo_height > frame_height or logo_width > frame_width:
                    logo = cv.resize(logo, (frame_width, frame_height))
                
                # Change the current directory  
                # to specified directory  
                os.chdir(directory)
                
                #cv.imwrite("screenshot.jpg", frame)
                #cv.imwrite("template.jpg", logo)
                    
                #TM_SQDIFF_NORMED / TM_CCORR_NORMED / TM_CCOEFF_NORMED
                #TM_SQDIFF / TM_CCORR / TM_CCOEFF
                if self.matchmode == "TM_SQDIFF":
                    matching_method = cv.TM_SQDIFF
                elif self.matchmode == "TM_SQDIFF_NORMED":
                    matching_method = cv.TM_SQDIFF_NORMED
                elif self.matchmode == "TM_CCORR":
                    matching_method = cv.TM_CCORR
                elif self.matchmode == "TM_CCORR_NORMED":
                    matching_method = cv.TM_CCORR_NORMED
                elif self.matchmode == "TM_CCOEFF":
                    matching_method = cv.TM_CCOEFF
                else:
                    matching_method = cv.TM_CCOEFF_NORMED
                    

                # Check if img is in frame
                res = cv.matchTemplate(frame, logo, matching_method)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

                # cv.imshow("ndi image", res)
                if max_val > self.threshold:
                    print("Logo detected")
                    self.ws.call(requests.SetCurrentScene(self.tv_scene))
                else:
                    #self.ws.call(requests.SetCurrentScene(self.ad_scene))
                    print("No logo detected, ads? 👀")

                time.sleep(self.sleep)

                if cv.waitKey(1) & 0xFF == 27:
                    break

            cv.destroyAllWindows()
            
        except Exception as e:
            print(f"Error: {e}")
        finally:

            self.ws.disconnect()
            return 0

if __name__ == "__main__":
    TVAdBlocker().main()