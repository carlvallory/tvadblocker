import base64
import cv2
import NDIlib as ndi
import numpy as np
import os
import pytextractor
import time
from obswebsocket import obsws, requests
from configparser import ConfigParser

#    O Orientation and script detection (OSD) only
#    1 Automatic page segmentation with OSD.
#    2 Automatic page segmentation, but no OSD, or OCR.
#    3 Fully automatic page segmentation, but no OSD. (Default)
#    4 Assume a single column of text of variable sizes.
#    5 Assume a single uniform block of vertically aligned text.
#    6 Assume a single uniform block of textJ
#    7 Treat the image as a single text line.
#    8 Treat the image as a single word.
#    9 Treat the image as a single word in a circle.
#    10 Treat the image as a single character.
#    11 Sparse text. Find as much text as possible in no particular order.
#    12 Sparse text with OSD.
#    13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract—specific.

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

        self.path           = config["opencv"]['path']
        self.fileName       = config["opencv"]['fileName']
        self.filePath       = None
        self.fileSaveName   = config["opencv"]['fileSaveName']

        self.config         = config["tesseract"]['config']
        self.detect         = config["tesseract"]['detect']
    
    def main(self):
        try:
            self.ws.connect()

            logo = cv2.imread(self.logo_path)
            directory = self.directory

            cv2.startWindowThread()
            
            while True:
                scene = self.ws.call(requests.GetCurrentScene())
                sceneName = scene.getName()
                file_path = os.path.join(directory, f"{sceneName}.{self.imgFormat}")
 

                screenshot = self.ws.call(requests.TakeSourceScreenshot(sourceName=sceneName, embedPictureFormat=self.imgFormat, saveToFilePath=file_path))
                #print(screenshot.data()['sourceName'])
                
                #screenshot_data_64 = screenshot.datain['img']
                #header, encoded_64 = screenshot_data_64.split("base64,", 1)
                #screenshot_data_byte = base64.b64decode(encoded_64)
                
                screenshot_blob = cv2.imread(file_path)

                # Convert the screenshot to a NumPy array
                #img = np.frombuffer(screenshot_data_byte, dtype=np.uint8)

                # Convert RGBA to BGR (OpenCV format)
                #frame = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
                #gray_frame = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
                frame = cv2.cvtColor(screenshot_blob, cv2.COLOR_RGBA2BGR)
                
                # Resize the logo image to be smaller than or equal to the frame image.
                logo_height, logo_width = logo.shape[:2]
                frame_height, frame_width = frame.shape[:2]

                # If the logo is larger than the frame, resize it to be smaller than or equal to the frame.
                if logo_height > frame_height or logo_width > frame_width:
                    logo = cv2.resize(logo, (frame_width, frame_height))
                
                # Change the current directory  
                # to specified directory  
                os.chdir(directory)
                
                #cv.imwrite("screenshot.jpg", frame)
                #cv.imwrite("template.jpg", logo)
                    
                #TM_SQDIFF_NORMED / TM_CCORR_NORMED / TM_CCOEFF_NORMED
                #TM_SQDIFF / TM_CCORR / TM_CCOEFF
                if self.matchmode == "TM_SQDIFF":
                    matching_method = cv2.TM_SQDIFF
                elif self.matchmode == "TM_SQDIFF_NORMED":
                    matching_method = cv2.TM_SQDIFF_NORMED
                elif self.matchmode == "TM_CCORR":
                    matching_method = cv2.TM_CCORR
                elif self.matchmode == "TM_CCORR_NORMED":
                    matching_method = cv2.TM_CCORR_NORMED
                elif self.matchmode == "TM_CCOEFF":
                    matching_method = cv2.TM_CCOEFF
                else:
                    matching_method = cv2.TM_CCOEFF_NORMED
                    

                # Check if img is in frame
                res = cv2.matchTemplate(frame, logo, matching_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                # cv.imshow("ndi image", res)
                if max_val > self.threshold:
                    print("Logo detected")
                    self.ws.call(requests.SetCurrentScene(self.tv_scene))
                else:
                    try:
                        path = self.path
                        fileName = self.fileName
                        filePath = os.path.join(path, f"{fileName}")
                        fileSaveName = self.fileSaveName
                        fileSavePath = os.path.join(path, f"{fileSaveName}")
                        code = cv2.COLOR_BGR2GRAY
                        myconfig = self.config #r"--psm 7 --oem 3" #--psm 6/7/10/11/13 --oem 3
                        flag = False
                        mylist = self.detect.split(",")


                        blob = cv2.imread(filePath)
                        frame = cv2.cvtColor(blob, code)
                        frame_height, frame_with = frame.shape[:2]
                        cv2.imwrite(fileSavePath, frame)


                        extractor = pytextractor.PyTextractor()
                        text_from_image = extractor.get_image_text(fileSavePath, config=myconfig)


                        for ele_arr in text_from_image:
                            for ele_list in mylist:
                                if ele_list.lower() in ele_arr.lower() or ele_arr.strip().lower() == ele_list.lower():
                                    flag = True
                                    break
                                else:
                                    continue
                            if flag:
                                break
                        
                        if flag == True:
                            print("Text detected")
                            self.ws.call(requests.SetCurrentScene(self.tv_scene))
                        else:
                            self.ws.call(requests.SetCurrentScene(self.ad_scene))
                            print("No Text Or Logo detected, ads? 👀")

                    except Exception as e:
                        print(f"Error: {e}")
                    finally:
                        print("Tesseract")

                time.sleep(self.sleep)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error: {e}")
        finally:

            self.ws.disconnect()
            return 0

if __name__ == "__main__":
    TVAdBlocker().main()
