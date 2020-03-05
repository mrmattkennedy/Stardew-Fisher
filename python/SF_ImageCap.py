import time
import cv2
import pdb
import sys
import numpy as np
from PIL import ImageGrab
from pynput.mouse import Button, Controller

"""
Using image analysis to catch fish in Stardew Valley.
Basic idea is to get the location of the fish, location of the bar,
and try to make them intersect by pressing the mouse (making the bar move up).
Done using opencv
"""
class stardew_fisher:
    def __init__(self):
        self.local_path = sys.path[0]
        #Mouse used for pressing mouse to adjust the bar
        self.mouse = Controller()
        #Used to find the image of the fish on the screen
        self.fish_template = cv2.imread(self.local_path[:self.local_path.rfind('\\')] + '\\images\\fish_template_home.png')
        #Used to find the image of the bar on the screen
        self.bar_template = cv2.imread(self.local_path[:self.local_path.rfind('\\')] + '\\images\\small_bar.jpg')

        #Begins capturing the screen
        self.capture_screen()

    """
    This moves the bar if necessary. There are a few cases.
    1. Bar is overlapping fish: nothing needs to happen.
    2. Mouse is being held: the bar will go up.
    3. Mouse is not being held: the mouse will go down.

    This is an insanely basic and, well, not useful algorithm. With time,
    I hope to make this into an actual q-learning function to find exact
    timings for the mouse pressing. But with no internet and laptop that
    is worth around $20, this will do for now.
    """
    def move_bar(self, fish_location, bar_location):
        print("fish:" + str(fish_location))
        print("bar:" + str(bar_location))
        print("")

        #If the fish y-coord is less than the bar y-coord, then the fish is above the bar.
        #So hold the mouse for 0.4 seconds, see if still above.
        if fish_location < bar_location:
            self.mouse.press(Button.left)
            time.sleep(0.4)
            self.mouse.release(Button.left)
        
    def capture_screen(self):
        num = 0
        arr = None
        time.sleep(1)
        while True:
            box_dims = (800, 215, 840, 765)
            screen = np.array(ImageGrab.grab(bbox=box_dims)) #capture the window (wrote script to resize window)
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            cv2.imshow('', screen) #see what this program sees
            
            if arr is None:
                arr = screen
            elif len(arr.shape) == 3:
                arr = np.stack((arr, screen), axis=0)
            else:
                arr = np.vstack((arr, screen[None]))
                
            num+=1
            cv2.imwrite('models/fish_identifier/data/' + str(num) + '.jpg', screen)
            #time.sleep(0.5)
            print(num)
            if num == 200:
                np.save("models/fish_identifier/train_imgs", arr)
                break
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
stardew_fisher() 
