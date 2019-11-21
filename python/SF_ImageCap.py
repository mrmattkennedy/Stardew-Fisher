import time
import cv2
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
        #Mouse used for pressing mouse to adjust the bar
        self.mouse = Controller()
        #Used to find the image of the fish on the screen
        self.fish_template = cv2.imread('images/fish.jpg')
        #Used to find the image of the bar on the screen
        self.bar_template = cv2.imread('images/small_bar.jpg')

        #Begins capturing the screen
        self.capture_screen()    

    #Uses template matching to find the image of the fish on the screen.
    #Once fish is found, send coordinates
    def locate_fish(self, original_img):
        img=cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        #See if the fish matches anywhere. Most accurate results with TM_SQDIFF_NORMED alg.
        res = cv2.matchTemplate(img, self.fish_template, cv2.TM_SQDIFF_NORMED)
        #Finds global min and max in an array.
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #Adjusting the last value determines how "loosely" algorithm matches.
        min_thresh = (min_val + 1e-6) * 1.2
        #Actual matching.
        match_locations = np.where(res <=min_thresh)

        #If there are more than 1 matches, than the fish is currently being caught
        #In other words, the bar is overlapping the fish, no need to move or continue
        if len(match_locations[1]) != 1:
            return None

        w, h = self.fish_template.shape[:-1]
        for (x, y) in zip(match_locations[1], match_locations[0]):
            #Draw rectangles on the original img to show location
            cv2.rectangle(original_img, (x, y), (x+w, y+h), [0, 255, 255], 2)

        #The x value never changes, it just moves up/down, so just send y value
        return match_locations[0][0]

    def locate_bar(self, original_img, fish_location):
        #If fish not found, no need to move because bar overlapping fish
        if fish_location is None:
            return
        #Just gets the image of the bar. Main idea was, instead of
        #trying to match a simple rectangle on the entire frame, limit
        #threshold to just the bar. The only square is the bar, so easy to find.
        fishing_bar = original_img[10:360, 573:597]
        #edges = cv2.Canny(fishing_bar,100,200) #thought about using edges, not the best idea.
        #cv2.imshow('', edges)
        
        img=cv2.cvtColor(fishing_bar, cv2.COLOR_BGR2RGB)

        res = cv2.matchTemplate(img, self.bar_template, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        min_thresh = (min_val + 1e-6) * 1.2
        match_locations = np.where(res <=min_thresh)

        w, h = self.bar_template.shape[:-1]
        for (x, y) in zip(match_locations[1], match_locations[0]):
            #Magic number offsets based on the fishing_bar image, so rectangles drawn at correct spot.
            cv2.rectangle(original_img, (x+550, y+24), (x+w+550, y+h+24), [0, 255, 255], 2)
        return match_locations[0][0]

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
        
        while True:
            screen = np.array(ImageGrab.grab(bbox=(40, 62, 900, 520))) #capture the window (wrote script to resize window)
            fish_location = self.locate_fish(screen) #get the location of the fish
            bar_location = self.locate_bar(screen, fish_location) #if fish found, that means fish is not being caught
            if (fish_location is not None):
                self.move_bar(fish_location, bar_location)
                
            #cv2.imshow('', screen) #see what this program sees
            #cv2.imwrite('video screenshots/' + str(num) + '.jpg', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)) #used for saving to use in video
            num += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        
stardew_fisher() 
