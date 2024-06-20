#modules (python has them all)

#to run: python3 FinalProj.py --network=resnet18-hand /dev/video0 file.mp4

#/dev/video0 is the camera, file.mp4 holds the recording of the camera.

###BETTER TO USE IF JETSON NANO IS CONNECTED TO A MONITOR!!!!###
###SOMETIMES IT MAY BE INACCURATE DUE TO HAND MISINTERPREATION WITH RESNET18-HAND###
import sys
import argparse
import time
from random import randint
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log
feeling_brave = True # inital variable that determines if the player is willing to guess
score = 0 # inital score
# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-hand", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv) 



# keeps running till player fails
while feeling_brave:
    #1/20 chance of getting the wrong door
    ghost_door = randint(1, 20)
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  
    poses = net.Process(img)
    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)
        # determines right hand
        if pose.Links == [(0,1)]:
            print("using right hand")
            print("used door 1")
            door = 1
            door_num = randint(1, 20)
        elif pose.Links == [(0, 1), (1,2), (2,3)]: # determines right hand
            print("using right hand")
            print("used door 1")
            door = 1
            door_num = randint(1, 20)
        else: # determines left hand
            print("using left hand")
            print("used door 2")
            door = 2
            door_num = randint(1, 20)
        
        if door_num == ghost_door: # failure
            print("GHOST!")
            print("goodnight lol, door " , door, " had a ghost behind it.")
            feeling_brave = False
        else: # success, makes you go again.
            print("good job.")
            time.sleep(2)
            score = score + 1
    # render the image
    output.Render(img)

    # update the title bar with the network, fps, and your total score.
    output.SetStatus("{:s} | Network  {:.0f} FPS | Score {:.0f}".format(args.network, net.GetNetworkFPS(), score))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break