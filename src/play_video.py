import sys
import cv2
import numpy as np
#ffpyplayer for playing audio
from ffpyplayer.player import MediaPlayer



#find out openCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

#get episode as terminal argument
if len(sys.argv)==2:
	path = sys.argv[1]
	video_path = path
	print(video_path)
else:
	print("You need to give me an episode as an argument. For now I will give you one of mine but make sure to give it back to me eventually because I need it.")	
	video_path="dataset/movie1/Muppets-02-01-01.avi"

def PlayVideo(video_path):
	video=cv2.VideoCapture(video_path)
	 
	if int(major_ver)  < 3 :
		fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
	else :
		fps = video.get(cv2.CAP_PROP_FPS)
	print("fps:", fps)
	frametime = round((1/fps)*1000) 
	#frametime = 37 #actually between 36 and 37
	print("frametime:", frametime)	 
	
	player = MediaPlayer(video_path)
	while True:
		grabbed, frame=video.read()
		audio_frame, val = player.get_frame()
		if not grabbed:
			print("End of video")
			break
		if cv2.waitKey(frametime) & 0xFF == ord("q"):
			break
		cv2.imshow("Video", frame)
		if val != 'eof' and audio_frame is not None:
			#audio
			img, t = audio_frame
	video.release()
	cv2.destroyAllWindows()
PlayVideo(video_path)



'''
import numpy as np
import cv2

cap = cv2.VideoCapture('dataset/movie1/Muppets-02-01-01.avi')

while(cap.isOpened()):
	ret, frame = cap.read()

	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow('frame',frame)
	if cv2.waitKey(16) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
'''