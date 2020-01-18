import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import sys

from ffpyplayer.player import MediaPlayer



class MyVideoCapture:
	def __init__(self, video_source=0, audio_player = 0):
		if video_source == 0:
			print("ERROR: No video selected")
		else:
			if audio_player == 0:
				print("no audio track")

			# Open the video source
			self.vid = cv2.VideoCapture(video_source)
			if not self.vid.isOpened():
				raise ValueError("Unable to open video source", video_source)
			self.audio_player = audio_player	

			# Get video source width and height
			self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
			self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

	def get_frame(self):
		if self.vid.isOpened():
			ret, frame = self.vid.read()
			audio_frame, val = self.audio_player.get_frame()
			if ret:
				# Return a boolean success flag and the current frame converted to BGR
				return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			else:
				return (ret, None)
			if val != 'eof' and audio_frame is not None:
				#audio
				img, t = audio_frame
		else:
			return (ret, None)

	# Release the video source when the object is destroyed
	def __del__(self):
		if self.vid.isOpened():
			self.vid.release()

class App:
	def __init__(self, window, window_title, video_source=0, audio_player=0):
		self.window = window
		self.window.title(window_title)
		self.video_source = video_source
		#self.video_source = "../dataset/movie1/Muppets-02-01-01.avi"
		self.audio_player = audio_player


		# open video source (by default this will try to open the computer webcam)
		self.vid = MyVideoCapture(self.video_source, self.audio_player)

		# Create a canvas that can fit the above video source size
		self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
		self.canvas.pack()

		self.video_lbl = tkinter.Label(window, text="Pig detected in video")
		self.video_lbl.pack(anchor=tkinter.CENTER, expand=True)
		self.video_lbl['bg'] = 'red'
		
		self.audio_lbl = tkinter.Label(window, text="Pig detected in audio")
		self.audio_lbl.pack(anchor=tkinter.CENTER, expand=True)
		self.audio_lbl['bg'] = 'red'

		# Button that lets the user take a snapshot
		#self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
		#self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

		# After it is called once, the update method will be automatically called every delay milliseconds
		self.delay = 15
		self.update()

		self.window.mainloop()

	def snapshot(self):
		# Get a frame from the video source
		ret, frame = self.vid.get_frame()

		if ret:
			cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

	def update(self):
		# Get a frame from the video source
		ret, frame = self.vid.get_frame()

		if ret:
			self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
			self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

		self.window.after(self.delay, self.update)




if len(sys.argv)==2:
	path = sys.argv[1]
	video_path = path
	
	audio_player = MediaPlayer(video_path)
	

	# Create a window and pass it to the Application object
	App(tkinter.Tk(), "Pig Finder 2000", video_path, audio_player)

else:
	raise ValueError("Invalid number of arguments (source file required)")	
