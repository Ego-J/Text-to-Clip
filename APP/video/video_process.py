from moviepy.editor  import VideoFileClip
import numpy as np
from subprocess import call

# path = "D:\\Data\\Text-to-Clip\\APP\\video\\reading.mp4"
# vfc = VideoFileClip(path).rotate(180)
# vfc.write_videofile("D:\\Data\\Text-to-Clip\\APP\\video\\reading2.mp4", codec='mpeg4', audio=False)

video_path = "D:\\Data\\Text-to-Clip\\APP\\video\\00HFP.mp4"
save_path = "D:\\Data\\Text-to-Clip\\APP\\video\\00HFP"
# call(["ffmpeg", "-i", video_path,"-r","16",save_path+"\\%06d.jpg"])
call(["ffmpeg", "-i", video_path,"-r","16","-q:v","10", save_path+"\\%06d.jpg"])