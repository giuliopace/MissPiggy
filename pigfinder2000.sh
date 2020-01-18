#!/bin/sh

echo $1

mkdir audio
mkdir audio_features

ffmpeg -i $1 -vn -acodec copy audio.aac  > /dev/null 2>&1
ffmpeg -i audio.aac -c:a libmp3lame -ac 2 -q:a 2 audio.mp3  > /dev/null 2>&1
rm audio.aac
ffmpeg -i audio.mp3 -f segment -segment_time 0.2 -c copy audio/audiofile_%06d.mp3  > /dev/null 2>&1
rm audio.mp3

mkdir images
ffmpeg -i $1 -f image2 -vf fps=5 images/image-%06d.jpg  > /dev/null 2>&1
echo all done


#python python_script.py 


#rm images
#rm audio
#rm audio_features
echo all good