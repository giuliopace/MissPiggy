#!/bin/sh
echo Welcome to PigFinder2000, the app that allows you to find pigs in a Muppet Show episode.

echo 
echo The video you chose is $1
echo 

rm -r out
mkdir out
mkdir out/audio
mkdir out/audio_features
mkdir out/audio_features/audio_features
mkdir out/labelled_images

echo Extracting audio files..
ffmpeg -i $1 -vn -acodec copy out/audio.aac  > /dev/null 2>&1
ffmpeg -i out/audio.aac -c:a libmp3lame -ac 2 -q:a 2 out/audio.mp3  > /dev/null 2>&1
rm out/audio.aac
ffmpeg -i out/audio.mp3 -f segment -segment_time 0.2 -c copy out/audio/audiofile_%06d.mp3  > /dev/null 2>&1
rm out/audio.mp3

echo Extracting images..
mkdir out/images
mkdir out/images/images
ffmpeg -i $1 -f image2 -vf fps=5 out/images/images/image-%06d.jpg  > /dev/null 2>&1
echo The dataset has been created.

echo Running 
python3 -W ignore testing_ui.py 

echo all good