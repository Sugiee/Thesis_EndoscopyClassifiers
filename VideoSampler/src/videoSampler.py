import os
import numpy as np
from PIL import Image  # For saving frames as images
from moviepy import VideoFileClip

def capture(inputDir, outputDir, numFrames):
    if not os.path.exists(inputDir):
        raise FileNotFoundError(f"The file {inputDir} does not exist.")
    
    # Create output directory for this specific video
    folder = os.path.basename(os.path.dirname(inputDir))
    videoOutputFolder = os.path.join(outputDir, folder)
    
    if not os.path.exists(videoOutputFolder):
        os.makedirs(videoOutputFolder)
        
    videoName = os.path.splitext(os.path.basename(inputDir))[0]  # Get file name without extension
    videoOutputDir = os.path.join(videoOutputFolder, videoName)
        
    if not os.path.exists(videoOutputDir):
        os.makedirs(videoOutputDir)
    
    try:
        # Load the video
        clip = VideoFileClip(inputDir)
        duration = clip.duration  # Total duration of the video in seconds

        if numFrames < 1:
            raise ValueError("Number of frames to extract must be at least 1.")
        
        # Calculate timestamps for evenly spaced frames
        timestamps = [i * (duration / numFrames) for i in range(numFrames)]

        for i, t in enumerate(timestamps):
            # Define output path for the frame
            outputPath = os.path.join(videoOutputDir, f"{videoName}_{i+1:04d}.jpg")
            
            # Skip saving if the file already exists
            if os.path.exists(outputPath):
                print(f"Frame {i+1} already exists at {outputPath}. Skipping...")
                continue
            
            # Extract and save the frame
            frame = clip.get_frame(t)
            
            # Convert NumPy array to an image and save
            image = Image.fromarray(np.uint8(frame))
            image.save(outputPath)
            print(f"Saved frame {i+1} at {t:.2f}s to {outputPath}")
        
        print(f"Extracted {numFrames} frames to {videoOutputDir}.")
    
    except Exception as e:
        print(f"Failed to process video {inputDir}: {e}")

# function arguments
inputDir = r""
outputDir = r""
numFrames = 10  # Number of images to extract
failedToCapture = []

for folder in os.listdir(inputDir):
    
    folderPath = os.path.join(inputDir, folder)
    
    if os.path.isdir(folderPath):  # Check if it's a directory
        
        for fileName in os.listdir(folderPath):
            
            if fileName.endswith('.mpg'):  # Check for MP3 file extension
                
                filePath = os.path.join(folderPath, fileName)
                
                capture(filePath, outputDir, 10)  # Call capture and get video length
