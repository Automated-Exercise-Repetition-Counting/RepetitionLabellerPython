# Automated Exercise Repetition Counting Moment Classifier Labeller
A tool to label repetitions within workout exercises, to be used to train a repetition counting architecture. 

## Installation
With a valid Python installation, navigate to the root folder of the project, and install all requirements with `pip install -r requirements.txt`.

## Preparing the script
Open [`repetition_labeller.py`](./repetition_labeller.py), and edit the elements in the `USER_INPUT` field. 

- `ABSOLUTE_VIDEO_PATH`: path to the video input. This should be trimmed to include only repetitions.  
- `OUTPUT_DIR_BASEPATH`: the absolute basepath of the output directory. i.e. the directory into which the output folder will be placed.   
- `DEFAULT_PLAYBACK_FPS`: the FPS to run the labeller at. 20 seems a good, slower speed for a 30 fps video.  

## Running the script
Simply run the script as per a normal python script with `python repetition_labeller.py` and follow the prompts.

## Controls
| **Key** | **Effect**                          |
|---------|-------------------------------------|
| n       | Increase repetition count           |
| space   | pause labelling                     |
| r       | restart current labelling run       | 
| q       | quit labelling (progress not saved) |
