# Automated Exercise Repetition Counting Moment Classifier Labeller
A tool to label "moments" within workout exercise repetitions, to be used to train a moment classifier. 

## Installation
With a valid Python installation, navigate to the root folder of the project, and install all requirements with `pip install -r requirements.txt`.

## Preparing the script
Open [`moment_classifier.py`](./moment_classifier.py), and edit the elements in the `USER_INPUT` field. 

`ABSOLUTE_VIDEO_PATH`: path to the video input. This should be trimmed to include only repetitions.  
`OUTPUT_DIR`: the absolute path of the output directory.  
`DEFAULT_PLAYBACK_FPS`: the FPS to run the labeller at. 20 seems a good, slower speed for a 30 fps video.  
`MOMENT_MAP`: the moment map represents the key moments of a particular movement, mapping these moments to int values as an enum. Different map types are visible in [`moment_map.py`](moment_map.py).  

## Running the script
Simply run the script as per a normal python script with `python moment_classifier.py` and follow the prompts.

## Controls
| **Key** | **Effect**                          |
|---------|-------------------------------------|
| n       | Start labelling next moment         |
| space   | pause labelling                     |
| q       | quit labelling (progress not saved) |
