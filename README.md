# ml_pulse_detection
A side project to try using machine learning (1D CNN) to detect a pulse in labelled data collected from my capstone project.

## Exporting Env

conda activate ml_pulse
conda env export > ml_pulse.yml

## Creating Env

conda env create -f ml_pulse.yml
conda activate ml_pulse

## To Do

- Sweep and find hyperparams
- Modify to only load data once
- Add script for running model only once
- Investigate large variation in results 

## To Explore

- skip connections
- BPM prediction
- https://www.hackster.io/news/easy-tinyml-on-esp32-and-arduino-a9dbc509f26c
