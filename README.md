# ML Pulse Detection

An attempt to use machine learning (1D CNN) to detect a pulse in labelled data collected from my capstone project.

Although originally started because I was interested to see how ML would perform for pulse detection, I quickly decided to
use this as my final project for CPEN 355. As such, it focuses more on the optimization of model accuracy for this data than
optimization for use in an embedded systems.

See the [report I created for this project](https://github.com/DuncanB6/ml_pulse_detection/blob/main/report/final_report/cpen355_report_duncan.pdf) for a detailed look at what I've done!

Duncan Boyd
duncan@wapta.ca
Mar 24, 2025

## Input Data

Data was collected on the LS-15 capstone device. This is a PPG (photoplethysmography) sensor, operated by an ESP32 C6. Data is formatted as 4 second segments of pulse data.
Segments overlap, with 1 second of new data on each subsequent sample. Samples are organized randomly. Data is collected at 60Hz (240 points/sample) and is a amplitude reading from our sensor (unitless).

Data is stored in H5 files, with each sample (4 second sequence) accompanied by a true/false label and a heartrate label in BPM if true. Data was labelled by hand, using subjective judgement as to whether or not a pulse is present. There is therefore some error to labelling.

Data is collected in a variety of locations on the body, as well as some non-pulse noise.

## Exporting Env

conda activate ml_pulse
conda env export > ml_pulse_windows.yml

## Creating Env

conda env create -f ml_pulse_windows.yml
conda activate ml_pulse

## To Explore

- skip connections
- BPM prediction
- https://www.hackster.io/news/easy-tinyml-on-esp32-and-arduino-a9dbc509f26c
