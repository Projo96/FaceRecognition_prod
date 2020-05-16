# Ubble Computer Vision home assignment

Welcome to Ubble Computer Vision home Assignment  ! Your goal will be to create a mini-ubble project, enjoy !

## Overview

As it is, the project allows to run a live face detection on the webcam, with landmarks computation.
You can access these landmarks in `face.py` file, as an output of the `compute_landmarks` method.  

## Requirements

- Python 3.6
- Librairies in requirements.txt

## Getting started

- We suggest creating a new virtualenv and run `pip install -r requirements.txt`.
- Run `python main.py`

## Objectives

- The goal of this assignment is to build an algorithm which can differentiate real people from
 spoofing attempts based on a video of the face.
 Spoofing attempts can include people filming a printed picture or a screen.
- To help you in this task you will be provided with a dataset (`video_dataset_for_home_assignment.zip`) on which you'll be able
 to test your solution. This dataset contains two types of videos:
    - videos of attempted spoofing
    - videos of real faces
- All these videos have been taken in "production conditions", which means that every user
 on these videos received the instruction to turn their heads sideways. This should allow
  you to facilitate the face validation process.


## Evaluation criteria & constraints

You will be evaluated on:
- Your capacity to solve this problem in a pragmatic way
- Your capacity to explain clearly your approach step by step, what you tried, what failed, what worked and looked promising, and things you would've liked to try.
- Your capacity to write clean, understandable and well commented code.

## How to submit your proposal

You should have received a compressed version of this repository. To submit your answer, simply commit your changes in a new branch and return us a compressed version of the updated repository.
Please join any material which might have helped (research papers, github repos, etc...).
