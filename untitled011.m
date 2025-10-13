%% Pedestrian Detection in Image using MATLAB
% File: pedestrian_detection.m

% --- Step 1: Read the image
img = imread("C:\Users\navee\OneDrive\Documents\APPLY 01\MSC DS\001 CV LAB\bus.jpg");

% --- Step 2: Initialize the pretrained people detector
% ACF = Aggregate Channel Features (works well for pedestrians)
peopleDetector = peopleDetectorACF('inria-100x41'); 

% --- Step 3: Detect pedestrians
[bboxes, scores] = detect(peopleDetector, img);

% --- Step 4: Annotate the image
detectedImg = insertObjectAnnotation(img, 'rectangle', bboxes, scores);

% --- Step 5: Display results
figure;
imshow(detectedImg);
title('Pedestrian Detection Results');