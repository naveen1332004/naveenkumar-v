% Clear workspace and command window
clear; clc;
% Read the image
filename = "chessboard.jpg"
img = imread(filename);
% Detect checkerboard corners
[imagePoints, boardSize] = detectCheckerboardPoints(img);
% Check if detection was successful
if isempty(imagePoints)
    disp('No checkerboard corners detected.');
    return;
end
% Create figure with two subplots
figure;

% --- Subplot 1: Original image ---
subplot(1, 2, 1);
imshow(img);
title('Original Image');
% --- Subplot 2: Image with red dot corners ---
subplot(1, 2, 2);
imshow(img); hold on;
plot(imagePoints(:,1), imagePoints(:,2), 'ro', ...
     'MarkerSize', 6, 'LineWidth', 1.5);
title('Checkerboard Corner Detection');
