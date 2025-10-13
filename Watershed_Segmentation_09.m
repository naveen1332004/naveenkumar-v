% Watershed Segmentation for the given image
imgPath = 'C:\Users\navee\OneDrive\Documents\APPLY 01\MSC DS\001 CV LAB\001 CV LAB\SEND\fruites.jpg';
img = imread(imgPath);

% Convert to grayscale
grayImg = rgb2gray(img);

% Use automatic thresholding to create markers
bw = imbinarize(grayImg);

% Compute the distance transform
D = -bwdist(~bw);

% Find regional minima as markers for watershed
mask = imextendedmin(D, 2);

% Modify the distance transform
D2 = imimposemin(D, mask);

% Apply watershed 
L = watershed(D2);

% Create watershed boundary mask
watershedBoundary = L == 0;

% Overlay boundaries on original image
imgOverlay = img;
imgOverlay(repmat(watershedBoundary, [1 1 3])) = 255; % white boundary

% Display results
figure;
subplot(1,2,1);
imshow(img);
title('Original Image');

subplot(1,2,2);
imshow(imgOverlay);
title('Watershed Segmentation');

