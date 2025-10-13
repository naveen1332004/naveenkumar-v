import clib.opencv.*;
import vision.opencv.*;

tiledlayout(3,2);

BW = imread('C:\Users\navee\OneDrive\Documents\APPLY 01\MSC DS\001 CV LAB\SEND\tiger.jpg');
nexttile
plot(1,1)
imshow(BW); title("Original Image");
 BW1 = rgb2gray(BW);
 se = strel('line',11,90);
BW2 = imerode(BW1,se); 
nexttile
plot(1,2)
imshow(BW2); title("Erosion Image");

BW3 = imdilate(BW1,se);
nexttile
plot(2,1)

imshow(BW3); title("Dilation Image");

BW4 = imopen(BW1,se); 

nexttile
plot(2,2)
imshow(BW4); title("Open Image");

BW5 = imclose(BW1,se);
nexttile
plot(3,1)
imshow(BW5); title("Close Image");