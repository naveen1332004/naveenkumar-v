img1 = imread( "C:\Users\navee\OneDrive\Documents\APPLY 01\MSC DS\001 CV LAB\001 CV LAB\SEND\Lena.png.jpg");
img2 = imread("C:\Users\navee\OneDrive\Documents\APPLY 01\MSC DS\001 CV LAB\Lenna.jpg");
if size(img1,3) == 3
    img1 = rgb2gray(img1);
end
if size(img2,3) == 3
    img2 = rgb2gray(img2);
end
disp(['MATLAB Version: ', version]);
pts1 = detectSIFTFeatures(img1);
pts2 = detectSIFTFeatures(img2);
[features1, validPts1] = extractFeatures(img1, pts1);
[features2, validPts2] = extractFeatures(img2, pts2);
indexPairs = matchFeatures(features1, features2, ...
    'MatchThreshold', 100, 'MaxRatio', 0.7); % 0.7 as in Lowe's paper
matchedPts1 = validPts1(indexPairs (: ,1));
matchedPts2 = validPts2(indexPairs (: ,2));
figure;
showMatchedFeatures (img1, img2, matchedPts1, matchedPts2, 'montage');
title ('SIFT Feature Matching using Lowe''s Ratio Test');
