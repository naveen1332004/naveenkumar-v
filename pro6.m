mainimage=imread("C:\Users\navee\OneDrive\Documents\APPLY 01\MSC DS\001 CV LAB\SEND\Screenshot 2024-12-28 205340.jpg");
templet=imread("C:\Users\navee\OneDrive\Documents\APPLY 01\MSC DS\001 CV LAB\SEND\OIP (1).jpeg");

if size (mainimage,3)==3
    mainImageGray=rgb2gray(mainimage);
else
    mainImageGray=mainimage;
end

if size(templet,3)==3
    templetGray=rgb2gray(templet);
else
    templetGray=templet;
end

correlationOutput=normxcorr2(templetGray,mainImageGray);

[maxCorr,maxIndex]=max(abs9correlationOutput(:));
[yPeak,xPeak]=ind2sub(size(correlationOutput),maxIndex);

yoffset=yPeak - size(templetGray,1);
xoffset=xPeak - size(templetGray,2);


figure;
imshow(mainImage);
title("templet matching result");
hold on;

region 
rectangle('position',[xoffset,yoffset,size(templetGray,2), size(templetGray,1)],'EdgeColor','r','LineWidth',2);

hold off: