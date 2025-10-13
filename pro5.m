img=imread('C:\Users\navee\OneDrive\Documents\APPLY 01\MSC DS\001 CV LAB\SEND\OIP.jpeg');
if size(img,3)==3
    img=rgb2gray(img);
end

img=double(img);

[M,N]=size(img);

histogram=zeros(1,256);
for i=1:M
    for j=1:N
        intensity=img(i,j);
        histogram(intensity+1)= histogram(intensity+1)+1;
    end
end

pdf=histogram / (M*N);

cdf=cumsum(pdf);

equalized=round(cdf*255);

output_img=zeros(M,N);
for i=1:M
    for j=1:N
        output_img(i,j)=equalized(img(i,j)+1);
    end
end

output_img=uint8(output_img);
imshow(output_img);


figure;
subplot(1,2,1);
imshow(uint8(img));
title("original image");

subplot(1,2,2);
imshow(uint8(output_img));
title('histogram equ img');

figure;
subplot(1,2,1);
imhist(uint8(img));
title('original histogram');

subplot(1,2,2);
imhist(output_img);
title('equ histogram');