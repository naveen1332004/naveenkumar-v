% Parameters
SZ = 20;                
CLASS_N = 10;           
DIGITS_FN = "C:\Users\navee\onedrive\documents\APPLY 01\MSC DS\001 CV LAB\Lenna.jpg";
%% ===== Utilities =====
function vis = mosaic(w, imgs)
    % Create mosaic of images in grid
    n = numel(imgs);
    [h, w_img, c] = size(imgs{1});
    vis = zeros(h*w, w_img*w, c, 'like', imgs{1});
    for i = 1:n
        x = mod(i-1, w);
        y = floor((i-1) / w);
        if y >= w, break; end
        vis( y*h+1:(y+1)*h, x*w_img+1:(x+1)*w_img, : ) = imgs{i};
    end
end
function cells = split2d(img, cell_size, flatten)
    if nargin < 3, flatten = true; end
    [h, w] = size(img);
    sx = cell_size(1); sy = cell_size(2);
    cells = mat2cell(img, repmat(sy,1,h/sy), repmat(sx,1,w/sx));
    if flatten
        cells = reshape(cells, [],1);
    end
end
function img2 = deskew(img, SZ)
    % Deskew using image moments
    [X,Y] = meshgrid(1:SZ, 1:SZ);
    m00 = sum(img(:));
    if m00 == 0
        img2 = img; return;
    end
    m10 = sum(X(:).*double(img(:)));
    m01 = sum(Y(:).*double(img(:)));
    m20 = sum((X(:).^2).*double(img(:)));
    m02 = sum((Y(:).^2).*double(img(:)));
    m11 = sum((X(:).*Y(:)).*double(img(:)));
    mu02 = m02/m00 - (m01/m00)^2;
    mu11 = m11/m00 - (m10/m00)*(m01/m00);
    if abs(mu02) < 1e-2
        img2 = img; return;
    end
    skew = mu11/mu02;
    M = [1 skew -0.5*SZ*skew; 0 1 0];
    tform = affine2d(M');
    img2 = imwarp(img, tform, 'OutputView', imref2d([SZ SZ]));
end
function out = highlightError(img, correct)
    % Ensure RGB
    if size(img,3) == 1
        img_rgb = repmat(img,1,1,3);
    else
        img_rgb = img;
    end
    if correct
        % Highlight misclassified digits in red
        out = img_rgb;
        out(:,:,2:3) = 0;  % remove G,B channels
    else
        out = img_rgb;   % keep normal
        end
end
%% ===== Load Data =====
disp("Loading digits...");
digits_img = imread(DIGITS_FN);
if size(digits_img,3)==3
    digits_img = rgb2gray(digits_img);
end
digits = split2d(digits_img, [SZ SZ], true);
labels = repelem((0:CLASS_N-1)', numel(digits)/CLASS_N);
%% ===== Preprocessing =====
disp("Preprocessing...");
rng(321);
idx = randperm(numel(digits));
digits = digits(idx);
labels = labels(idx);
% Deskew
digits_deskewed = cellfun(@(x) deskew(x,SZ), digits, 'UniformOutput', false);
% Extract HOG features
cellSize = [10 10];
hogFeatures = cellfun(@(x) extractHOGFeatures(x,'CellSize',cellSize), ...
                      digits_deskewed, 'UniformOutput', false);
samples = vertcat(hogFeatures{:});
%% ===== Train/Test Split =====
train_n = round(0.9 * size(samples,1));
digits_train = digits_deskewed(1:train_n);
digits_test  = digits_deskewed(train_n+1:end);
samples_train = samples(1:train_n,:);
samples_test  = samples(train_n+1:end,:);
labels_train  = labels(1:train_n);
labels_test   = labels(train_n+1:end);
figure; imshow(mosaic(25, cellfun(@(x) repmat(x,1,1,3), digits_test, 'UniformOutput', false))); 
title("Test Set (10%)");
%% ===== k-Nearest Neighbors =====
disp("Training kNN...");
knn = fitcknn(samples_train, labels_train, 'NumNeighbors', 4);
resp_knn = predict(knn, samples_test);
% Evaluation
err_knn = mean(labels_test ~= resp_knn);
fprintf ('kNN Error: %.2f %%\n', err_knn*100);
conf_knn = confusionmat(labels_test, resp_knn);
disp('Confusion matrix (kNN):'); disp(conf_knn);
% Visualize errors
vis_knn = cellfun(@(img,correct) ...
    highlightError(img, correct), digits_test, num2cell(resp_knn==labels_test), 'UniformOutput', false);
figure; imshow(mosaic(25, vis_knn)); title("kNN Result");
%% ===== SVM =====
disp("Training SVM...");
svm = fitcecoc(samples_train, labels_train, ...
    'Learners', templateSVM('KernelFunction','rbf','BoxConstraint',2.67,'KernelScale',1/5.383));
resp_svm = predict(svm, samples_test);
% Evaluation
err_svm = mean(labels_test ~= resp_svm);
fprintf('SVM Error: %.2f %%\n', err_svm*100);
conf_svm = confusionmat(labels_test, resp_svm);
disp('Confusion matrix (SVM):'); disp(conf_svm);
% Visualize errors
vis_svm = cellfun(@(img,correct) ...
    highlightError(img, correct), digits_test, num2cell(resp_svm==labels_test), 'UniformOutput', false);
figure; imshow(mosaic(25, vis_svm)); title("SVM Result");
%% ===== Save SVM Model =====
disp("Saving SVM model to digits_svm.mat...");
save('digits_svm.mat','svm');
CUSTERING:
import numpy as np
import cv2 as cv
def make_gaussians(cluster_n=5, img_size=512):
    points = []
    for _ in range(cluster_n):
        center = np.random.randint(0, img_size, 2)
        cov = np.diag(np.random.randint(20, 100, 2))
        pts = np.random.multivariate_normal(center, cov, 100)
        points.extend(pts)
    points = np.float32(points)
    return points, None
def main():
    cluster_n = 5
    img_size = 512
    colors = np.zeros((1, cluster_n, 3), np.uint8)
    colors[0, :] = 255
    colors[0, :, 0] = np.arange(0, 180, 180.0 / cluster_n)
    colors = cv.cvtColor(colors, cv.COLOR_HSV2BGR)[0]
    while True:
        print('Generating Gaussian clusters...')
        points, _ = make_gaussians(cluster_n, img_size)
        term_crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        _, labels, _ = cv.kmeans(points, cluster_n, None, term_crit, 10,      cv.KMEANS_RANDOM_CENTERS)
        img = np.zeros((img_size, img_size, 3), np.uint8)
        for (x, y), label in zip(np.int32(points), labels.ravel()):
            cv.circle(img, (x, y), 1, colors[label].tolist(), -1)
       cv.imshow("K-means Clustering", img)
        key = cv.waitKey(0)
        if key == 27:  # ESC key
            break
   cv.destroyAllWindows()
if __name__ == '__main__':
    main()
