%% Load images
%[he,map] = imread('D:\ShachiLab\Mukul\img\he\roi_copy2_20x.tif');
[he,map] = imread('E:\transfer_roi\he\43028.tif');
%ir = imread('D:\ShachiLab\Mukul\Registration_matlab\results\registered\im_ir_not.png');
ir = imread('E:\transfer_roi\ir\2_final.tif');

%% get the moving and fixed images

[h,w,d] = size(he);
%[h,w] = size(ir);
%ir = rgb2gray(ir);
fixed = imresize(ir, [h,w], 'nearest');
%fixed = ir;
%%imwrite(fixed,'fixed_resized.png')
%moving1 = he(:,:,1);
moving = rgb2gray(he);

%moving = imresize(moving, [h,w], 'nearest');
%%imwrite(moving,'moving_gray.png')
tic;
%% Create the optimizer and metric, using modality to the 'monomodel' just to compare the result
[optimizer,metric] = imregconfig('monomodel');
optimizer.MaximumIterations = 1000;
%%
startup;
tform = imregtform(moving,fixed,'affine',optimizer,metric,'DisplayOptimization',true);

%%
duration1 = toc;
fprintf(' duration1');
disp(duration1);
%moving_reg1 = imwarp(moving, tform,'OutputView',imref2d(size(fixed)));
%%
%tform_final is the final transformation matrix
tic;
[registered1, imref, tform_final] = imregister1(moving, fixed, 'affine', optimizer, metric, 'InitialTransformation', tform, 'DisplayOptimization', true);
imshowpair(registered1, fixed)
%moving_reg1 = imwarp(moving, tform,'OutputView',imref2d(size(fixed)))    ;
duration2 = toc;
fprintf(' duration2');
disp(duration2);
%%
imwrite(registered1, 'registered_gradient_descent.tif');
%% final HE 
%HE_reg = cat(3,registered1,registered2,registered3);
%imshow(HE_reg)
