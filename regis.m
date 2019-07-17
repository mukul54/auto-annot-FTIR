function tform_all = regis(he_folder, ir_folder, out_folder, resized_ir_folder)

%register all the images of  he_folder to ir_folder 
%save registered images to out_folder and return the
%transformation matrix to tform_all
%ir_folder = 'E:\transfer_roi\ir\original_ir'
%he_folder = 'E:\transfer_roi\he\he_original'
%out_folder = 'E:\transfer_roi\result\registered\unmarked'
%resized_ir_folder = 'E:\transfer_roi\ir\unmarked_resized'
%% Load images

%[he,map] = imread('D:\ShachiLab\Mukul\img\he\roi_copy2_20x.tif');
he_images = dir(he_folder);
ir_images = dir(ir_folder);


%% Create the optimizer and metric, using modality to the 'multimodel' since the images came from different sensors
[optimizer,metric] = imregconfig('multimodal');
optimizer.InitialRadius = 6.25e-04;
optimizer.MaximumIterations = 500;
metric.NumberOfHistogramBins = 510;
%%
n1 = length(he_images);

tform_all{n1-2, 1} = [];

for k = 3: n1
    
    he = imread(fullfile(he_folder,he_images(k).name));
    [h,w,d] = size(he);
    if d > 3
        he = he(:,:,1:3);
    end
    
    ir = imread(fullfile(ir_folder, ir_images(k).name));
    
    if ndims(ir) > 3 
        ir = ir(:,:,1:3);
    end
   
    fixed = imresize(ir, [h,w], 'nearest');
    moving = rgb2gray(he);
    
    %save the resized ir image for later use
    ir_file_name = fullfile(resized_ir_folder, ir_images(k).name);
    imwrite(fixed, ir_file_name)
    %%
    %use imregtform to get initial transformation for imregister
    tic;
    tform = imregtform(moving,fixed,'affine',optimizer,metric,'DisplayOptimization',true);
    duration1 = toc;
    fprintf(' duration1');
    disp(duration1);
    
    %%
    
    tic;
    [registered1, ~, tform_final] = imregister1(moving, fixed, 'affine', optimizer, metric, 'InitialTransformation', tform, 'DisplayOptimization', true);
    duration2 = toc;
    
    %imshowpair(registered1, fixed)
    %moving_reg1 = imwarp(moving, tform,'OutputView',imref2d(size(fixed))); % this will give same output as registered
    fprintf(' duration2');
    disp(duration2);
    
    tform_all{k-2} = tform_final.T;
    out_path = fullfile(out_folder, he_images(k).name);
    imwrite(registered1, out_path);
end