function [weights_exemplar] = get_exemplar_weights(cache_dir,vid_dir,fr_indx,phase_number,ex_cnsnt_pick_arr) 
% Load synthesized E-SVM part filters


%load exemplar_indxs
exemplar_indxs = []; 
for ph = 2:phase_number
    [exemplar_indxs, new_exemplar_indxs] = get_phasewise_exemplar_indxs(cache_dir,vid_dir,exemplar_indxs,ph,ex_cnsnt_pick_arr(ph-1));
end;

addpath('exemplarsvm');

weightout_dir = fullfile(cache_dir,vid_dir,['exemplar_out/' num2str(phase_number)]);   % temp manual
if ~exist(weightout_dir)
    system(['mkdir -p ' weightout_dir]);
end;

weightout_mat = fullfile(weightout_dir,[num2str(exemplar_indxs(fr_indx)) '_weights.mat']);
if exist(weightout_mat)
    load(weightout_mat);
    return;
end;

% compile;

%load YR-BASE model
load('models/PARSE_305_final_66666666666666666666666666.mat');

ims = ['dataset/' vid_dir];
imlist = dir(fullfile(ims,'*png'));

indx = 1;
scale_arr = [];
pyrascale_arr = {};

temp_dir = fullfile(cache_dir, 'cache_out');
if ~exist(temp_dir)
    system(['mkdir ' temp_dir]);
end;


fprintf('scales are: ');
for fr = exemplar_indxs(fr_indx)
    
    %load image and get feature pyramid
%     im_path = sprintf(ims,fr);
    im_path = fullfile(ims,imlist(fr).name);
    im = imread(im_path);
    pyra     = featpyramid(im,model);
    interval = model.interval;
    levels   = 1:length(pyra.feat);

    %load detected pose: gives boxes %diggy
    load(fullfile(cache_dir, vid_dir, ['detections' '/ph' num2str(phase_number-1) '/' num2str(ex_cnsnt_pick_arr(phase_number-1)) '/' imlist(fr).name(1:end-4) '_pose.mat']));

    %get optimal scale from predicted pose
    [ScO, det_points] = get_optimal_scale(im,boxes,(model.sbin./pyra.scale));

    fprintf('%d ',ScO);
    scale_arr = [scale_arr;ScO];
    pyrascale_arr{end+1} = pyra.scale;
    
    %GET EXEMPLAR WEIGHT AT SCALE ScO
    optimal_feats = pyra.feat{ScO};
    optimal_scale = pyra.scale(ScO);
    
    % points at the scale of ScO's feature map and translate by 3 (cuz padding)
    ScO_det_points = round(det_points/optimal_scale)+3;
    
    % learn the w_ex using esvm
    curr_scale = model.sbin/pyra.scale(ScO);
    im_ = imresize(im,curr_scale); im_ = padarray(im_,[25 25],255);
    det_ = ceil(det_points*curr_scale); det_ = det_ + 25;
    
    for i=1:size(det_,1)
        try
            curr_im = im_(det_(i,2)-13:det_(i,2)+14,det_(i,1)-13:det_(i,1)+14,:);
            vid_name = strtok(vid_dir,'/');
            curr_path = fullfile(temp_dir,[vid_name(vid_name >= '0' & vid_name <= '9') '_' num2str(fr) '_' num2str(i) '.png']);
            imwrite(curr_im,curr_path);
            tic;
%             keyboard;
            [models,~] = dg_esvm_demo_train_poseparts_fast(curr_path,num2str(i),'negpose_data');
            time_taken = toc;
            fprintf('Time %d part took %f\n',i,time_taken);
            curr_weight{i} = models{1}.model.w;
            all_models{i} = models;
            
        catch
            fprintf('Need to do something about bb going out of bounds. Think for a while!\n');
            keyboard;
        end;
    end;
    
    weights_exemplar{indx} = curr_weight;
    indx = indx + 1;
    
end
fprintf('\n');
save(weightout_mat,'weights_exemplar','all_models');

end

% return optimal scale of human in an image from det pose points
function [ScO,pose_arr] = get_optimal_scale(im,det_boxes,pyra_scales)

% gives det_pose_points
pose_arr = zeros(26,2);
for i=1:4:length(det_boxes)-2
    indx = ceil(i/4);
    pose_arr(indx,1) = (det_boxes(i) + det_boxes(i+2))/2;
    pose_arr(indx,2) = (det_boxes(i+1) + det_boxes(i+3))/2;
end;
                     
load('util_mats/ratio_r.mat');        %gives ratio mat

% code copied from point2box.m in learning
pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
len = zeros(1,length(pa)-1);
for p = 2:size(pose_arr,1)
    len(1,p-1) = norm(abs(pose_arr(p,1:2)-pose_arr(pa(p),1:2)));
end

ratio = len(1,:)./r;
boxsize(1) = quantile(ratio,0.75);

boxes = zeros(4,26);
boxes(1,:) = pose_arr(:,1) - boxsize(1)/2;
boxes(2,:) = pose_arr(:,2) - boxsize(1)/2;
boxes(3,:) = pose_arr(:,1) + boxsize(1)/2;
boxes(4,:) = pose_arr(:,2) + boxsize(1)/2;

% minsize = prod(model.maxsize*model.sbin);
minsize = 20*20;
box_im = double(im(ceil(boxes(2,2)):ceil(boxes(4,2)),ceil(boxes(1,2)):ceil(boxes(3,2)),:));

ScO = 1;
for i=1:numel(pyra_scales)
    scaled = resize(box_im,pyra_scales(i));
    if size(scaled,1)*size(scaled,2) > minsize
        ScO = i;
    end;
end;

% fprintf('Done\n');
end

