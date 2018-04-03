function test_video(cache_dir,mat_file, vid_dir, img_name, exemplar_constant, phase_number, calib_flag, ex_cnsnt_pick_arr, exemplar_flag, exemplar_indxs, weights_exemplar)
% test_video : test a video sequence for diffferent phases

addpath YR;
addpath YR/visualization;
if isunix()
  addpath YR/mex_unix;
elseif ispc()
  addpath YR/mex_pc;
end

load(fullfile(mat_file));
exemplar_constant = exemplar_constant/10;

fprintf('Getting exemplar weights');  %diggy
ex_dir = fullfile(cache_dir, vid_dir,'exemplar_out');
if ~exist(ex_dir)
    system(['mkdir -p ' ex_dir]);
end;

if exemplar_flag ~= 1
    exemplar_indxs = [];
    for ph = 2:phase_number
        [exemplar_indxs, new_exemplar_indxs] = get_phasewise_exemplar_indxs(cache_dir,vid_dir,exemplar_indxs,ph,ex_cnsnt_pick_arr(ph-1));
        if isempty(new_exemplar_indxs)
            return;
        end;
    end;
    
    if ~isempty(exemplar_indxs) && phase_number ~= 2
        if numel(dir([ex_dir '/' num2str(phase_number) '/*mat'])) ~= numel(exemplar_indxs)
            fprintf('Copying previous exemplars..\n');
            if ~exist([ex_dir '/' num2str(phase_number)])
                system(['mkdir ' ex_dir '/' num2str(phase_number)]);
            end;
            system(['cp ' ex_dir '/' num2str(phase_number-1) '/*mat ' ex_dir '/' num2str(phase_number)]);
        end;
    end;
    
    weights_exemplar = {};
    for i=1:numel(exemplar_indxs)
        tmp_ = get_exemplar_weights(cache_dir,vid_dir,i,phase_number,ex_cnsnt_pick_arr);
        weights_exemplar{i} = tmp_{1};
    end;
end;
fprintf('%d\n',numel(weights_exemplar));

vid_in_dir = fullfile('dataset', vid_dir);
imlist = dir(fullfile(vid_in_dir,img_name));

%% test and visualize on the cricket-train set
out_dir = fullfile(cache_dir, vid_dir, ['detections/' 'ph' num2str(phase_number)]);
out_dir = fullfile(out_dir,num2str(floor((exemplar_constant)*10)));
if ~exist(out_dir)
    system(['mkdir -p ' out_dir]);
end;

% response output directory for storing resp_exemplar/getting calibration min-max values
resp_outdir = fullfile(cache_dir,vid_dir,['resp_exemplar/' 'ph' num2str(phase_number)]);
if ~exist(resp_outdir)
    system(['mkdir -p ' resp_outdir]);
end;

no_exemplar_detection_idx = [];

fprintf('Video set: ');

for i = 1:length(imlist)
    fprintf('%d',i);
    
    % load and display image
    im = imread(fullfile(vid_in_dir,imlist(i).name));
    
    if ~exist(fullfile(out_dir, [imlist(i).name(1:end-4) '_pose.mat']))
        curr_weights = weights_exemplar;
        
        % call detect function
        if phase_number > 1

            boxes = detect_fast_exemplar_EX_ST_autoTopK(im, model, min(model.thresh,-1), curr_weights,exemplar_constant,i,calib_flag,resp_outdir,exemplar_indxs);            
            
            if isempty(boxes)
                fprintf('Failed \n');
                no_exemplar_detection_idx = [no_exemplar_detection_idx,i];
                boxes = detect_fast(im, model, min(model.thresh,-1));
            end;
        else
            boxes = detect_fast(im, model, min(model.thresh,-1));
%             boxes = detect_fast_getscores(im, model, min(model.thresh,-1));
        end;
        
        boxes = nms(boxes, .1); % nonmaximal suppression
        boxes = boxes(1,:);
        save(fullfile(out_dir, [imlist(i).name(1:end-4) '_pose.mat']),'boxes');
    else
        load(fullfile(out_dir, [imlist(i).name(1:end-4) '_pose.mat']),'boxes');
    end;
    
    best_box = boxes(1:104);
    det_26 = [];
    for j=1:4:length(best_box)
        det_26 = [det_26;[mean([best_box(j),best_box(j+2)]),mean([best_box(j+1),best_box(j+3)])]];
    end;
    det_all{i} = det_26;
end
fprintf('\n');
% keyboard;

% get gt boxes
% for i=1:length(imlist)
%     load(fullfile(vid_in_dir,'gt',[imlist(i).name(1:end-4) '_pose.mat']));
%     gt_all{i} = arr_26;
% end;
% [out_pck_train, score_train] = my_eval_pck(det_all,gt_all);
% fprintf('Mean pck is : %f %f\n',out_pck_train,out_pck_train/26);
% disp('done');
% clear gt_all det_all;
% close all;
% 
% 
% %% Save train and test matrix scores
% [score,minscore_indxs] = sort(score_train);
% save(fullfile(out_dir,'test_results.mat'),'score_train','out_pck_train','minscore_indxs','score','no_exemplar_detection_idx');

%% Get calibration min max values
if calib_flag == 1
    all_mats = dir(fullfile(resp_outdir,'*resp.mat'));
    
    min_ = 1000; max_ = -1000;
    for i=1:numel(all_mats)
        fprintf('%d',i);
        load(fullfile(resp_outdir,all_mats(i).name));
        for j=1:numel(resp_exemplar)
            for k=1:numel(resp_exemplar{j})
                max_ = max(max_,max(resp_exemplar{j}{k}(:)));
                min_ = min(min_,min(resp_exemplar{j}{k}(:)));
            end;
        end;
    end;
    
    save(fullfile(resp_outdir,'calib.mat'),'min_','max_');
end;

end

