function [val_,val_ub] = post_processor_interpolate(dir_name,gt_dir)
% run post-processor for full body

all_mats = dir(fullfile(dir_name,'*pose.mat'));

% get offbeat scores
load('models/model_ph1_ph4_binary_v2.mat');
out_dir = dir_name;
offbeat_feat = get_pruning_criteria_feats_binary(out_dir);
svm_score_v2 = [];
for i=1:size(offbeat_feat,1)
    [~,~,t_] = predict([-1],sparse(offbeat_feat(i,:)), model_,'-b 1 -q 1');
    svm_score_v2 = [svm_score_v2, t_(1)];
end;

% get indexes to process
indxs_ = [];
for i=2:numel(svm_score_v2)-1
    if svm_score_v2(i-1) > svm_score_v2(i) && svm_score_v2(i+1) > svm_score_v2(i)
        indxs_ = [indxs_, i];
    end;
end;
% keyboard;

det_all = {}; det_ub_all = {};
for i=1:numel(all_mats)
    load(fullfile(dir_name,all_mats(i).name));
    curr_det = get_det_from_boxes(boxes);
    det_all{i} = curr_det;
    det_ub_all{i} = curr_det([3,5,7,15,17,19],:);
end;

% run interpolation : v1
% for i=indxs_
%     det_all{i} = (det_all{i-1} + det_all{i+1})/2;
% end;

% run interpolation : v2
for i=indxs_
%     alfa = abs(svm_score_v2(i-1)-svm_score_v2(i)); bita = abs(svm_score_v2(i+1)-svm_score_v2(i));
%     gama = abs(svm_score_v2(i-1)+svm_score_v2(i+1) - 2*svm_score_v2(i));
%     det_all{i} = (alfa*det_all{i-1} + bita*det_all{i+1})/gama; % v2

    det_all{i} = 0.5*det_all{i-1} + 0.5*det_all{i+1}; %v1
    dt_ = det_all{i};
    det_ub_all{i} = dt_([3,5,7,15,17,19],:);
end;

vid_in_dir = gt_dir;
for i=1:length(all_mats)
    load(fullfile(vid_in_dir,all_mats(i).name));
    gt_all{i} = arr_26;
    gt_ub_all{i} = arr_26([3,5,7,15,17,19],:);
end;

[out_pck_train, score_train] = my_eval_pck(det_all,gt_all);
[out_pck_train_ub, score_train_ub] = my_eval_pck(det_ub_all,gt_ub_all,0.2);
val_ = out_pck_train; val_ub = out_pck_train_ub;
% fprintf('Post-processing pck is : %f  \n',out_pck_train/26);
% disp('done');
end

function [det_26] = get_det_from_boxes(boxes)
best_box = boxes(1,1:104);
det_26 = [];
for j=1:4:length(best_box)
    det_26 = [det_26;[mean([best_box(j),best_box(j+2)]),mean([best_box(j+1),best_box(j+3)])]];
end;
end