function [all_exemplar_indxs,topK_instances] = get_phasewise_exemplar_indxs(cache_dir,vid_dir,exemplar_indxs,phase_num,ex_cnsnt_val)
% get indexes to be picked in each phase of iteration

addpath('liblinear-incdec-2.01/matlab');
addpath('YR');

out_dir = fullfile(cache_dir, vid_dir,['detections/' 'ph' num2str(phase_num-1) '/' num2str(ex_cnsnt_val)]);
% out_dir = ['/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-wrapper/mat_dir/video_det/test' num2str(vid_num) '/EX_ST/ph' num2str(phase_num-1) '/detections/' num2str(ex_cnsnt_val)];

all_mats = dir(fullfile(out_dir,'*_pose.mat'));
sc_ = [];
for i=1:numel(all_mats)
    load(fullfile(out_dir,all_mats(i).name)); sc_ = [sc_,boxes(end)]; clear boxes;
end;

%new picking

offbeat_feat = get_pruning_criteria_feats_binary(out_dir); % SVM initialization, binary features
load('models/model_ph1_ph4_binary_v2.mat');
svm_score_v2 = [];
for i=1:size(offbeat_feat,1)
    [~,~,t_] = predict([-1],sparse(offbeat_feat(i,:)), model_,'-b 1 -q 1');
    svm_score_v2 = [svm_score_v2, t_(1)];
end;

old_nbds = [];
for i=1:numel(exemplar_indxs)
    old_nbds = [old_nbds,ceil(exemplar_indxs(i)/10)];
end;
old_nbds = unique(old_nbds);

new_nbds = 1:ceil(numel(all_mats)/10);
nbd_avg_onbeat= []; nbd_avg_freq = []; nbd_avg_sc= [];
for i=1:ceil(numel(all_mats)/10);
    curr_arr = (i-1)*10+1:i*10;
    curr_arr(curr_arr > numel(all_mats)) = [];
    
    nbd_avg_onbeat = [nbd_avg_onbeat,mean(svm_score_v2(curr_arr))];
    nbd_avg_freq = [nbd_avg_freq,numel(curr_arr)];
    nbd_avg_sc = [nbd_avg_sc,mean(sc_(curr_arr))];
end;

new_nbds = setdiff(new_nbds,old_nbds,'stable'); 
if numel(new_nbds) > 2
    new_nbd_avg_onbeat = nbd_avg_onbeat(new_nbds);
    new_nbd_avg_sc = nbd_avg_sc(new_nbds);
    new_nbd_avg_freq = nbd_avg_freq(new_nbds);
    
    nbd_pruned = (find(new_nbd_avg_onbeat >= mean(new_nbd_avg_onbeat)));
else
    new_nbd_avg_onbeat = nbd_avg_onbeat(new_nbds);
    new_nbd_avg_sc = nbd_avg_sc(new_nbds);
    new_nbd_avg_freq = nbd_avg_freq(new_nbds);
    nbd_pruned = 1:numel(new_nbds);
end;

n_ = new_nbd_avg_sc(nbd_pruned); nbd_sc_thresh = mean(n_);
new_nbd_sc_picked = find(n_ >= nbd_sc_thresh); total_nbds = numel(nbd_avg_onbeat);
[~,b_] = sort(n_(new_nbd_sc_picked),'descend'); 
if ceil(total_nbds/numel(new_nbd_sc_picked)) < 3
    final_num_to_pick = floor(total_nbds/3);
    nbd_final_picked = new_nbds(nbd_pruned(new_nbd_sc_picked(b_(1:final_num_to_pick))));
else
    nbd_final_picked = new_nbds(nbd_pruned(new_nbd_sc_picked(b_)));
end;

topK_instances = [];
for nbd_picked = nbd_final_picked
    % pick instances from nbd_picked
    
    curr_nbd_pick_arr = (nbd_picked-1)*10+1:nbd_picked*10;
    curr_nbd_pick_arr(curr_nbd_pick_arr > numel(all_mats)) = [];
    
    %diggy: add first OFFBEAT picking and on clash, go with scores for phase2,
    
    pick_ = [];
    
    to_pick = min(numel(curr_nbd_pick_arr),3);
    curr_nbd_pick_sc = svm_score_v2(curr_nbd_pick_arr);
    curr_nbd_pick_sc_ = sc_(curr_nbd_pick_arr);
    while numel(pick_) < to_pick
        [val_,indx_] = sort(curr_nbd_pick_sc,'descend');
        to_consider = find(val_ == max(val_));
        pick_left = to_pick - numel(pick_);
        
        if numel(to_consider) <= pick_left
            pick_ = [pick_ , curr_nbd_pick_arr(indx_(to_consider))];
        else
            local_score = curr_nbd_pick_sc_(indx_(to_consider));
            [val__,indx__] = sort(local_score,'descend');
            pick_ = [pick_, curr_nbd_pick_arr(indx_(to_consider(indx__(1:pick_left))))];
        end;
        
        curr_nbd_pick_arr = setdiff(curr_nbd_pick_arr, pick_, 'stable');
        curr_nbd_pick_sc = svm_score_v2(curr_nbd_pick_arr);
        curr_nbd_pick_sc_ = sc_(curr_nbd_pick_arr);
    end;
    
    topK_instances = sort([pick_, topK_instances]);
end;

all_exemplar_indxs = [topK_instances, exemplar_indxs];

end