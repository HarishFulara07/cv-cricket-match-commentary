function run_half
% run_half : run "Fine-Tuning Human Pose Estimation" for video sequences
% pose setting: Upper Human Body parts

addpath(genpath('exemplarsvm'));
addpath('liblinear-incdec-2.01/matlab');
addpath('YR');

vid_dir = 'seq18';
cache_dir = 'CACHE/';

mat_file = 'models/FLIC_final_666666666666666666.mat';
all_imgs = dir(fullfile('dataset',vid_dir,'*.png'));

try
    ex_cnst_arr = [0];
    test_video_half(cache_dir,mat_file,vid_dir,0,1,0,ex_cnst_arr,0,[],{});
    
    for phase_num = 2:10
        
        % insert stopping criteria
        
        fprintf('Running phase %d\n',phase_num);
        test_video_half(cache_dir,mat_file,vid_dir,0,phase_num,1,ex_cnst_arr,0,[],{});
        
        %         keyboard;
        %load exemplars here and run inference for rest
        exemplar_indxs = [];
        for ph = 2:phase_num
            [exemplar_indxs, new_exemplar_indxs] = get_phasewise_exemplar_indxs_half(cache_dir,vid_dir,exemplar_indxs,ph,ex_cnst_arr(ph-1));
        end;
        weights_exemplar = {}; fprintf('Exemplars: ');
        for i=1:numel(exemplar_indxs)
            fprintf('%d ',exemplar_indxs(i));
            tmp_ = get_exemplar_weights_half(cache_dir,vid_dir,i,phase_num,ex_cnst_arr);
            weights_exemplar{i} = tmp_{1};
        end;
        fprintf('\n');
        
        for i=1:10
            test_video_half(cache_dir,mat_file,vid_dir,i,phase_num,0,ex_cnst_arr,1,exemplar_indxs,weights_exemplar);
        end;
        ex_cnst_arr = [ex_cnst_arr, get_suboptimal_ex_cnst(cache_dir,vid_dir, phase_num)];
        out_mat = fullfile(cache_dir, vid_dir, 'ex_cnst_final_svm_r1.mat');
        
        save(out_mat,'ex_cnst_arr');
        
        %stopping criteria
        if ceil(numel(exemplar_indxs)/3)/ceil(numel(all_imgs)/10) > 0.6
            break;
        end;
    end;
    out_mat = fullfile(cache_dir, vid_dir, 'ex_cnst_final_svm_r1.mat');
    save(out_mat,'ex_cnst_arr');
catch
    fprintf('Stoppped! PIT CHECK!\n'); keyboard;
end;

% Run evaluation and post-processor if gt available
gt_dir = fullfile('dataset',vid_dir,'gt');
if exist(gt_dir)
    det_dir = fullfile(cache_dir, vid_dir, ['detections/ph' num2str(numel(ex_cnst_arr))], num2str(ex_cnst_arr(end)));
    [out_pck_test, score_test] = evaluate_dir_half(cache_dir,vid_dir,det_dir);
    fprintf('Mean pck is : %.4f\n',out_pck_test/6);
    save(fullfile(cache_dir, vid_dir,'test_results.mat'),'score_test','out_pck_test');
    
    %Run post-processor
    [pp_pck] = post_processor_interpolate_half(det_dir,gt_dir);
    fprintf('Post processor pck is : %.4f\n',pp_pck/6);
end;

% Move final estimations 
fprintf('Copying to final_detection directory\n');
det_dir = fullfile(cache_dir, vid_dir, ['detections/ph' num2str(numel(ex_cnst_arr))], num2str(ex_cnst_arr(end)));
out_dir = fullfile(cache_dir, vid_dir, 'detections_final');
if ~exist(out_dir)
    system(['mkdir ' out_dir]);
end;
cmnd = ['cp ' det_dir '/*_pose.mat ' out_dir];
system(cmnd);
fprintf('Done!\n');


end

function suboptimal_ex_cnst = get_suboptimal_ex_cnst(cache_dir,vid_dir,phase_num)

offbeat_score = zeros(1,11);
load('models/model_test17_v3_PIW_binary.mat');
for i=0:10
    in_dir = fullfile(cache_dir, vid_dir, 'detections', ['ph' num2str(phase_num) '/' num2str(i)]);
    offbeat_feat = get_pruning_criteria_feats_binary_half(in_dir); % SVM initialization, binary features
    svm_score_v2 = [];
    for j=1:size(offbeat_feat,1)
        [~,~,t_] = predict([-1],sparse(offbeat_feat(j,:)), model_,'-b 1 -q 1');
        svm_score_v2 = [svm_score_v2, t_(1)];
    end;
    offbeat_score(i+1) = sum(svm_score_v2);
end;

flag = 0;
for i=1:numel(offbeat_score)-1
    if offbeat_score(i) >= offbeat_score(i+1)
        suboptimal_ex_cnst = i-1;
        flag = 1;
        break;
    end;
end;
if flag == 0
    suboptimal_ex_cnst = 10;
end;

end
