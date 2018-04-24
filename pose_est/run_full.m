function [pose_mat, height, width] = run_full(vid_dir, img_name, show_pose_img)
% run_full : run "Fine-Tuning Human Pose Estimation" for video sequences
% video_dir : directory containing the frames extracted from the video.
% img_name : Extension of images in video_dir (for e.g., .jpg, .png, etc.)
% show_pose_img: Whether to show the image with pose vectors plotted on it.
% pose setting: Full Human Body parts

addpath(genpath('exemplarsvm'));
addpath('liblinear-incdec-2.01/matlab');
addpath('YR');

cache_dir = 'CACHE/';

mat_file = 'models/PARSE_305_final_66666666666666666666666666.mat';
all_imgs = dir(fullfile('dataset',vid_dir,img_name));

try
    ex_cnst_arr = [0];
    test_video(cache_dir,mat_file,vid_dir,img_name,0,1,0,ex_cnst_arr,0,[],{});

    for phase_num = 2:10

        fprintf('Running phase %d\n',phase_num);
        test_video(cache_dir,mat_file,vid_dir,0,phase_num,1,ex_cnst_arr,0,[],{});

        %load exemplars here and run inference for rest
        exemplar_indxs = [];
        for ph = 2:phase_num
            [exemplar_indxs, new_exemplar_indxs] = get_phasewise_exemplar_indxs(cache_dir,vid_dir,exemplar_indxs,ph,ex_cnst_arr(ph-1));
        end;

        weights_exemplar = {}; fprintf('Exemplars: ');
        for i=1:numel(exemplar_indxs)
            fprintf('%d ',exemplar_indxs(i));
            tmp_ = get_exemplar_weights(cache_dir,vid_dir,i,phase_num,ex_cnst_arr);
            weights_exemplar{i} = tmp_{1};
        end;
        fprintf('\n'); 

        for i=1:10
            test_video(cache_dir,mat_file,vid_dir,i,phase_num,0,ex_cnst_arr,1,exemplar_indxs,weights_exemplar);
        end;
        ex_cnst_arr = [ex_cnst_arr, get_suboptimal_ex_cnst(cache_dir,vid_dir, phase_num)];
        out_mat = fullfile(cache_dir, vid_dir, 'ex_cnst_final_svm_r1.mat');

        save(out_mat,'ex_cnst_arr');

        % stopping criteria
        if ceil(numel(exemplar_indxs)/3)/ceil(all_imgs/10) > 0.6
            break;
        end;
    end;
    out_mat = fullfile(cache_dir, vid_dir, 'ex_cnst_final_svm_r1.mat');
    save(out_mat,'ex_cnst_arr');
catch
    fprintf('Stoppped! PIT CHECK!\n');
end;

% Move final estimations 
fprintf('Copying to final_detection directory\n');
det_dir = fullfile(cache_dir, vid_dir, ['detections/ph' num2str(numel(ex_cnst_arr))], num2str(ex_cnst_arr(end)));
out_dir = fullfile(cache_dir, vid_dir, 'detections_final');
disp(out_dir);
if ~exist(out_dir)
    system(['mkdir ' out_dir]);
end;
cmnd = ['cp ' det_dir '/*_pose.mat ' out_dir];
system(cmnd);
fprintf('Done!\n');
img_name_without_ext = strsplit(img_name, '.');
img_name_without_ext = char(img_name_without_ext(1));
load_path = sprintf('/home/harish/CV/cv-cricket-match-commentary/pose_est/CACHE/%s/detections_final/%s_pose.mat', vid_dir, img_name_without_ext);
load(load_path);
pose_mat = boxes(1:104);
img_dir = sprintf('dataset/%s/%s', vid_dir, img_name);
img = imread(img_dir);
[height, width, ~] = size(img);
if show_pose_img
    pose_mat_y = pose_mat(2:2:end);
    pose_mat_x = pose_mat(1:2:end);
    imshow(img);
    hold on;
    plt = plot(pose_mat_x, pose_mat_y, 'r*', 'LineWidth', 2, 'MarkerSize', 2);
    saveas(plt, sprintf('dataset/%s/pose_%s', vid_dir, img_name));
end;
end

function suboptimal_ex_cnst = get_suboptimal_ex_cnst(cache_dir,vid_dir,phase_num)

offbeat_score = zeros(1,11);
load('models/model_ph1_ph4_binary_v2.mat');
for i=0:10
    in_dir = fullfile(cache_dir, vid_dir, 'detections', ['ph' num2str(phase_num) '/' num2str(i)]);
    offbeat_feat = get_pruning_criteria_feats_binary(in_dir); % SVM initialization, binary features
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
