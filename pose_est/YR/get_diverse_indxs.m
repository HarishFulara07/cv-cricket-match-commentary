function arr20 = get_diverse_indxs(in_indxs,flag,phase_number,prev_diverse_indxs)
% flag = 1 : top20 -1 : bottom20

% because of ease of understanding: iterate from the start
if flag == 1
    in_indxs = fliplr(in_indxs);
end;

curr_indx = 1;
arr20 = [];

if flag == 1
    if phase_number == 1
        pose_path1 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase1/test3/train_detections/image_%03d_pose.mat';
%         pose_path1 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase2/test3/exemplar_out_dir_esvm_test3_full_top1/image_%03d_pose.mat';
%         pose_path2 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase1/test3/exemplar_out_dir_dvrs/ph2/1/image_%03d_pose.mat';
        im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/test_frames/test3/image_%03d.png';
    elseif phase_number == 2
%         pose_path = fullfile(vid_dir,'image%03d_pose.mat');     % detected pose path
        pose_path1 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase2/test3/exemplar_out_dir_esvm_test3_full_top1/image_%03d_pose.mat';
%         pose_path = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase2/train_detections/image%03d_pose.mat';
        im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/test_frames/test3/image_%03d.png';
    elseif phase_number == 3
        pose_path1 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase3/test3/exemplar_out_dir_esvm_test3_full_top/1/image_%03d_pose.mat';
        im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/test_frames/test3/image_%03d.png';
    elseif phase_number == 4
        pose_path1 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase4/test3/exemplar_out_dir_esvm_test3_full_top/2/image_%03d_pose.mat';
        im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/test_frames/test3/image_%03d.png';
    elseif phase_number == 5
        pose_path1 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase5/test3/exemplar_out_dir_esvm_test3_full_top/2/image_%03d_pose.mat';
        im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/test_frames/test3/image_%03d.png';
    elseif phase_number == 6
        pose_path1 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase6/test3/exemplar_out_dir_esvm_test3_full_top/1/image_%03d_pose.mat';
        im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/test_frames/test3/image_%03d.png';
    elseif phase_number == 7
        pose_path1 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase7/test3/exemplar_out_dir_esvm_test3_full_top/1/image_%03d_pose.mat';
        im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/test_frames/test3/image_%03d.png';
    elseif phase_number == 8
        pose_path1 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase8/test3/exemplar_out_dir_esvm_test3_full_top/3/image_%03d_pose.mat';
        im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/test_frames/test3/image_%03d.png';
    elseif phase_number == 9
        pose_path1 = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases_partition/phase9/test3/exemplar_out_dir_esvm_test3_full_top/3/image_%03d_pose.mat';
        im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/test_frames/test3/image_%03d.png';
    end;
    
else
    pose_path1 = '/Pulsar1/users/digvijay.singh/cricket_dataset/re_partition/train/image%03d_pose.mat';
    im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/re_partition/train/image%03d.png';
end;

% im_path = fullfile(vid_dir,'image%03d.png');    %corresponding image path
% im_path = '/Pulsar1/users/digvijay.singh/cricket_dataset/re_partition/train/image%03d.png';

all_poses = {};
all_ims = {};

% fill all_poses with pose of previous phase dvrs poses/indxs
if flag == 1
    for i=1:numel(prev_diverse_indxs)
        curr_pose = sprintf(pose_path1,prev_diverse_indxs(i));
        load(curr_pose);
        
        arr_26 = [];
        for j=1:26
            x_ = (boxes((j-1)*4+1) + boxes((j-1)*4+3))/2;
            y_ = (boxes((j-1)*4+2) + boxes((j-1)*4+4))/2;
            arr_26 = [arr_26;[x_,y_]];
        end;
        arr_14 = arr_26([1,2,3,5,7,10,12,14,15,17,19,22,24,26],:);
        curr_pose = get_angular_representation(arr_14);
        
        all_poses{i} = curr_pose;
    end;
end;

while numel(arr20) < 20 
    
    clear boxes;
    curr_pose = sprintf(pose_path1,in_indxs(curr_indx));
    load(curr_pose);
    curr_im = imread(sprintf(im_path,in_indxs(curr_indx)));
    
    if flag == 1
        arr_26 = [];
        for i=1:26
            x_ = (boxes((i-1)*4+1) + boxes((i-1)*4+3))/2;
            y_ = (boxes((i-1)*4+2) + boxes((i-1)*4+4))/2;
            arr_26 = [arr_26;[x_,y_]];
        end;
        arr_14 = arr_26([1,2,3,5,7,10,12,14,15,17,19,22,24,26],:);
    end;
    curr_pose = get_angular_representation(arr_14);
    
%     fprintf('%d: ',curr_indx);
    similar_flag = 0;
    for i=1:numel(all_poses)
        sc_pose = check_pose_similarity(curr_pose,all_poses{i});
        sc_image = 0; % check_image_similarity(curr_im,all_ims{i});
        if sc_image < 0.005
            if sc_pose == 0
                similar_flag = 1;
            end
        end;
        
%         fprintf('%d,%.3f ',sc_pose,sc_image);
    end;
%     fprintf('\n');
    
    if similar_flag == 0
        all_poses{end+1} = curr_pose;
        all_ims{end+1} = curr_im;
        arr20 = [arr20,in_indxs(curr_indx)];
    end
    
    
    curr_indx = curr_indx + 1;
end;


end

function ret_arr = get_angular_representation(arr_14)

ret_arr = [];
x_ = [1,2,3,4,3,6,7,2,9, 10,9, 12,13];
y_ = [2,3,4,5,6,7,8,9,10,11,12,13,14];

for i=1:13
    curr_angle = atan2(arr_14(x_(i),1)-arr_14(y_(i),1), arr_14(x_(i),2)-arr_14(y_(i),2));
    ret_arr = [ret_arr; cos(curr_angle),sin(curr_angle)];
end;

end

function score = check_pose_similarity(arr_1,arr_2)

tmp_score = power((arr_1(:,1)-arr_2(:,1)),2)' + power((arr_1(:,2)-arr_2(:,2)),2)';
score = sum(tmp_score > 0.15);

end

function score = check_image_similarity(arr_1,arr_2)

if length(size(arr_1)) == 2
    arr_1 = cat(3,arr_1,arr_1,arr_1);
end;
if length(size(arr_2)) == 2
    arr_2 = cat(3,arr_2,arr_2,arr_2);
end;

arr_1 = imresize(arr_1,[100,100]);
arr_2 = imresize(arr_2,[100,100]);

r_1 = arr_1(:,:,1); g_1 = arr_1(:,:,2); b_1 = arr_1(:,:,3);
r_2 = arr_2(:,:,1); g_2 = arr_2(:,:,2); b_2 = arr_2(:,:,3);

hr_1 = imhist(r_1)./numel(r_1); hg_1 = imhist(g_1)./numel(g_1); hb_1 = imhist(b_1)./numel(b_1);
hr_2 = imhist(r_2)./numel(r_2); hg_2 = imhist(g_2)./numel(g_2); hb_2 = imhist(b_2)./numel(b_2);

Fr = sum((hr_1 - hr_2).^2); Fg = sum((hg_1 - hg_2).^2); Fb = sum((hb_1 - hb_2).^2);
% score = (Fr+Fg+Fb)/3;
score = (0.2989*Fr+0.5870*Fg+0.1140*Fb);

end