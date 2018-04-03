function offbeat_feat = get_pruning_criteria_feats_binary(in_dir)
% Function producing features for pose quality ranking SVM

all_mats = dir(fullfile(in_dir,'*mat'));
flag = 0;
for i=1:numel(all_mats)
    if isequal(all_mats(i).name ,'test_results.mat')
        flag = 1;
        break;
    end;
end;
if flag == 1
    all_mats(i) = [];
end;

all_dets = {};
for i=1:numel(all_mats)
    load(fullfile(in_dir,all_mats(i).name));
    all_dets{i} = get_det_from_boxes(boxes);
end;

% fprintf('Getting neighborhood averages\n');
for i=1:numel(all_mats)
    indxs=[i-3:i-1,i+1:i+3];
    indxs(indxs<1) = []; indxs(indxs>numel(all_mats)) = [];
    if numel(indxs) ~= 6    %lame hack for starting and ending indexes
        if i < 10 % very lame
            indxs = 1:10; indxs(indxs == i) = [];
            indxs = indxs(1:6);
        else
            indxs = numel(all_mats)-10:numel(all_mats); indxs(indxs == i) = [];
            indxs = indxs(end-5:end);
        end;
    end;
    
    [nbd(i).torso_len1, nbd(i).torso_len2] = get_mean_torso_len(indxs,all_dets);
    [nbd(i).legs_len1, nbd(i).legs_len2] = get_mean_legs_len(indxs,all_dets);
    [nbd(i).arms_len1, nbd(i).arms_len2] = get_mean_arms_len(indxs,all_dets);
    [nbd(i).hip_loc1, nbd(i).hip_loc2] = get_mean_hip_loc(indxs,all_dets);
    [nbd(i).sho_loc1, nbd(i).sho_loc2] = get_mean_sho_loc(indxs,all_dets);
    [nbd(i).lrsho_dist, nbd(i).lrhip_dist] = get_mean_lr_dist(indxs,all_dets);
    nbd(i).sho_traversal_dist = get_mean_sho_traversal_dist(indxs,all_dets);
    nbd(i).pose_scale = get_mean_pose_scale(indxs,all_dets);
    nbd(i).lrlegpart_sum_dist = get_mean_lrlegpart_sum_dist(indxs,all_dets);
end;

% Iterate over each index and compute unconventionality/offbeat score
% offbeat_score = [];
offbeat_feat = [];
for i=1:numel(all_mats)
%     curr_score = 0;
    curr_feat = []; curr_det = all_dets{i};
    
    % global faults
%     curr_score = curr_score + get_torso_intersection(i,all_dets); %fprintf('Torso intersection: %f\n',curr_score);
    curr_feat = [curr_feat, get_torso_intersection(curr_det)];
%     curr_score = curr_score + 2*check_sho_swap(i,all_dets); %fprintf('Shoulder swap: %f\n',curr_score);
    curr_feat = [curr_feat, check_sho_swap(curr_det)];
%     curr_score = curr_score + 2*check_hip_swap(i,all_dets); %fprintf('Hip swap: %f\n',curr_score);
    curr_feat = [curr_feat, check_hip_swap(curr_det)];
%     curr_score = curr_score + check_unequal_torso_lines(i,all_dets); %fprintf('Unequal Torso: %f\n',curr_score);
    curr_feat = [curr_feat, check_unequal_torso_lines(curr_det)];
    
    %local faults
%     curr_score = curr_score + 0.2*get_leg_parts_intersection(i,all_dets); %fprintf('Leg joints intersection: %f\n',curr_score);
    curr_feat = [curr_feat, get_leg_parts_intersection(curr_det)];
    
    % neighborhood faults
%     curr_score = curr_score + check_torso_len(all_dets{i},nbd(i).torso_len1,nbd(i).torso_len2); %fprintf('Mean torso length: %f\n',curr_score);
    curr_feat = [curr_feat, check_torso_len(curr_det,nbd(i).torso_len1,nbd(i).torso_len2)];
%     curr_score = curr_score + 0.3*check_legs_len(all_dets{i},nbd(i).legs_len1,nbd(i).legs_len2); %fprintf('Mean legs length:%f\n',curr_score);
    curr_feat = [curr_feat, check_legs_len(curr_det,nbd(i).legs_len1,nbd(i).legs_len2)];
%     curr_score = curr_score + 0.3*check_arms_len(all_dets{i},nbd(i).arms_len1,nbd(i).arms_len2); %fprintf('Mean arms length:%f\n',curr_score);
    curr_feat = [curr_feat, check_arms_len(curr_det,nbd(i).arms_len1,nbd(i).arms_len2)];
%     curr_score = curr_score + 2*check_hip_loc(all_dets{i},nbd(i).hip_loc1, nbd(i).hip_loc2); %fprintf('Mean hip displacement: %f\n',curr_score);
    curr_feat = [curr_feat, check_hip_loc(curr_det,nbd(i).hip_loc1, nbd(i).hip_loc2)];
%     curr_score = curr_score + 2*check_sho_loc(all_dets{i},nbd(i).sho_loc1, nbd(i).sho_loc2); %fprintf('Mean shoulder displacement: %f\n',curr_score);
    curr_feat = [curr_feat, check_sho_loc(curr_det,nbd(i).sho_loc1, nbd(i).sho_loc2)];
%     curr_score = curr_score + check_lr_dist(all_dets{i},nbd(i).lrsho_dist, nbd(i).lrhip_dist); %fprintf('Mean lr-hipsho distance: %f\n',curr_score);
    curr_feat = [curr_feat, check_lr_dist(curr_det,nbd(i).lrsho_dist, nbd(i).lrhip_dist)];
%     curr_score = curr_score + check_sho_traversal_dist(all_dets{i},nbd(i).sho_traversal_dist); %fprintf('Mean shoulder traversal displacement: %f\n',curr_score);
    curr_feat = [curr_feat, check_sho_traversal_dist(curr_det,nbd(i).sho_traversal_dist)];
%     curr_score = curr_score + 2*check_pose_scale(all_dets{i},nbd(i).pose_scale); %fprintf('Mean pose scale: %f\n',curr_score);
    curr_feat = [curr_feat, check_pose_scale(curr_det,nbd(i).pose_scale)];
%     curr_score = curr_score + 0.3*check_lrlegpart_sum_dist(all_dets{i},nbd(i).lrlegpart_sum_dist); %fprintf('Mean leg joints distance sum: %f\n',curr_score);
    curr_feat = [curr_feat, check_lrlegpart_sum_dist(curr_det,nbd(i).lrlegpart_sum_dist)];
    
    curr_feat = [curr_feat, check_converging_torsos(curr_det)];
    
    curr_feat = [curr_feat, check_half_parts_breaking(curr_det)];
%     offbeat_score = [offbeat_score, curr_score];
    offbeat_feat(i,:) = curr_feat;
    %fprintf('%d: %f\n',i,curr_score);
%     load(fullfile(in_dir,all_mats(i).name)); im = imread(fullfile(img_dir,[all_mats(i).name(1:end-9) '.png']));
%     showskeletons(im, boxes, colorset, pa);
%     waitforbuttonpress; close all;
end;
% keyboard;
% keyboard;
end

function num_fails = check_half_parts_breaking(curr_det)

angle_ = [];
angle_ = [angle_,get_angle_bw_parts(3, 4, 5,curr_det)];
angle_ = [angle_,get_angle_bw_parts(5, 6, 7,curr_det)];
angle_ = [angle_,get_angle_bw_parts(15, 16, 17,curr_det)];
angle_ = [angle_,get_angle_bw_parts(17, 18, 19,curr_det)];
angle_ = [angle_,get_angle_bw_parts(10, 11, 12,curr_det)];
angle_ = [angle_,get_angle_bw_parts(12, 13, 14,curr_det)];
angle_ = [angle_,get_angle_bw_parts(22, 23, 24,curr_det)];
angle_ = [angle_,get_angle_bw_parts(24, 25, 26,curr_det)];
num_fails = numel(find(angle_ >= 30));

end

function angl = get_angle_bw_parts(indx1, indx2, indx3,det_26)

% indx1 = 12; indx2 = 13; indx3 = 14;
a1 = det_26(indx1,:); a2 = det_26(indx2,:);
b1 = det_26(indx2,:); b2 = det_26(indx3,:);

len_a = sqrt(power(a2(1)-a1(1),2) + power(a2(2)-a1(2),2));
len_b = sqrt(power(b2(1)-b1(1),2) + power(b2(2)-b1(2),2));

angl = rad2deg(acos(dot(a2-a1,b2-b1) / (len_b*len_a)));

end

function num_fails = check_converging_torsos(curr_det)
    num_fails = [0 0];
    
    %get subtle torso ratios
    torso_dist1 = get_eucldn_dist(curr_det(20,1),curr_det(20,2),curr_det(8,1),curr_det(8,2));
    torso_dist2 = get_eucldn_dist(curr_det(21,1),curr_det(21,2),curr_det(9,1),curr_det(9,2));
    torso_dist3 = get_eucldn_dist(curr_det(22,1),curr_det(22,2),curr_det(10,1),curr_det(10,2));
    
    num_fails(1) = max(0,(torso_dist2-torso_dist1)/max(1,torso_dist2));
    num_fails(2) = max(0,(torso_dist3-torso_dist2)/max(1,torso_dist3));
end

function num_fails = check_lrlegpart_sum_dist(curr_det,mean_lrlegpart_sum_dist)
num_fails = 0;
lrlegpart_sum_dist = get_eucldn_dist(curr_det(11,1),curr_det(11,2),curr_det(23,1),curr_det(23,2)) + get_eucldn_dist(curr_det(12,1),curr_det(12,2),curr_det(24,1),curr_det(24,2)) + get_eucldn_dist(curr_det(13,1),curr_det(13,2),curr_det(25,1),curr_det(25,2)) + get_eucldn_dist(curr_det(14,1),curr_det(14,2),curr_det(26,1),curr_det(26,2));
ratio_ = lrlegpart_sum_dist/mean_lrlegpart_sum_dist;
% num_fails = abs(1-ratio_);
if ratio_ > 1.5 || ratio_ < 0.5
    num_fails = num_fails + 1;
end;
end

function num_fails = check_pose_scale(curr_det,mean_pose_scale)
num_fails = 0;
pose_scale = abs(max(curr_det(:,2)) - min(curr_det(:,2)));
ratio_ = pose_scale/mean_pose_scale;
% num_fails = abs(1-ratio_);
if ratio_ > 1.25 || ratio_ < 0.75
    num_fails = num_fails + 1;
end;
end

function num_fails = check_sho_traversal_dist(curr_det,mean_sho_traversal_dist)
num_fails = 0;
sho_traversal_dist = get_eucldn_dist(curr_det(3,1),curr_det(3,2),curr_det(2,1),curr_det(2,2)) + get_eucldn_dist(curr_det(2,1),curr_det(2,2),curr_det(15,1),curr_det(15,2));
ratio_ = sho_traversal_dist/mean_sho_traversal_dist;
% num_fails = abs(1-ratio_);
if ratio_ > 1.4 || ratio_ < 0.6
    num_fails = num_fails + 1;
end;
end

function num_fails = check_lr_dist(curr_det,mean_lrsho_dist,mean_lrhip_dist)
num_fails = [0 0];
lrsho_dist = get_eucldn_dist(curr_det(3,1),curr_det(3,2),curr_det(15,1),curr_det(15,2));
lrhip_dist = get_eucldn_dist(curr_det(10,1),curr_det(10,2),curr_det(22,1),curr_det(22,2));
sho_ratio = lrsho_dist/mean_lrsho_dist; hip_ratio = lrhip_dist/mean_lrhip_dist;
% num_fails = [abs(1-sho_ratio), abs(1-hip_ratio)];
if sho_ratio > 2.0 || sho_ratio < 0.60
    num_fails(1) = 1;
end;
if hip_ratio > 2.0 || hip_ratio < 0.75
    num_fails(2) = 1;
end;
end

function num_fails = check_sho_loc(curr_det,mean_sho_loc1,mean_sho_loc2)
num_fails = [0 0];
sho_dist1 = get_eucldn_dist(mean_sho_loc1(1),mean_sho_loc1(2),curr_det(3,1),curr_det(3,2));
sho_dist2 = get_eucldn_dist(mean_sho_loc2(1),mean_sho_loc2(2),curr_det(15,1),curr_det(15,2));
thresh_dist = 0.25*(max(curr_det(:,2)) - min(curr_det(:,2)));
% num_fails = [sho_dist1/thresh_dist, sho_dist2/thresh_dist];
if sho_dist1 >= thresh_dist
    num_fails(1) = 1;
end;
if sho_dist2 >= thresh_dist
    num_fails(2) = 1; 
end;
end

function num_fails = check_hip_loc(curr_det,mean_hip_loc1,mean_hip_loc2)
num_fails = [0 0];
hip_dist1 = get_eucldn_dist(mean_hip_loc1(1),mean_hip_loc1(2),curr_det(10,1),curr_det(10,2));
hip_dist2 = get_eucldn_dist(mean_hip_loc2(1),mean_hip_loc2(2),curr_det(22,1),curr_det(22,2));
thresh_dist = 0.25*(max(curr_det(:,2)) - min(curr_det(:,2)));
% num_fails = [hip_dist1/thresh_dist, hip_dist2/thresh_dist];
if hip_dist1 >= thresh_dist
    num_fails(1) = 1;
end;
if hip_dist2 >= thresh_dist
    num_fails(2) = 1;
end;
end

function num_fails = check_arms_len(curr_det,mean_arms_len1,mean_arms_len2)
num_fails = [0 0]; 
arms_len1 = get_eucldn_dist(curr_det(3,1),curr_det(3,2),curr_det(4,1),curr_det(4,2)) + get_eucldn_dist(curr_det(4,1),curr_det(4,2),curr_det(5,1),curr_det(5,2)) + get_eucldn_dist(curr_det(5,1),curr_det(5,2),curr_det(6,1),curr_det(6,2)) + get_eucldn_dist(curr_det(6,1),curr_det(6,2),curr_det(7,1),curr_det(7,2));
arms_len2 = get_eucldn_dist(curr_det(15,1),curr_det(15,2),curr_det(16,1),curr_det(16,2)) + get_eucldn_dist(curr_det(16,1),curr_det(16,2),curr_det(17,1),curr_det(17,2)) + get_eucldn_dist(curr_det(17,1),curr_det(17,2),curr_det(18,1),curr_det(18,2)) + get_eucldn_dist(curr_det(18,1),curr_det(18,2),curr_det(19,1),curr_det(19,2));

arms1_ratio = arms_len1/mean_arms_len1; arms2_ratio = arms_len2/mean_arms_len2;
% num_fails = [abs(1-arms1_ratio), abs(1-arms2_ratio)];
if arms1_ratio > 2.0 || arms1_ratio < 0.5
    num_fails(1) = 1;%num_fails + 1;
end;
if arms2_ratio > 2.0 || arms2_ratio < 0.5
    num_fails(2) = 1;%num_fails + 1;
end;
end

function num_fails = check_legs_len(curr_det,mean_legs_len1,mean_legs_len2)
num_fails = [ 0 0]; 
legs_len1 = get_eucldn_dist(curr_det(10,1),curr_det(10,2),curr_det(11,1),curr_det(11,2)) + get_eucldn_dist(curr_det(11,1),curr_det(11,2),curr_det(12,1),curr_det(12,2)) + get_eucldn_dist(curr_det(12,1),curr_det(12,2),curr_det(13,1),curr_det(13,2)) + get_eucldn_dist(curr_det(13,1),curr_det(13,2),curr_det(14,1),curr_det(14,2));
legs_len2 = get_eucldn_dist(curr_det(22,1),curr_det(22,2),curr_det(23,1),curr_det(23,2)) + get_eucldn_dist(curr_det(23,1),curr_det(23,2),curr_det(24,1),curr_det(24,2)) + get_eucldn_dist(curr_det(24,1),curr_det(24,2),curr_det(25,1),curr_det(25,2)) + get_eucldn_dist(curr_det(25,1),curr_det(25,2),curr_det(26,1),curr_det(26,2));

legs1_ratio = legs_len1/mean_legs_len1; legs2_ratio = legs_len2/mean_legs_len2;
% num_fails = [abs(1-legs1_ratio), abs(1-legs2_ratio)];
if legs1_ratio > 1.25 || legs1_ratio < 0.75
    num_fails(1) = 1;
end;
if legs2_ratio > 1.25 || legs2_ratio < 0.75
    num_fails(2) = 1;
end;
end

function num_fails = check_torso_len(curr_det,mean_torso_len1,mean_torso_len2)
num_fails = [0 0]; 

torso_len1 = get_eucldn_dist(curr_det(3,1),curr_det(3,2),curr_det(8,1),curr_det(8,2)) + get_eucldn_dist(curr_det(8,1),curr_det(8,2),curr_det(9,1),curr_det(9,2)) + get_eucldn_dist(curr_det(9,1),curr_det(9,2),curr_det(10,1),curr_det(10,2));
torso_len2 = get_eucldn_dist(curr_det(15,1),curr_det(15,2),curr_det(20,1),curr_det(20,2)) + get_eucldn_dist(curr_det(20,1),curr_det(20,2),curr_det(21,1),curr_det(21,2)) + get_eucldn_dist(curr_det(21,1),curr_det(21,2),curr_det(22,1),curr_det(22,2));

torso1_ratio = torso_len1/mean_torso_len1; torso2_ratio = torso_len2/mean_torso_len2;
% num_fails = [abs(1-torso1_ratio), abs(1-torso2_ratio)];
if torso1_ratio > 1.25 || torso1_ratio < 0.75
    num_fails(1) = 1;
end;
if torso2_ratio > 1.25 || torso2_ratio < 0.75
    num_fails(2) = 1;
end;
end

function num_intersection = get_leg_parts_intersection(curr_det)
num_intersection = 0; 
if curr_det(11,1) >= curr_det(23,1)
    num_intersection = num_intersection + 1;
end;
if curr_det(12,1) >= curr_det(24,1)
    num_intersection = num_intersection + 1;
end;
if curr_det(13,1) >= curr_det(25,1)
    num_intersection = num_intersection + 1;
end;
if curr_det(14,1) >= curr_det(26,1)
    num_intersection = num_intersection + 1;
end;
end

function unequal_flag = check_unequal_torso_lines(curr_det)
% unequal_flag = 0; 
torso_len1 = get_eucldn_dist(curr_det(3,1),curr_det(3,2),curr_det(8,1),curr_det(8,2)) + get_eucldn_dist(curr_det(8,1),curr_det(8,2),curr_det(9,1),curr_det(9,2)) + get_eucldn_dist(curr_det(9,1),curr_det(9,2),curr_det(10,1),curr_det(10,2));
torso_len2 = get_eucldn_dist(curr_det(15,1),curr_det(15,2),curr_det(20,1),curr_det(20,2)) + get_eucldn_dist(curr_det(20,1),curr_det(20,2),curr_det(21,1),curr_det(21,2)) + get_eucldn_dist(curr_det(21,1),curr_det(21,2),curr_det(22,1),curr_det(22,2));
torso_ratio = torso_len1/torso_len2;
if torso_ratio < 0.75 || torso_ratio > 1.25
    unequal_flag = 1;
else
    unequal_flag = 0;
end;
% unequal_flag = abs(1-(torso_len1/torso_len2));
end

function intersection_flag = check_hip_swap(curr_det)
intersection_flag = 0; 
if curr_det(10,1) >= curr_det(22,1)
    intersection_flag = 1;
end;
end

function intersection_flag = check_sho_swap(curr_det)
intersection_flag = 0; 
if curr_det(3,1) >= curr_det(15,1)
    intersection_flag = 1;
end;
end

function intersection_flag = get_torso_intersection(curr_det)
intersection_flag = 0;
if any(curr_det([3,8,9,10],1) >= curr_det([15,20,21,22],1))
    intersection_flag = 1;
end;
end

function min_dist_lrhip_flag = get_min_dist_lrhip_flag(indxs,all_dets)
min_dist_lrhip_flag = 0;
for i=indxs
    curr_det = all_dets{i};
    temp_dist = get_eucldn_dist(curr_det(10,1),curr_det(10,2),curr_det(22,1),curr_det(22,2));
    if temp_dist > 10
        min_dist_lrhip_flag = 1;
    end;
end;
end

function lrlegpart_sum_dist = get_mean_lrlegpart_sum_dist(indxs,all_dets)
lrlegpart_sum_dist = deal([]);

for i=indxs
    curr_det = all_dets{i};
    lrlegpart_sum_dist = [lrlegpart_sum_dist, get_eucldn_dist(curr_det(11,1),curr_det(11,2),curr_det(23,1),curr_det(23,2)) + get_eucldn_dist(curr_det(12,1),curr_det(12,2),curr_det(24,1),curr_det(24,2)) + get_eucldn_dist(curr_det(13,1),curr_det(13,2),curr_det(25,1),curr_det(25,2)) + get_eucldn_dist(curr_det(14,1),curr_det(14,2),curr_det(26,1),curr_det(26,2))];
end;
lrlegpart_sum_dist = mean(lrlegpart_sum_dist);
end

function pose_scale = get_mean_pose_scale(indxs,all_dets)
pose_scale = [];

for i=indxs
    curr_det = all_dets{i};
    pose_scale = [pose_scale ; abs(max(curr_det(:,2)) - min(curr_det(:,2)))];
end;
pose_scale = mean(pose_scale);
end

function sho_traversal_dist = get_mean_sho_traversal_dist(indxs,all_dets)
sho_traversal_dist = [];

for i=indxs
    curr_det = all_dets{i};
    sho_traversal_dist = [sho_traversal_dist ; get_eucldn_dist(curr_det(3,1),curr_det(3,2),curr_det(2,1),curr_det(2,2)) + get_eucldn_dist(curr_det(2,1),curr_det(2,2),curr_det(15,1),curr_det(15,2))];
end;
sho_traversal_dist = mean(sho_traversal_dist);
end

function [lrsho_dist, lrhip_dist] = get_mean_lr_dist(indxs,all_dets)
[lrsho_dist, lrhip_dist] = deal([]);

for i=indxs
    curr_det = all_dets{i};
    lrsho_dist = [lrsho_dist; get_eucldn_dist(curr_det(3,1),curr_det(3,2),curr_det(15,1),curr_det(15,2))];
    lrhip_dist = [lrhip_dist; get_eucldn_dist(curr_det(10,1),curr_det(10,2),curr_det(22,1),curr_det(22,2))];
end;
lrsho_dist = mean(lrsho_dist); lrhip_dist = mean(lrhip_dist);
end

function [sho_loc1, sho_loc2] = get_mean_sho_loc(indxs,all_dets)
[sho_loc1, sho_loc2] = deal([]);

for i=indxs
    curr_det = all_dets{i};
    sho_loc1 = [sho_loc1; [curr_det(3,1),curr_det(3,2)]];
    sho_loc2 = [sho_loc2; [curr_det(15,1),curr_det(15,2)]];
end;
sho_loc1 = mean(sho_loc1); sho_loc2 = mean(sho_loc2);
end

function [hip_loc1, hip_loc2] = get_mean_hip_loc(indxs,all_dets)
[hip_loc1, hip_loc2] = deal([]);

for i=indxs
    curr_det = all_dets{i};
    hip_loc1 = [hip_loc1; [curr_det(10,1),curr_det(10,2)]];
    hip_loc2 = [hip_loc2; [curr_det(22,1),curr_det(22,2)]];
end;
hip_loc1 = mean(hip_loc1); hip_loc2 = mean(hip_loc2);
end

function [arms_len1, arms_len2] = get_mean_arms_len(indxs,all_dets)
[arms_len1, arms_len2] = deal([]);

for i=indxs
    curr_det = all_dets{i};
    arms_len1 = [arms_len1, get_eucldn_dist(curr_det(3,1),curr_det(3,2),curr_det(4,1),curr_det(4,2)) + get_eucldn_dist(curr_det(4,1),curr_det(4,2),curr_det(5,1),curr_det(5,2)) + get_eucldn_dist(curr_det(5,1),curr_det(5,2),curr_det(6,1),curr_det(6,2)) + get_eucldn_dist(curr_det(6,1),curr_det(6,2),curr_det(7,1),curr_det(7,2))];
    arms_len2 = [arms_len2, get_eucldn_dist(curr_det(15,1),curr_det(15,2),curr_det(16,1),curr_det(16,2)) + get_eucldn_dist(curr_det(16,1),curr_det(16,2),curr_det(17,1),curr_det(17,2)) + get_eucldn_dist(curr_det(17,1),curr_det(17,2),curr_det(18,1),curr_det(18,2)) + get_eucldn_dist(curr_det(18,1),curr_det(18,2),curr_det(19,1),curr_det(19,2))];
end;
arms_len1 = mean(arms_len1); arms_len2 = mean(arms_len2);
end

function [legs_len1, legs_len2] = get_mean_legs_len(indxs,all_dets)
[legs_len1, legs_len2] = deal([]);

for i=indxs
    curr_det = all_dets{i};
    legs_len1 = [legs_len1, get_eucldn_dist(curr_det(10,1),curr_det(10,2),curr_det(11,1),curr_det(11,2)) + get_eucldn_dist(curr_det(11,1),curr_det(11,2),curr_det(12,1),curr_det(12,2)) + get_eucldn_dist(curr_det(12,1),curr_det(12,2),curr_det(13,1),curr_det(13,2)) + get_eucldn_dist(curr_det(13,1),curr_det(13,2),curr_det(14,1),curr_det(14,2))];
    legs_len2 = [legs_len2, get_eucldn_dist(curr_det(22,1),curr_det(22,2),curr_det(23,1),curr_det(23,2)) + get_eucldn_dist(curr_det(23,1),curr_det(23,2),curr_det(24,1),curr_det(24,2)) + get_eucldn_dist(curr_det(24,1),curr_det(24,2),curr_det(25,1),curr_det(25,2)) + get_eucldn_dist(curr_det(25,1),curr_det(25,2),curr_det(26,1),curr_det(26,2))];
end;
legs_len1 = mean(legs_len1); legs_len2 = mean(legs_len2);
end

function [torso_len1, torso_len2] = get_mean_torso_len(indxs,all_dets)
[torso_len1,torso_len2] = deal([]);

for i=indxs
    curr_det = all_dets{i};
    torso_len1 = [torso_len1, get_eucldn_dist(curr_det(3,1),curr_det(3,2),curr_det(8,1),curr_det(8,2)) + get_eucldn_dist(curr_det(8,1),curr_det(8,2),curr_det(9,1),curr_det(9,2)) + get_eucldn_dist(curr_det(9,1),curr_det(9,2),curr_det(10,1),curr_det(10,2))];
    torso_len2 = [torso_len2, get_eucldn_dist(curr_det(15,1),curr_det(15,2),curr_det(20,1),curr_det(20,2)) + get_eucldn_dist(curr_det(20,1),curr_det(20,2),curr_det(21,1),curr_det(21,2)) + get_eucldn_dist(curr_det(21,1),curr_det(21,2),curr_det(22,1),curr_det(22,2))];
end;
torso_len1 = mean(torso_len1); torso_len2 = mean(torso_len2);
end

function dist_ = get_eucldn_dist(x1,y1,x2,y2)
dist_ = sqrt(power(x1-x2,2) + power(y1-y2,2));
end

function [det_26] = get_det_from_boxes(boxes)
best_box = boxes(1,1:104);
det_26 = [];
for j=1:4:length(best_box)
    det_26 = [det_26;[mean([best_box(j),best_box(j+2)]),mean([best_box(j+1),best_box(j+3)])]];
end;
end