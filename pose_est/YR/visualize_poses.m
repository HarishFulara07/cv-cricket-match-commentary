function visualize_poses(dir_path,model_mat,results_mat)
% visualize_poses('/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases/phase7/K_6','PARSE_phase7_K6_final_66666666666666666666666666.mat','test_results.mat');
% takes gt_poses from dir_path/gt_pose and prints visualizations in
% dir_path/visualize_gt_pose

addpath visualization;
if isunix()
  addpath mex_unix;
elseif ispc()
  addpath mex_pc;
end
compile

out_dir = fullfile(dir_path,'visualize_PARSE_pose');
if ~exist(out_dir)
    system(['mkdir ' out_dir]);
end;

min_indxs = [];
load('/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases/phase1/K_6/test_results.mat');
min_indxs = minscore_indxs(1:20);
load('/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases/phase2/K_6/test_results.mat');
minscore_indxs = setdiff(minscore_indxs,min_indxs,'stable');
min_indxs = [min_indxs,minscore_indxs(1:20)];
load('/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases/phase3/K_6/test_results.mat');
minscore_indxs = setdiff(minscore_indxs,min_indxs,'stable');
min_indxs = [min_indxs,minscore_indxs(1:20)];
load('/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases/phase4/K_6/test_results.mat');
minscore_indxs = setdiff(minscore_indxs,min_indxs,'stable');
min_indxs = [min_indxs,minscore_indxs(1:20)];
load('/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases/phase5/K_6/test_results.mat');
minscore_indxs = setdiff(minscore_indxs,min_indxs,'stable');
min_indxs = [min_indxs,minscore_indxs(1:20)];
load('/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/phases/phase6/K_6/test_results.mat');
minscore_indxs = setdiff(minscore_indxs,min_indxs,'stable');
min_indxs = [min_indxs,minscore_indxs(1:20)];

load(fullfile(dir_path,model_mat));
load(fullfile(dir_path,results_mat));
minscore_indxs = setdiff(minscore_indxs,min_indxs,'stable');
min_indxs = [min_indxs,minscore_indxs(1:20)];

ims = '/Pulsar1/users/digvijay.singh/cricket_dataset/train_dramanan/scaled/image%03d.png';
colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
for fr = min_indxs
    im_name = sprintf(ims,fr);
    im = imread(im_name);
    [~,name,~] = fileparts(im_name);
    load(fullfile(dir_path,['train_detections/' name '_pose.mat']),'boxes');
    showskeletons(im, boxes,colorset,pa); % show the best detection
    
    %find eval PCK score
    best_box = boxes(1:104);
    det_26 = [];
    for j=1:4:length(best_box)
        det_26 = [det_26;[mean([best_box(j),best_box(j+2)]),mean([best_box(j+1),best_box(j+3)])]];
    end;
    load(fullfile('/Pulsar1/users/digvijay.singh/cricket_dataset/train_dramanan/scaled',[name '_pose.mat']));
    [out_pck_train, score_train] = my_eval_pck({det_26},{arr_26});
    
%     print([fullfile(out_dir,imlist(i).name(1:end-4)) '.jpg'],'-djpeg');
    text(1,1,num2str(out_pck_train),'BackgroundColor',[1,1,1],'FontSize',14)
    print(fullfile(out_dir,[name '.jpg']),'-djpeg');
%     keyboard
    close all
end



% all_imgs = dir(fullfile(dir_path,'*png'));
% fprintf('%d: ',numel(all_imgs));
% for i=1:numel(all_imgs)
%     fprintf('%d ',i);
%     im = imread(fullfile(dir_path,all_imgs(i).name));
%     box_size = (size(im,1) + size(im,2))/400;
%     load(fullfile(dir_path,['gt_pose/' all_imgs(i).name(1:end-4) '_pose.mat']));    %gives gt_pose
%     
% %     keyboard;
%     
%     arr_26 = uint32(arr_26);
%     for j=1:size(arr_26,1)
%         curr_x = arr_26(j,2); curr_y = arr_26(j,1);
%         for ix = curr_x-box_size:curr_x+box_size
%             for iy = curr_y-box_size:curr_y+box_size
%                 im(ix,iy,:) = uint8([255,0,0]);
%             end;
%         end;
% %         im(curr_x-box_size:curr_x+box_size,curr_y-box_size:curr_y+box_size,:) = uint8([255,0,0]);
%     end;
%     
%     arr_14 = uint32(arr_14);
%     for j=1:size(arr_14,1)
%         curr_x = arr_14(j,2); curr_y = arr_14(j,1);
%         for ix = curr_x-box_size:curr_x+box_size
%             for iy = curr_y-box_size:curr_y+box_size
%                 im(ix,iy,:) = uint8([0,0,255]);
%             end;
%         end;
% %         im(curr_x-box_size:curr_x+box_size,curr_y-box_size:curr_y+box_size,:) = uint8([0,0,255]);
%     end;
% %     keyboard;
%     imwrite(im,fullfile(out_dir,all_imgs(i).name));
% end;


addpath visualization;

%for 26-parts
colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];

% for 18 parts
colorset = {'g','g','y','m','m','m','m','y','y','y','y','r','r','r','r','c','c','c','c','y','y','y','b','b','b','b'};
colorset = colorset(1:18); 
pa = [0 1 2 3 4 5 6 3 8 9 2 11 12 13 14 11 16 17];

img_dir = '/Pulsar1/users/digvijay.singh/datasets/poses_in_the_wild_public/dataset/selected_seqs/seq15';
mat_dir = '/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-wrapper/mat_dir/video_det/seq15/EX_ST/ph1/detections_autoTopScore/0/';
out_dir = fullfile(mat_dir,'vis_out');
system(['mkdir ' out_dir]);

all_ims = dir(fullfile(img_dir,'*png'));
% all_mats = dir(fullfile(mat_dir,'*mat'));

for i = 1:numel(all_ims)
    fprintf('%d',i);
    im = imread(fullfile(img_dir,all_ims(i).name));
    clf; imagesc(im); axis image; axis off; drawnow; hold on;
    
    load(fullfile(mat_dir,[all_ims(i).name(1:end-4) '_pose.mat']));
    
%     best_box = boxes(1,1:72); 
    best_box = boxes(1,1:104);
    det_26 = [];
    for j=1:4:length(best_box)
        det_26 = [det_26;[mean([best_box(j),best_box(j+2)]),mean([best_box(j+1),best_box(j+3)])]];
    end;
%     plot(det_26(2,1),det_26(2,2),'o'); plot(mean(det_26([3,15],1)),mean(det_26([3,15],2)),'rx');
%     plot(mean(det_26([3,8,9,10,15,20,21,22],1)),mean(det_26([3,8,9,10,15,20,21,22],2)),'g*');
%     plot(mean(det_26([3,10,15,22],1)),mean(det_26([3,10,15,22],2)),'k+');
%     plot(mean(det_26([10,22],1)),mean(det_26([10,22],2)),'bd');
%     plot(mean(det_26([2,3,15],1)),mean(det_26([2,3,15],2)),'kv');
    
    showskeletons(im, boxes, colorset, pa);
    print(gcf,fullfile(out_dir,[all_ims(i).name(1:end-4) '.jpg']),'-djpeg');
    close all;
%     keyboard;
    
    clear boxes;
end;


end