function [out_pck_train, score_train] = evaluate_dir_half(cache_dir,vid_dir,det_dir)

out_mat = fullfile(cache_dir, vid_dir, 'ex_cnst_final_svm_r1.mat');
load(out_mat);
gt_dir = fullfile('dataset',vid_dir,'gt');
all_imgs = dir(fullfile('dataset',vid_dir,'*.png'));

for i=1:numel(all_imgs)
    % load detection
    load(fullfile(det_dir,[all_imgs(i).name(1:end-4) '_pose.mat']));
    best_box = boxes(1:72);
    det_18 = [];
    for j=1:4:length(best_box)
        det_18 = [det_18;[mean([best_box(j),best_box(j+2)]),mean([best_box(j+1),best_box(j+3)])]];
    end;
    det_all{i} = det_18;
    
    %load gt
    load(fullfile(gt_dir,[all_imgs(i).name(1:end-4) '_pose.mat']));
    gt_all{i} = arr_18;
end;

%get PCK
[out_pck_train, score_train] = my_eval_pck(det_all,gt_all,0.2);

    

end