function [pck,tps] = my_eval_pck_half(det, gt, thresh)
% det is a cell of size "num_samples" with each cell element having "num_keypoints * 2" vector of pose detection
% gt is a cell of size "num_samples" with each cell element having "num_keypoints * 2" vector of ground-truth pose

if nargin < 3
  thresh = 0.2;
end

assert(numel(det) == numel(gt));
scale = [];
% Compute the scale of the ground truths
for n = 1:numel(gt)
    curr_gt = gt{n};
    scale = [scale,max(max(curr_gt(:,1))-min(curr_gt(:,1)), max(curr_gt(:,2))-min(curr_gt(:,2)))];
end

tps = [];
for n = 1:numel(gt)
    curr_det = det{n}; curr_gt = gt{n};
    dist = sqrt(sum((curr_gt - curr_det).^2,2));    %this will give me array of length "num_keypoints"
    tps = [tps, numel(find((dist <= thresh * scale(n)) == 1))];
end

% tps

pck = mean(tps);