% rough incomplete code to do platt calibration

addpath('/Pulsar3/digvijay.singh/code/liblinear-1.96/matlab');
[label_vector, instance_matrix] = libsvmread('/Pulsar1/users/digvijay.singh/active_learning/dramanan_pose/pose_iterative/20121128-pose-release-ver1.3/code-basic/posneg_data'); 
model = train(label_vector, instance_matrix, '-s 0');
% [predict_label, accuracy, dec_values] = predict(heart_scale_label, heart_scale_inst, model); % test the training data

all_pos = [];
all_ = [];
for i=1:size(all_resp,2)    %20
    for j=1:26              %20*26
        max_ = -1000;
        for k=1:numel(all_resp{i})  % 30 levels
            for l=(j-1)*6+1:(j-1)*6+6   % 6 mixtures for each 'j' or component
%                 all_ = [all_,unique(all_resp{i}{k}{l})'];
                if max(max(all_resp{i}{k}{l})) > max_
                    max_ = max(max(all_resp{i}{k}{l}));
                end;
            end;
        end;
        all_pos = [all_pos,max_];
    end;
end;

load platt_data.mat;

fp = fopen('posneg_data','w+');
for i=1:numel(pos)
    fprintf(fp,'+1 1:%f\n',pos(i));
end;
for i=1:numel(neg)
    fprintf(fp,'-1 1:%f\n',neg(i));
end;


fclose(fp);


