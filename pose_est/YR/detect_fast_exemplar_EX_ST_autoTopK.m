function boxes = detect_fast_exemplar_EX_ST_autoTopK(im,model,thresh,filters_exemplar,exemplar_constant,fr_indx,calib_flag,resp_outdir,exemplar_indxs)
% boxes = detect_fast_exemplar(im, model, min(model.thresh,-1), weights_exemplar,0.5);
% calib_flag == 1 means get responses without calibration and get min-max

% Compute the feature pyramid and prepare filter
pyra     = featpyramid(im,model);
interval = model.interval;
levels   = 1:length(pyra.feat);
min_exemplar_score = -0.3378; max_exemplar_score = 0.5996;

% L-2 normalization of exemplar filters.
for i=1:numel(filters_exemplar)
    curr_cell = filters_exemplar{i};
    for j=1:numel(curr_cell)
        l2_norm_val = norm(reshape(curr_cell{j},[1, size(curr_cell{j},1)*size(curr_cell{j},2)*size(curr_cell{j},3)]));
        curr_cell{j} = curr_cell{j}/l2_norm_val;
    end;
    filters_exemplar{i} = curr_cell;
end;

% L-2 normalisation of image pyramid
pyra_l2_normed = pyra;
for i=1:numel(pyra_l2_normed.feat)
    curr_cell = pyra_l2_normed.feat{i};
    l2_norm_val = norm(reshape(curr_cell,[1,size(curr_cell,1)*size(curr_cell,2)*size(curr_cell,3)]));
    curr_cell = curr_cell/l2_norm_val;
    pyra_l2_normed.feat{i} = curr_cell;
end;

% change filters_exemplar from 26*H*W to 156*H*W for code compatibility
filters_exemplar_156 = {};
for i=1:numel(filters_exemplar)
    curr_cell = filters_exemplar{i};
    new_cell = {};
    for j=1:numel(curr_cell)
        for k=1:6
            new_cell{(j-1)*6+k} = curr_cell{j};
        end;
    end;
    filters_exemplar_156{i} = new_cell;
end;

% Cache various statistics derived from model
[components,filters,resp] = modelcomponents(model,pyra);
boxes = zeros(10000,length(components{1})*4+2);
cnt   = 0;
if calib_flag == 1
    resp_exemplar = resp;
else
    % load resp_exemplar
    load(fullfile(resp_outdir,[num2str(fr_indx) '_resp.mat']));
    load(fullfile(resp_outdir,'calib.mat'));
end;

% Iterate over scales and components,
for rlevel = levels,
    for c  = 1:length(model.components),
        
        if calib_flag == 1
            % pick the max W_ex for current scale level
            
            if ~isempty(find(exemplar_indxs == fr_indx)) % if curr indx is one of exemplars, no need to find max
                max_exemplar_indx = find(exemplar_indxs == fr_indx);
            else
                max_exemplar_indx = 0; max_exemplar_val = 0;
                for i=1:numel(filters_exemplar)
                    curr_exemplar_response = fconv(pyra.feat{rlevel},filters_exemplar{i},1,length(filters_exemplar{i}));
                    max_val = [];
                    for j=1:numel(curr_exemplar_response)
                        max_val = [max_val, max(max(curr_exemplar_response{j}))];
                    end;
                    
                    if sum(max_val) > max_exemplar_val
                        max_exemplar_indx = i;
                        max_exemplar_val = sum(max_val);
                    end;
                end;
            end;
        end;
        
        parts    = components{c};
        numparts = length(parts);
        
        % Local scores
        for k = 1:numparts,
            f     = parts(k).filterid;
            level = rlevel-parts(k).scale*interval;
            if isempty(resp{level}),
                % Y-R unary response
                resp{level} = fconv(pyra.feat{level},filters,1,length(filters));
                
                if calib_flag == 1
                    resp_exemplar{level} = fconv(pyra.feat{level},filters_exemplar_156{max_exemplar_indx},1,length(filters_exemplar_156{max_exemplar_indx}));
                end;
                
                % PLAIN normalization (using min-max)
                % On observing, resp_exemplar score lies in the range [0,5.7773];
                % So normalizing by 6
                if calib_flag ~= 1
                    for ki=1:numel(resp_exemplar{level})
                        resp_exemplar{level}{ki} = (resp_exemplar{level}{ki}./(max_-min_)) + (abs(min_)/(max_-min_)); 
                    end;
                end;
                
                % THE EQUATION
                temp_resp = {};
                for ki = 1:numel(resp{level})
                    temp_resp{ki} = (resp_exemplar{level}{ki}*(max_exemplar_score-(min_exemplar_score)) + (min_exemplar_score));
                    
                    temp_resp{ki} = (1-exemplar_constant)*resp{level}{ki} + exemplar_constant*temp_resp{ki};
                end;
                resp{level} = temp_resp;
            end
            
            for fi = 1:length(f)
                parts(k).score(:,:,fi) = resp{level}{f(fi)};
            end
            parts(k).level = level;
        end
        
        % Walk from leaves to root of tree, passing message to parent
        for k = numparts:-1:2,
            par = parts(k).parent;
            [msg,parts(k).Ix,parts(k).Iy,parts(k).Ik] = passmsg(parts(k),parts(par));
            parts(par).score = parts(par).score + msg;
        end
        
        % Add bias to root score
        parts(1).score = parts(1).score + parts(1).b;
        [rscore Ik]    = max(parts(1).score,[],3);
        
        % Walk back down tree following pointers
        [Y,X] = find(rscore >= thresh);
        if length(X) > 1,
            I   = (X-1)*size(rscore,1) + Y;
            box = backtrack(X,Y,Ik(I),parts,pyra);
            i   = cnt+1:cnt+length(I);
            boxes(i,:) = [box repmat(c,length(I),1) rscore(I)];
            cnt = i(end);
        else
            [Y,X] = find(rscore == max(max(rscore)));
            I   = (X-1)*size(rscore,1) + Y;
            box = backtrack(X,Y,Ik(I),parts,pyra);
            i   = cnt+1:cnt+length(I);
            boxes(i,:) = [box repmat(c,length(I),1) rscore(I)];
            cnt = i(end);
        end
    end
end


% save resp_exemplar
if calib_flag == 1
    save(fullfile(resp_outdir,[num2str(fr_indx) '_resp.mat']),'resp_exemplar');
end;

boxes = boxes(1:cnt,:);

% Cache various statistics from the model data structure for later use
function [components,filters,resp] = modelcomponents(model,pyra)
components = cell(length(model.components),1);
for c = 1:length(model.components),
    for k = 1:length(model.components{c}),
        p = model.components{c}(k);
        [p.w,p.defI,p.starty,p.startx,p.step,p.level,p.Ix,p.Iy] = deal([]);
        [p.scale,p.level,p.Ix,p.Iy] = deal(0);
        
        % store the scale of each part relative to the component root
        par = p.parent;
        assert(par < k);
        p.b = [model.bias(p.biasid).w];
        p.b = reshape(p.b,[1 size(p.biasid)]);
        p.biasI = [model.bias(p.biasid).i];
        p.biasI = reshape(p.biasI,size(p.biasid));
        p.sizx  = zeros(length(p.filterid),1);
        p.sizy  = zeros(length(p.filterid),1);
        
        for f = 1:length(p.filterid)
            x = model.filters(p.filterid(f));
            [p.sizy(f) p.sizx(f) foo] = size(x.w);
            %         p.filterI(f) = x.i;
        end
        for f = 1:length(p.defid)
            x = model.defs(p.defid(f));
            p.w(:,f)  = x.w';
            p.defI(f) = x.i;
            ax  = x.anchor(1);
            ay  = x.anchor(2);
            ds  = x.anchor(3);
            p.scale = ds + components{c}(par).scale;
            % amount of (virtual) padding to hallucinate
            step     = 2^ds;
            virtpady = (step-1)*pyra.pady;
            virtpadx = (step-1)*pyra.padx;
            % starting points (simulates additional padding at finer scales)
            p.starty(f) = ay-virtpady;
            p.startx(f) = ax-virtpadx;
            p.step   = step;
        end
        components{c}(k) = p;
    end
end

resp    = cell(length(pyra.feat),1);
filters = cell(length(model.filters),1);
for i = 1:length(filters),
    filters{i} = model.filters(i).w;
end

% Given a 2D array of filter scores 'child',
% (1) Apply distance transform
% (2) Shift by anchor position of part wrt parent
% (3) Downsample if necessary
function [score,Ix,Iy,Ik] = passmsg(child,parent)
INF = 1e10;
K   = length(child.filterid);
Ny  = size(parent.score,1);
Nx  = size(parent.score,2);
[Ix0,Iy0,score0] = deal(zeros([Ny Nx K]));

for k = 1:K
    [score0(:,:,k),Ix0(:,:,k),Iy0(:,:,k)] = shiftdt(child.score(:,:,k), child.w(1,k), child.w(2,k), child.w(3,k), child.w(4,k),child.startx(k),child.starty(k),Nx,Ny,child.step);
end

% At each parent location, for each parent mixture 1:L, compute best child mixture 1:K
L  = length(parent.filterid);
N  = Nx*Ny;
i0 = reshape(1:N,Ny,Nx);
[score,Ix,Iy,Ix,Ik] = deal(zeros(Ny,Nx,L));
for l = 1:L
    b = child.b(1,l,:);
    [score(:,:,l),I] = max(bsxfun(@plus,score0,b),[],3);
    i = i0 + N*(I-1);
    Ix(:,:,l)    = Ix0(i);
    Iy(:,:,l)    = Iy0(i);
    Ik(:,:,l)    = I;
end

% Backtrack through DP msgs to collect ptrs to part locations
function box = backtrack(x,y,mix,parts,pyra)
numx     = length(x);
numparts = length(parts);

xptr = zeros(numx,numparts);
yptr = zeros(numx,numparts);
mptr = zeros(numx,numparts);
box  = zeros(numx,4,numparts);

for k = 1:numparts,
    p   = parts(k);
    if k == 1,
        xptr(:,k) = x;
        yptr(:,k) = y;
        mptr(:,k) = mix;
    else
        % I = sub2ind(size(p.Ix),yptr(:,par),xptr(:,par),mptr(:,par));
        par = p.parent;
        [h,w,foo] = size(p.Ix);
        I   = (mptr(:,par)-1)*h*w + (xptr(:,par)-1)*h + yptr(:,par);
        xptr(:,k) = p.Ix(I);
        yptr(:,k) = p.Iy(I);
        mptr(:,k) = p.Ik(I);
    end
    scale = pyra.scale(p.level);
    x1 = (xptr(:,k) - 1 - pyra.padx)*scale+1;
    y1 = (yptr(:,k) - 1 - pyra.pady)*scale+1;
    x2 = x1 + p.sizx(mptr(:,k))*scale - 1;
    y2 = y1 + p.sizy(mptr(:,k))*scale - 1;
    box(:,:,k) = [x1 y1 x2 y2];
end
box = reshape(box,numx,4*numparts);
