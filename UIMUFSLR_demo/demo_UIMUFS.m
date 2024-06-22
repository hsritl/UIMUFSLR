clear;
clc;
addpath(genpath(pwd));

dataName='BBCSport';
load (dataName)
instance_num=size(labels,2);
maxiter = 50;
neighbor_num=5;
% optimal parameters for the BBCSport dataset with a 20% missing rate
alpha = 10;
beta=1;
tau=1;
% The construction of two indicator matrices for missing sample
for per_idx=1:4
    XF{per_idx}=[];
    for view_idx=1:view_num
        XF{per_idx}=[XF{per_idx};X{per_idx,view_idx}];
        M{per_idx,view_idx} = eye(instance_num);
        M{per_idx,view_idx}(zero_indices{per_idx,view_idx},:) = [];
        W{per_idx,view_idx} = eye(instance_num);
        W{per_idx,view_idx}(one_indices{per_idx,view_idx},:) = [];
    end
end
% UFS stage
ri=1;
for per=1:4
    for view_idx=1:view_num
        XFX=XF{per};
        MX{view_idx}=M{per,view_idx};
        X_missingX{view_idx}=X_missing{per,view_idx};
        zero_indicesX{view_idx}=zero_indices{per,view_idx};
    end
    tic;
    [rank,C,Y,ttt,A] = UIMUFS5S2(X_missingX,XFX,MX,zero_indicesX,class_num,maxiter,neighbor_num,alpha,beta,tau);
    disp(num2str(toc));
    ranking{ri}=rank;
    ri=ri+1;
    clear XFX MX X_missingX zero_indicesX
end
% Clustering stage
for per_idx=1:4
    XFT{per_idx}=XF{per_idx}';
end
dim_num = size(XF{1},1); %The number of total features.
ri=1;
for per=1:4
    tic;
    for t1=1:9
        prop = (t1+1)*0.05; %The proportion of selected features.
        Xsub = XFT{per}(: , ranking{ri}(1 : floor(prop*dim_num)));
        [res] = litekmeans(Xsub, class_num, 'Replicates',10);
        clear Xsub
        R= EvaluationMetrics(labels', res);
        acc{ri}(t1)=R(1);
        nmi{ri}(t1)=R(2);
        pu{ri}(t1)=R(3);
        fs{ri}(t1)=R(4);
        pre{ri}(t1)=R(5);
    end
    disp(num2str(toc));
    ri=ri+1;
end
for per=1:4
 accx(per)=max(acc{per});
 nmix(per)=max(nmi{per});
 pux(per)=max(pu{per});
 fsx(per)=max(fs{per});
 prex(per)=max(pre{per});
end

fprintf('result with the optimal feature selection percentage while missing ratio 0.1: ACC%f, NMI%f, Purity%f, Fscore%f, Precision%f\n', 100*accx(1), 100*nmix(1), 100*pux(1), 100*fsx(1), 100*prex(1));
fprintf('result with the optimal feature selection percentage while missing ratio 0.2: ACC%f, NMI%f, Purity%f, Fscore%f, Precision%f\n', 100*accx(2), 100*nmix(2), 100*pux(2), 100*fsx(2), 100*prex(2));
fprintf('result with the optimal feature selection percentage while missing ratio 0.3: ACC%f, NMI%f, Purity%f, Fscore%f, Precision%f\n', 100*accx(3), 100*nmix(3), 100*pux(3), 100*fsx(3), 100*prex(3));
fprintf('result with the optimal feature selection percentage while missing ratio 0.4: ACC%f, NMI%f, Purity%f, Fscore%f, Precision%f\n', 100*accx(4), 100*nmix(4), 100*pux(4), 100*fsx(4), 100*prex(4));
