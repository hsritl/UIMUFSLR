function [ranking,C,Y,ttt,A] = UIMUFS5S2(X_missing,XF,M,zero_indices,class_num,maxiter,neighbor_num,alpha,beta,tau)
gamma=10000;
view_num=size(X_missing,2);
X=cell(1,view_num);
for view_idx=1:view_num
    instance_num(view_idx)=size(X_missing{view_idx},2);
    dim_num(view_idx)=size(X_missing{view_idx},1);
    X{view_idx}=X_missing{view_idx}*M{view_idx};
end
sample_num=size(X{1},2);
S=cell(1,view_num);
for i=1:view_num
    S{i}=Updata_Sv(X_missing{i},class_num,neighbor_num,1);
end
SC = zeros(sample_num,sample_num);
B=cell(1,view_num);
C=cell(1,view_num);
Ci2=cell(1,view_num);
d=cell(1,view_num);
D=cell(1,view_num);
for i =1:view_num
    S{i}=M{i}'*S{i}*M{i};
    SC = SC + S{i};
    bb = ones(1,sample_num);
    bb(zero_indices{i}) = instance_num(i)/sample_num; 
    B{i}=diag(bb);
    C{i} = ones(dim_num(i),class_num);
    D{i} = eye(dim_num(i));
end
% Initialize cluster indicator matrix Y
options.method = 'k_means';
YT = init_H(XF',class_num,options);
Y=YT';

for iter = 1:maxiter
    % update A,SC
    SBA=zeros(sample_num,sample_num);
    BA=zeros(sample_num,sample_num);
    A=cell(1,view_num);
    for i = 1:view_num
        tem=SC-S{i};
        A{i}=diag(0.5./sqrt(sum(tem.*tem,1)+eps));
        SBA=SBA+S{i}*B{i}*A{i};
        BA=BA+B{i}*A{i};
    end
    YY = L2_distance_1(Y,Y);
    SC=SC.*(pos(2*SBA)+neg(2*SC*BA+tau*YY))./max(pos(2*SC*BA+tau*YY)+neg(2*SBA),eps) ;
    SC=SC*diag(1./(sum(SC,1)+eps));
    % update L
    tem = diag( sum( (SC + SC.')/2 ) );
    L = tem - (SC + SC')/2 ;
    % update C
    for i = 1:view_num
        C_u=X{i}*Y'+beta*X{i}*X{i}'*C{i};
        C_d=X{i}*X{i}'*C{i}+alpha*D{i}*C{i}+beta*(X{i}*X{i}')*C{i}*ones(class_num,1)*ones(1,class_num);
        C{i}=C{i}.*(pos(C_u)+neg(C_d))./max(pos(C_d)+neg(C_u),eps) ;
        tempD = 0.5 * (sqrt(sum(C{i}.^2,2) + eps)).^(-1);
        D{i} = diag(tempD);
    end
    % update Y
    CTX=zeros(class_num,sample_num);
    for i = 1:view_num
        CTX=CTX+C{i}'*X{i};
    end
    Y_u=CTX+2*gamma*Y;
    Y_d=Y+2*tau*Y*L+2*gamma*Y*Y'*Y;
    Y=Y.*(pos(Y_u)+neg(Y_d))./max(pos(Y_d)+neg(Y_u),eps) ;
    Y=Y*diag(1./max(sum(Y,1),eps));
end

WW = [];
for i = 1:view_num
    WW = [WW;C{i}];
end
[~,ranking] = sort(sum(WW.*WW,2),'descend');
ttt=sum(WW.*WW,2);
ranking=ranking';

    function result = pos(data)
        result =(abs(data)+data)/2;
    end
    function result = neg(data)
        result =(abs(data)-data)/2;
    end
end