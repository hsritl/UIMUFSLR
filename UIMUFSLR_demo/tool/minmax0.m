% Source Code for Min-Max [0, 1] normalization
%
% Input Parameters:
% train:    training data where rows represent instances and columns 
%           represents  features
%
% Output Parameters: 
% ntrain:   Min-Max normalized training data
%
% If you use the code, please cite the paper:
%
% Dalwinder Singh and Birmohan Singh,"Investigating the impact of data 
% normalization on classification performance", Applied Soft Computing,
% 2019.
% https://doi.org/10.1016/j.asoc.2019.105524

function ntrain = minmax0( train )
[in,d]=size(train);
ntrain=zeros(in,d);
mn=min(train);
mx=max(train);
for j=1:d
    ntrain(:,j)=(train(:,j)-mn(j))/((mx(j)-mn(j))+eps);
end
end