function [A,H] = ADA_ORTH_init(X,c)
%input: X -- each row represents a sample.
%       c -- cluster number.
%Construct the affinity matrix
t = optSigma(X)^2; %Median of Euclidean distance between all two samples
A = constructW(X, struct('k',5, 'WeightMode', 'HeatKernel', 't', t)); 
diag_ele_arr = sum(A);  
diag_ele_arr_t = diag_ele_arr.^(-1/2);
L = eye(size(X,1)) - diag(diag_ele_arr_t)* A *diag(diag_ele_arr_t);
L = (L + L')/2;
[eigvec, eigval] = eig(L);
[~, t1] = sort(diag(eigval), 'ascend');
eigvec = eigvec(:, t1(1:c));
eigvec = bsxfun(@rdivide, eigvec, sqrt(sum(eigvec.^2,2) + eps));  %Normalized eigenvector

%init H
rand('twister',5489); 
label = litekmeans(eigvec,c,'Replicates',10); 
H = rand(size(X,1),c);
for i = 1:size(X,1)
    H(i,label(i)) = 1;
end
H = H + 0.2;

end