function H = init_H(XX,c,options)
%XX: each row represents a sample.
%There are two options to initialize H (i.e., k-means or spectral clusterig).
n = size(XX,1);
if (~exist('options','var'))
   options = [];
end
if ~isfield(options,'init_method')
    options.init_method = 'SC'; %spectral clustering
end
switch lower(options.init_method)
    case {lower('SC')}
        [~,H] = ADA_ORTH_init(XX,c);
    case {lower('k_means')}
        labels = litekmeans(XX,c,'MaxIter',20,'Replicates',2);        
        H = zeros(n,c);
        for i = 1:n
            H(i,labels(i)) = 1;
        end
        H = H ./ repmat(sqrt(sum(H)),n,1);
    otherwise
        error('The init_method does not exist!');
end

end