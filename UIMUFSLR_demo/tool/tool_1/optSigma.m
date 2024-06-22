function sigma = optSigma(X)
%input：X: row-sample  column-feature
%output:sigma
N = size(X,1); %sample number
dist = EuDist2(X,X);   
dist = reshape(dist,1,N*N); 
sigma = median(dist); 
end