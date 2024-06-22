function S = Update_S(A,Y,V,beta,gamma,delta)
n = size(A{1,1},1);
G = cell(1,V);
for i = 1:V
    G{1,i} = L2_distance_1(Y{1,i},Y{1,i});  
end

S = zeros(n);
for i = 1:n
    gi = zeros(1,n);
    deltaAi = zeros(1,n);
    for i1 = 1:V
        gi = gi + G{1,i1}(i,:);  
        deltaAi = deltaAi + 2 * delta(i1) * A{1,i1}(i,:);
    end
    gi_temp = gamma / (2 * beta) * gi;
    r = (  deltaAi - gi_temp ) / (2 * V);
    S(i,:) = EProjSimplex_new(r,1);  
end


end