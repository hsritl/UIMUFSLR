function delta = solve_delta(S,A)
% Problem
%
%  min   sigma(v=1 to V)|| S - deta^v * A^v||^2
%  s.t. deta>=0, 1'deta=1
%
View = size(A,2);
p = [];
q = [];
for v = 1:View
    p_v = trace(A{1,v} * S');
    q_v = trace(A{1,v} * A{1,v}');
    p = [p; p_v];
    q = [q; q_v];
end

g = [];
for v = 1:View
    g_v = p(v) / q(v) + ( 1-sum(p./q) ) / ( q(v)*sum(1./q) );
    g = [g; g_v ];
end

gmin = min(g);
ft = 1; %Maximum number of iterations of Newton method
if gmin < 0
    f = 1; %Initial function value
    miu = 0; %Initial miu
    sumq = sum(1./q);
    der_temp = ( 1/sumq )./q; %Coefficient of derivative of summation term
    while abs(f) > 10^-10 %Until we find the root
        max0 = ( miu/sumq )./q - g; %Summation term
        posidx = max0>0; %The part whose summation term is greater than 0
        der = der_temp(posidx) -1; %Derivative of iteration point
        f = sum(max0(posidx)) - miu; %Function value of iteration point
        miu = miu - f/der;
        ft = ft + 1;
        if ft > 1000
            break;
        end
    end
    delta_temp = g - ( miu/sumq ) ./ q  ;
    delta = max(delta_temp,0);
else
    delta = max(g,0);
end
end