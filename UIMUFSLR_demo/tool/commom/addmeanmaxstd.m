function [C]=addmeanmax(A)
%c 详见相册
[rnum,cnum]=size(A);
% 计算各行平均值
row_means = mean(A, 2);
row_std = std(A,0, 2);
% 将平均值作为新的列添加到矩阵
B = [A, row_means,row_std];
% 计算各列平均值
col_means = mean(B);
[~,max_idx]=max(B(:,cnum+1));
% 计算各列最大值
col_max=B(max_idx,:);
% 将平均值和最大值作为新的两行添加到矩阵
C = [B; col_means; col_max];