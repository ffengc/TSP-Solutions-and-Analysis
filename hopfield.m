%
% hopfield.m
% write by Yufc: https://github.com/ffengc, 2024-12-24
%

clear all; clc;
global A D;

tic;  % 开始计时
%-----------------------------
% 1. 生成城市位置数据
%-----------------------------
citys=rand(40, 2);

%-----------------------------
% 2. 计算城市之间的距离矩阵
%-----------------------------
distance = dist(citys,citys'); % 这里使用 dist 函数，计算 citys 与其自身的欧氏距离矩阵

%-----------------------------
% 3. 初始化神经网络参数
%-----------------------------
N = size(citys, 1);         % 城市的数量
A = 200;                    % 能量函数中的惩罚系数，用于约束行、列的约束项
D = 100;                    % 能量函数中的惩罚系数，用于约束路径距离的项
U0 = 0.1;                   % tan-sigmoid 函数的规模参数
step = 0.0001;              % 每次迭代更新的步长
delta = 2 * rand(N,N) - 1;  % 在 (-1,1) 之间随机初始化
U = U0 * log(N-1) + delta;  % 初始输入神经元状态 U
V = (1 + tansig(U/U0))/2;   % 初始输出神经元状态 V，映射到 (0,1) 之间
iter_num = 10000;           % 迭代次数
E = zeros(1,iter_num);      % 用于记录每次迭代的能量值

%-----------------------------
% 4. 迭代寻优
%-----------------------------
for k = 1:iter_num
    dU = diff_u(V, distance);   % 计算 dU，即 Hopfield 网络的状态更新量
    U = U + dU*step;            % 更新输入神经元状态 U
    V = (1 + tansig(U/U0))/2;   % 通过双曲正切函数更新输出神经元状态 V
    e = energy(V,distance);     % 计算当前网络的能量函数值
    E(k) = e;                   % 记录本次迭代的能量函数值
end

%-----------------------------
% 5. 判断网络输出是否为有效解
%-----------------------------
[valid_flag, V1] = is_valid(V, N);
if valid_flag == 1
    disp('Get a optimal path');
    plot_result(N, citys, V1, iter_num, E);
else
    disp('The optimal path is invalid');
end

elapsedTime = toc;  % 结束计时，并返回运行时间（单位为秒）
fprintf('Opt Time: %.4f', elapsedTime);

%% functions
% -------------------------------------------
% 函数部分 1: is_valid(V, N)
% 判断路径有效性，即是否满足“每个城市只访问一次”并构成完整回路
% -------------------------------------------
function [flag, V1] = is_valid(V, N)
% V 是最终的输出神经元矩阵 (N×N)，理想情况是：
%   - 在每一列（对应城市在第几步访问），只有一个输出为1
%   - 在每一行（对应第 i 个城市），也只有一个输出为1
[rows,cols] = size(V);
V1 = zeros(rows,cols);  % 先构造一个同样规模的 0 矩阵
[~,V_ind] = max(V);     % 对每一列找到最大值（即最可能选中的神经元下标）
for j = 1:cols
    V1(V_ind(j),j) = 1; % 将 V1 对应位置赋值为 1，构建确定解
end
C = sum(V1,1);          % 每一列的求和，判断是否都为1
R = sum(V1,2);          % 每一行的求和，判断是否都为1
flag = isequal(C,ones(1,N)) & isequal(R',ones(1,N));
% 若每一列、每一行都是 1，则说明是有效解
end

% -------------------------------------------
% 函数部分 2: plot_result(N, citys, V1, iter_num, E)
% 绘制初始随机路径 VS. Hopfield 输出路径，并绘制能量函数收敛曲线
% -------------------------------------------
function plot_result(N, citys, V1, iter_num, E)
%-----------------------------
% 2.1 先计算一个随机路径的长度
%-----------------------------
sort_rand = randperm(N);            % 随机打乱城市顺序
citys_rand = citys(sort_rand,:);    % 得到随机的城市访问顺序
Length_init = dist(citys_rand(1,:),citys_rand(end,:)');
% 计算随机路径长度
for i = 2:size(citys_rand,1)
    Length_init = Length_init+dist(citys_rand(i-1,:),citys_rand(i,:)');
end
%-----------------------------
% 2.2 绘制随机路径
%-----------------------------
figure('Position', [100, 100, 1000, 300]);
hold on;
subplot(1, 3, 1);
plot([citys_rand(:,1);citys_rand(1,1)],[citys_rand(:,2);citys_rand(1,2)],'o-','LineWidth',1, 'Color', '#FFA500');
for i = 1:length(citys)
    text(citys(i,1),citys(i,2),['   ' num2str(i)])
end
text(citys_rand(1,1),citys_rand(1,2),['       Start Point' ])
text(citys_rand(end,1),citys_rand(end,2),['       End Point' ])
title(['Orginal Solution(Length：' num2str(Length_init) ')'])
axis([0 1 0 1])
grid on;
xlabel('X axis');
ylabel('Y axis');
%-----------------------------
% 2.3 使用 Hopfield 确定的路径
%-----------------------------
[~,V1_ind] = max(V1);           % 每列选出最大值的索引，得到具体访问顺序
citys_end = citys(V1_ind,:);    % 据此得到优化后的城市访问顺序

% 计算该最优路径长度
Length_end = dist(citys_end(1,:),citys_end(end,:)');
for i = 2:size(citys_end,1)
    Length_end = Length_end+dist(citys_end(i-1,:),citys_end(i,:)');
end
disp('Hopfield Solution Matrix');   % 控制台显示提示
% 绘制最优路径
% figure(2)
subplot(1, 3, 2);
plot([citys_end(:,1);citys_end(1,1)],...
    [citys_end(:,2);citys_end(1,2)],'o-','LineWidth',1, 'Color', '#FFA500');
for i = 1:length(citys)
    text(citys(i,1),citys(i,2),['  ' num2str(i)])
end
text(citys_end(1,1),citys_end(1,2),['       Start Point' ])
text(citys_end(end,1),citys_end(end,2),['       End Point' ])
title(['Hopfield Solution(Length：' num2str(Length_end) ')'])
axis([0 1 0 1])
grid on;
xlabel('X axis');
ylabel('Y axis');
%-----------------------------
% 2.4 绘制能量函数迭代曲线
%-----------------------------
subplot(1, 3, 3);
plot(1:iter_num,E,'LineWidth',2);
ylim([0 2000])
title(['Energy function(Optimal Energy：' num2str(E(end)) ')']);
xlabel('Epoch');
ylabel('Energy function');
grid on;
legend('Energy function');

sgtitle(sprintf('Number of cities: %d', N));
hold off;
end

% -------------------------------------------
% 函数部分 3: diff_u(V, d)
% 计算 Hopfield 网络中状态 U 的导数 dU
% -------------------------------------------
function du=diff_u(V,d)
global A D;
% V 大小为 N×N， d 为距离矩阵
n=size(V,1);
% 1) sum(V,2) - 1 表示每一行之和距离 1 的偏差 (V 满足行约束)
%    sum_x 就是将这个偏差复制到 n×n 矩阵
sum_x=repmat(sum(V,2)-1,1,n);
% 2) sum(V,1) - 1 表示每一列之和距离 1 的偏差 (V 满足列约束)
%    sum_i 就是将这个偏差复制到 n×n 矩阵
sum_i=repmat(sum(V,1)-1,n,1);
% 3) 处理路径距离约束：将输出矩阵 V 右移 1 列做比较
V_temp=V(:,2:n);
V_temp=[V_temp V(:,1)];
% 4) 求 d * V_temp，即计算距离矩阵与下一列神经元对应元素相乘的加权和
sum_d=d*V_temp;
% 5) dU 即 Hopfield 神经网络的状态更新方程
du=-A*sum_x-A*sum_i-D*sum_d;
%    注意这里的负号是因为梯度下降法，对于能量函数 E，dU = -∂E/∂V
end


% -------------------------------------------
% 函数部分 4: energy(V, d)
% 计算当前输出 V 对应的能量值 E
% -------------------------------------------
function E=energy(V,d)
global A D
n=size(V,1);
% 1) sum(V,2) - 1 表示每行之和与 1 的偏差，用 sumsqr() 计算误差平方和
sum_x=sumsqr(sum(V,2)-1);
% 2) sum(V,1) - 1 表示每列之和与 1 的偏差
sum_i=sumsqr(sum(V,1)-1);
% 3) 处理路径距离约束：同上，将 V 右移 1 列形成环
V_temp=V(:,2:n);
V_temp=[V_temp V(:,1)];
sum_d=d*V_temp; % d 为距离矩阵
% 对应能量中路径部分
sum_d=sum(sum(V.*sum_d)); % V.*(d * V_temp) 并求和
% 4) 总能量函数为 0.5(A * sum_x + A * sum_i + D * sum_d)
%    其中系数 0.5 是因为 E = 1/2 * ∑(...)，可避免重复计算
E=0.5*(A*sum_x+A*sum_i+D*sum_d);
end


