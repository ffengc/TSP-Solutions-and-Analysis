%
% dp.m
% write by Yufc: https://github.com/ffengc, 2024-12-24
%

clear;clc;
close all;

DP_TSP_main();

function DP_TSP_main()
% =========== 1. 生成城市位置 & 距离矩阵 ===========
citys = rand(5, 2);                 % n个城市的随机坐标
N = size(citys, 1);                  % N=30
distance = dist(citys, citys');      % 计算城市间距离矩阵

% =========== 2. 动态规划求解TSP ===========
% 这里返回最优路径 bestPath，以及其对应的最短距离 bestCost
[bestPath, bestCost] = tsp_dp(distance);

% =========== 3. 将bestPath转换为与plot_result相同格式的V1矩阵 ===========
%   V1是一个 N×N 的0-1矩阵，第 j 列为第 j 步访问的城市
%   若第 j 步访问的是 bestPath(j)，则 V1( bestPath(j), j ) = 1
V1 = zeros(N, N);
for stepIdx = 1:N
    cityIdx = bestPath(stepIdx);
    V1(cityIdx, stepIdx) = 1;
end

% =========== 4. 绘图结果 ===========
%   直接调用之前的plot_result函数，展示初始随机路径 vs DP获得的最优路径
plot_result(N, citys, V1);

% 控制台输出
disp(['DP best route： ', mat2str(bestPath)]);
disp(['DP shortest distance： ', num2str(bestCost)]);
end


function [bestPath, bestCost] = tsp_dp(distance)
% TSP_DP 使用 Held-Karp 动态规划算法求解TSP
%   distance: N×N 距离矩阵
%   bestPath: 最优路线（长度 N 的向量），从城市1出发并回到城市1
%   bestCost: 对应的最优路长

% ------ 一些初始化 ------
N = size(distance, 1);
INF = 1e10;   % 大数

% 我们假定：从城市1开始，并最终回到城市1
% 状态定义：dp[mask, i] 表示在“已访问城市集合”为 mask，且“最后访问城市”为 i 时的最小路径距离
% mask 用二进制表示哪几个城市已访问过

nStates = 2^N;
dp = INF * ones(nStates, N, 'double');
% 用于回溯路径
parent = -1 * ones(nStates, N, 'int32');

% 初始状态：只访问了城市1，最后停在城市1
dp(1, 1) = 0;

% ------ 动态规划: 遍历所有子集和可能的末尾城市 ------
for mask = 1 : nStates
    % 对应已访问过的城市集合: mask
    % 找出集合mask中包含哪些城市
    % 这里通过位运算来判断
    for lastCity = 1 : N
        % 若lastCity不在mask中，则跳过
        if ~bitand(mask, bitshift(1, lastCity-1))
            continue;
        end
        % 当前dp[mask, lastCity]
        currCost = dp(mask, lastCity);
        if currCost >= INF
            continue;
        end
        % 枚举下一个要访问的城市 nextCity
        for nextCity = 1 : N
            % 如果 nextCity 已在集合mask中，则跳过
            if bitand(mask, bitshift(1, nextCity-1))
                continue;
            end
            % 新的子集
            nextMask = bitor(mask, bitshift(1, nextCity-1));
            % 更新dp
            newCost = currCost + distance(lastCity, nextCity);
            if newCost < dp(nextMask, nextCity)
                dp(nextMask, nextCity) = newCost;
                parent(nextMask, nextCity) = lastCity;
            end
        end
    end
end

% ------ 最后，还需回到城市1 ------
% mask = (1<<N)-1 表示所有城市都已访问
fullMask = bitshift(1, N) - 1;
bestCost = INF;
bestLastCity = -1;

for i = 2 : N
    tempCost = dp(fullMask, i) + distance(i, 1);
    if tempCost < bestCost
        bestCost = tempCost;
        bestLastCity = i;
    end
end

% ------ 回溯得到最优路径 ------
bestPath = zeros(1, N, 'int32');
bestPath(N) = 1;          % 回到城市1
mask = fullMask;
curCity = bestLastCity;

for pos = N-1 : -1 : 1
    bestPath(pos) = curCity;
    prevCity = parent(mask, curCity);
    mask = bitxor(mask, bitshift(1, curCity-1));
    curCity = prevCity;
end

% 此时bestPath是[1,x2,x3,...,xN(=1)]，
% 但若希望 strictly "从城市1出发"并在最后返回1，
% 我们可以把最末一个1去掉，因为plot_result里只需要访问顺序
% 不过看你的需求，本例里把N个城市顺序输出就行了
% （plot_result里会自动连回首城市）
end

function plot_result(N, citys, V1)
% ===== 1) 计算并绘制初始随机路径 =====
sort_rand = randperm(N);
citys_rand = citys(sort_rand, :);
Length_init = dist(citys_rand(1,:), citys_rand(end,:)');
for i = 2:N
    Length_init = Length_init + dist(citys_rand(i-1,:), citys_rand(i,:)');
end

figure('Position', [100, 100, 1000, 300]);
hold on;
subplot(1, 2, 1);
plot([citys_rand(:,1); citys_rand(1,1)], ...
    [citys_rand(:,2); citys_rand(1,2)], 'o-', ...
    'LineWidth',1,'Color','#FFA500');
title(['Random Path (Length = ' num2str(Length_init) ')']);
axis([0 1 0 1]); grid on; xlabel('X axis'); ylabel('Y axis');

% ===== 2) 由 V1 获取最终路径并绘制 =====
[~, V1_ind] = max(V1);   % 每列最大值索引
citys_end = citys(V1_ind, :);
Length_end = dist(citys_end(1,:), citys_end(end,:)');
for i = 2:N
    Length_end = Length_end + dist(citys_end(i-1,:), citys_end(i,:)');
end

subplot(1, 2, 2);
plot([citys_end(:,1); citys_end(1,1)], ...
    [citys_end(:,2); citys_end(1,2)], 'o-', ...
    'LineWidth',1,'Color','#FFA500');
title(['DP Solution (Length = ' num2str(Length_end) ')']);
axis([0 1 0 1]); grid on; xlabel('X axis'); ylabel('Y axis');
sgtitle(sprintf('Number of cities: %d', N));
hold off;
end