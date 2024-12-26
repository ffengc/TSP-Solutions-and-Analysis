%
% ga.m
% Written by Yufc: https://github.com/ffengc, 2024-12-24
%

clear;clc;
close all;

GA_TSP_main();

function GA_TSP_main()
%
% ga function
%
citys = rand(100, 2); % 数据
N = size(citys, 1);
distance = pdist2(citys, citys, 'euclidean');  % 计算城市间距离矩阵

% 遗传算法参数
populationSize = 50;     % 种群大小
generations = 5000;      % 迭代代数
break_generations = 200; % 退出条件, 如果连续break_generations代bestCost没有变化, 则退出循环
crossoverRate = 0.8;     % 交叉概率
mutationRate = 0.5;      % 变异概率
% eliteCount = 2;        % 精英保留数量

% 初始化种群
population = init_population(populationSize, N);
% 计算此时种群的适应度
fitness = eval_fitness(populationSize, population, distance);
% 记录初始种群最佳适应度
[bestFitness, bestIdx] = min(fitness);
bestPath = population(bestIdx, :);
bestCost = bestFitness;
costList = [];
% 开始迭代进化
for gen = 1:generations
    % 选择：基于轮盘赌选择
    selected = selection(population, fitness, populationSize);
    assert(is_population_valid(selected, populationSize), 'the population is invalid');
    % note: 选择完之后, 种群里面一定有重复的元素
    % 交叉：有概率进行交叉
    offspring = crossover(selected, crossoverRate, N);
    assert(is_population_valid(offspring, populationSize), 'the population is invalid');
    % 变异
    offspring = mutation(offspring, mutationRate, N);
    assert(is_population_valid(offspring, populationSize), 'the population is invalid');
    % 计算当前种群的适应度
    offspringFitness = eval_fitness(populationSize, offspring, distance);

    % 精英保留：保留最好的 eliteCount 个个体
    % 将当前种群和后代种群的适应度和个体进行拼接
    combinedFitness = [fitness; offspringFitness];
    combinedPopulation = [population; offspring];

    % 对所有适应度进行排序，选择最优的 populationSize 个个体
    [sortedFitness, sortedIdx] = sort(combinedFitness);
    population = combinedPopulation(sortedIdx(1:populationSize), :); % 选择fitness更小的50个对象
    fitness = sortedFitness(1:populationSize);

    % 更新最佳路径
    if fitness(1) < bestCost
        bestCost = fitness(1);
        bestPath = population(1, :);
    end
    costList = [costList, bestCost];
    % 可选：显示进度
    if mod(gen, 1) == 0
        fprintf('Generation %d/%d: Best Cost = %.4f\n', gen, generations, bestCost);
    end
    % break?
    if length(costList) > break_generations
        last_n_elements = costList(end - break_generations + 1 : end);
        if length(unique(last_n_elements)) == 1
            disp('Best Cost remains stable, break');
            break; % 长时间没有优化
        end
    end
end
% disp(bestPath); % 最佳路线
V1 = zeros(N, N);
for stepIdx = 1:N
    cityIdx = bestPath(stepIdx);
    V1(cityIdx, stepIdx) = 1;
end
plot_result(N, citys, V1, costList);
end

%% functions
function population = init_population(populationSize, N)
%
% input:
%   populationSize 需要初始化的种群大小
%   N 城市的数量
% return:
%   初始化种群, 返回一个二维数组, N列, populationSize行
%
for i = 1:populationSize
    population(i, :) = randperm(N);
end
end

function fitness = eval_fitness(populationSize, population, distance)
%
% input:
%   populationSize 种群大小; population种群; distance距离矩阵
% return:
%   fitness 长为 populationSize 的数组, 表示适应度
%
fitness = zeros(populationSize, 1);
for i = 1:populationSize
    route = population(i, :);
    fitness(i) = calculate_route_distance(route, distance);
end
end

function routeDistance = calculate_route_distance(route, distance)
%
% input:
%   route 路径; distance距离矩阵;
% return:
%   route_distance计算这个路径的距离
%
N = length(route);
routeDistance = 0;
for j = 1:N-1
    routeDistance = routeDistance + distance(route(j), route(j+1));
end
% 回到起点
routeDistance = routeDistance + distance(route(N), route(1));
end

function selected = selection(population, fitness, populationSize)
%
% input:
%   population 当前种群; fitness当前种群的适应度; populationSize当前种群的大小
% return:
%   selected 被选择过后的种群
%
% 将适应度转换为适应度值越大越好（此处最小化问题，逆转适应度）
maxFit = max(fitness);
adjustedFitness = maxFit - fitness + 1e-6;  % 防止为0
totalFit = sum(adjustedFitness);
prob = adjustedFitness / totalFit;

% 计算累积概率
cumProb = cumsum(prob);

% 选择
selected = zeros(size(population));
for i = 1:populationSize
    r = rand();
    selectedIdx = find(cumProb >= r, 1, 'first');
    selected(i, :) = population(selectedIdx, :);
end
end

function offspring = crossover(selected, crossoverRate, N)
%
% 交叉操作
%
populationSize = size(selected, 1);
offspring = selected;
for i = 1:2:populationSize-1
    if rand() < crossoverRate
        parent1 = selected(i, :);
        parent2 = selected(i+1, :);
        [child1, child2] = pmx(parent1, parent2, N);
        offspring(i, :) = child1;
        offspring(i+1, :) = child2;
    end
end
end

% PMX交叉实现
function [child1, child2] = pmx(parent1, parent2, N)
% 随机选择交叉点
pt = sort(randperm(N, 2));
pt1 = pt(1);
pt2 = pt(2);

% 初始化子代
child1 = parent1;
child2 = parent2;

% 复制交叉区域
child1(pt1:pt2) = parent2(pt1:pt2);
child2(pt1:pt2) = parent1(pt1:pt2);
benchmark = 1:N;
% parent
% parent1: 1 3 4 5 2
% parent2: 4 5 2 1 3
% fix duplicate
% child1: 1 3 2 1 3
% child2: 4 5 4 5 2
child1_left = setdiff(benchmark, child1(pt1:pt2));
child2_left = setdiff(benchmark, child2(pt1:pt2));

% 先处理 child1 中的问题
n1 = length(child1_left);
n2 = length(child2_left);
assert(n1 == n2);
randomIdx1 = randperm(n1);
randomIdx2 = randperm(n2);
child1_left_random = child1_left(randomIdx1);
child2_left_random = child2_left(randomIdx2);
insert_idx = 1;
for i=1:N
    if i < pt1 || i > pt2
        child1(i) = child1_left_random(insert_idx);
        child2(i) = child2_left_random(insert_idx);
        insert_idx = insert_idx + 1;
    end
end
assert(length(unique(child1)) == length(child1));
assert(length(unique(child2)) == length(child2));
end

function offspring = mutation(offspring, mutationRate, N)
%
% 变异操作
% input:
%   offspring后代种群; mutationRate变异率; N城市个数;
%
populationSize = size(offspring, 1);
for i = 1:populationSize
    if rand() < mutationRate
        % 随机选择两个基因位置进行交换
        swapIdx = randperm(N, 2);
        % 交换
        temp = offspring(i, swapIdx(1));
        offspring(i, swapIdx(1)) = offspring(i, swapIdx(2));
        offspring(i, swapIdx(2)) = temp;
    end
end
end

function flag = is_population_valid(population, populationSize)
for i = 1:populationSize
    cur_route = population(i,:);
    cur_route_unique = unique(cur_route);
    if length(cur_route_unique) ~= length(cur_route)
        flag = false;
        return;
    end
    flag = true;
end
end


function plot_result(N, citys, V1, costList)
% ===== 1) 计算并绘制初始随机路径 =====
sort_rand = randperm(N);
citys_rand = citys(sort_rand, :);
Length_init = dist(citys_rand(1,:), citys_rand(end,:)');
for i = 2:N
    Length_init = Length_init + dist(citys_rand(i-1,:), citys_rand(i,:)');
end

figure('Position', [100, 100, 1000, 300]);
hold on;
subplot(1, 3, 1);
plot([citys_rand(:,1); citys_rand(1,1)], ...
    [citys_rand(:,2); citys_rand(1,2)], 'o-', ...
    'LineWidth',1,'Color','#FFA500');
title(['Origin Path (Length = ' num2str(Length_init) ')']);
axis([0 1 0 1]); grid on; xlabel('X axis'); ylabel('Y axis');

% ===== 2) 由 V1 获取最终路径并绘制 =====
[~, V1_ind] = max(V1);   % 每列最大值索引
citys_end = citys(V1_ind, :);
Length_end = dist(citys_end(1,:), citys_end(end,:)');
for i = 2:N
    Length_end = Length_end + dist(citys_end(i-1,:), citys_end(i,:)');
end

subplot(1, 3, 2);
plot([citys_end(:,1); citys_end(1,1)], ...
    [citys_end(:,2); citys_end(1,2)], 'o-', ...
    'LineWidth',1,'Color','#FFA500');
title(['GA Solution (Length = ' num2str(Length_end) ')']);
axis([0 1 0 1]); grid on; xlabel('X axis'); ylabel('Y axis');
subplot(1, 3, 3);
plot(1:length(costList), costList, "LineWidth", 2);
grid on;
legend("Best Cost");
xlabel("Epoch");
ylabel("Best Cost");
title("Best Cost in Epochs");
sgtitle(sprintf('Number of cities: %d', N));
hold off;
end


