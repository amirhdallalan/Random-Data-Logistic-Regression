clear
close all
clc

%% Import and Plot Data

% Load data
data = load('dataset.txt');
X = data(:,[1,2]);
y = data(:,3);

% Plot Data
PlotData(X, y);
hold on;

xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% Regularization and Optimization

X = MapFeature(X(:, 1), X(:, 2));
initial_theta = zeros(size(X, 2),1);
lambda = 1;

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, J, exit_flag] = ...
    fminunc(@(t) (CostFunc(t, X, y, lambda)), initial_theta, options);

PlotDecisionBoundary(theta, X, y);
hold on;

title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')

hold off;
%% Prediction and Resault

pred = Predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(pred == y)) * 100);

