function [maerr, rmserr, Wabs] = RNN_v05_4(varargin)
% RNN_v05.1 A recurrent neural network with certain training phase
% Ref: Susillo and Abbott, 2009
% This version sets up the basic flow of the program, with FORCE training
% It plots the activity of nGN and actual output z.
% run by run_auto.m
% Update: from v05.3, use GPU computing, change matrix to singles, skip
% stationary networks

% v01 by Emilio Salinas, January 2021
% Junda Zhu, 3-17-2021
% clear all
%% parameters
para = varargin{1};
if length(para) ~= 8
    % network parameters
    nGN = 500;     % number of generator (recurrent) neurons
    tau = 10;    % membrane time constant, in ms
    p_GG = 0.1; % p of non zero recurrence
    p_z = 1; % p of non zero output
    alpha = 1;
    g = 1.5;
    % run parameters
    Tmax = 12000;   % training time (in ms)
    dt = 1;      % integration time step (in ms)
    
else % parameters given by user input
    nGN = para(1);
    tau = para(2);
    p_GG = para(3);
    p_z = para(4); % p of non zero output
    alpha = para(5);
    g = para(6);
    Tmax = para(7);
    dt = para(8);
end

whichfunc = 2; % which target function used (1-4)
%% initialize arrays
isstationary = 1;
while isstationary
x = gpuArray(2*rand(nGN,1,'single') - 1);
H = tanh(x);
J = gpuArray(zeros(nGN,'single'));
J(randperm(round(length(J(:))),round(p_GG*length(J(:))))) = randn(round(p_GG*length(J(:))),1)*g/sqrt(p_GG*nGN); %recurrent weight matrix
JGz = gpuArray(2*rand(nGN,1,'single')-1); %feedback weight matrix
W = gpuArray(randn(nGN,1,'single')/sqrt(p_z*nGN)); %output weight vector
P = gpuArray(eye(nGN,'single')/alpha); %update matrix
z = 0; %output
f = 0; %target
eneg = 0;

switch whichfunc % Target function
    case 1 % triangular wave of period 600 ms
        peri = 600;
        func = @(t,peri)(2*triangle(2*pi*(1/peri)*t)-1);
    case 2 % periodic function of period 1200 ms
        peri = 1200;
        func = @(t,peri)(sin(1.0*2*pi*(1/peri)*t) + ...
            1/2*sin(2.0*2*pi*(1/peri)*t) + ...
            1/6*sin(3.0*2*pi*(1/peri)*t) + ...
            1/3*sin(4.0*2*pi*(1/peri)*t));
    case 3 % square wave of period 600 ms
        peri = 600;
        func = @(t,peri)(2*(sin(t/peri*2*pi)>0)-1);
    case 4 % sine wave of period 60 ms or 8000 ms
        peri = 80*tau;
        func = @(t,peri)(sin(t/peri*2*pi));
end

nTmax = Tmax/dt;
T_start = peri * 2;
T_end = T_start + nTmax;
t=0;

% set space for data to be plotted
% tplot = NaN(1, nTmax);
% xplot = NaN(nplot, nTmax);
% Hplot = NaN(nplot, nTmax);
zplot = gpuArray(NaN(1, T_end+5*Tmax));
eplot = gpuArray(NaN(1, T_end+5*Tmax));

%Pretraining
for i=1:T_start-1
    H = tanh(x); % firing rates
    z = W' * H; % output
%     dw = - eneg * P * H; %dw
    dxdt = (-x + J*H + JGz*z)/tau;
    x = x + dxdt*dt;
    t = t + dt; 
    % save data
    %     tplot(i) = t;
    %     Hplot(:,i) = H(1:nplot);
        zplot(i) = z;
    %     dwplot(i) = norm(dw);
end
if sum(zscore(zplot(T_start-1:-1:T_start-peri))) ~= 0
    isstationary = 0;
end
end
%% -------------
% loop over time
%--------------- 
% Precompute target function
it = 1:1:T_end+5*Tmax;
f = gpuArray(func(it,peri));

% Main loop
for i=T_start:T_end
    H = tanh(x); % firing rates
    PH = P*H;
    P = P - PH*PH'/(1+H'*PH); % update P
    eneg = z - f(i); % error
    dw = - eneg * P * H;
    W = W + dw; % update W
    J = J + repmat(dw', nGN, 1); %update J (recurrent)
    z = W' * H; % output
    %             epos = z - f(i); % error after update
    
    dxdt = (-x + J*H) / tau;
    x = x + dxdt*dt;
    t = t + dt;
    
    %             % save some data for plotting
    %             tplot(i) = t;
    %             xplot(:,i) = x(1:nplot);
    %             Hplot(:,i) = H(1:nplot);
    %             zplot(i) = z;
    %             eplot(i) = epos - eneg;
    %             dwplot(i) = norm(dw);
    %             dwplot(T_start) = 0;
end

% Post training
for i=T_end+1:T_end+5*Tmax
    H = tanh(x); % firing rates
    eneg = z - f(i);
    z = W' * H; % output
    %             epos = z - f(i);    
    dxdt = (-x + J*H) / tau;
    x = x + dxdt*dt;
    t = t + dt;
    
    % save data
    %             tplot(i) = t;
    %             Hplot(:,i) = H(1:nplot);
    %             zplot(i) = z;
    eplot(i) = eneg;
    %             dwplot(i) = 0;
end
maerr = mean(abs(eplot(T_end+4*Tmax+1:T_end+5*Tmax)));
rmserr = sqrt(mean(eplot(T_end+4*Tmax+1:T_end+5*Tmax).^2));
Wabs = gather(norm(W));
end
