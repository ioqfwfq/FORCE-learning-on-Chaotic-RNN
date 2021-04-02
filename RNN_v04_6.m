function [maerr, rmserr, Wabs] = RNN_v04_6(varargin)
% RNN_v04.2 A recurrent neural network with certain training phase
% Ref: Susillo and Abbott, 2009
% This version sets up the basic flow of the program, with FORCE training
% only on W_out
% run by run_auto_v04.m
% Update: addde g as param input

% v01 by Emilio Salinas, January 2021
% Junda Zhu, 3-15-2021
% clear all
%% parameters
para = varargin{1};
if length(para) ~= 6
    % network parameters
    nGN = 1000;     % number of generator (recurrent) neurons
    tau = 10;    % membrane time constant, in ms
    % run parameters
    Tmax = 12000;   % training time (in ms)
    dt = 1;      % integration time step (in ms)
    g = 1.5;
    p_GG = 0.1; % p of non zero recurrence
else % parameters given by user input
    nGN = para(1);
    tau = para(2);
    Tmax = para(3);
    dt = para(4);
    g = para(5);
    p_GG = para(6);
end

whichfunc = 4; % which target function used (1-4)
p_z = 1; % p of non zero output
alpha = 1;
%% initialize arrays
x = 2*rand(nGN,1) - 1;
H = tanh(x);
J = zeros(nGN);
J(randperm(length(J(:)),p_GG*length(J(:)))) = randn(round(p_GG*length(J(:))),1)*g/sqrt(p_GG*nGN); %recurrent weight matrix
JGz = 2*rand(nGN,1)-1; %feedback weight matrix
W = randn(nGN,1)/sqrt(p_z*nGN); %output weight vector
P = eye(nGN)/alpha; %update matrix
z = 0; %output
f = 0; %target
eneg = 0;

% set space for data to be plotted
nTmax = Tmax/dt;
% tplot = NaN(1, nTmax);
% xplot = NaN(nplot, nTmax);
% Hplot = NaN(nplot, nTmax);
% zplot = NaN(1, nTmax);

% Target function
switch whichfunc
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
%% -------------
% loop over time
%---------------
% con = 'Y';
T_start = 2;
T_end = T_start + nTmax;
t=0;
for it=1:T_end+5*Tmax % precompute target function
    f(it) = func(it,peri);
    fplot = f;
end
% while con ~= 'N'
%     if con == 'Y'
for i=1:T_start-1
    H = tanh(x); % firing rates
    z = W' * H; % output
    dw = eneg * P * H; %dw
    dxdt = -x/tau + J*H/tau + JGz*z/tau;
    x = x + dxdt*dt;
    t = t + dt;
    
%     % save data
%     tplot(i) = t;
%     Hplot(:,i) = H(1:nplot);
%     zplot(i) = z;
%     dwplot(i) = norm(dw);
end
% Main loop
for i=T_start:T_end
    H = tanh(x); % firing rates
    PH = P*H;
    P = P - PH*PH'/(1+H'*PH); % update P
    eneg = z - f(i); % error
    dw = - eneg * P * H;
    W = W + dw; % update W
    z = W' * H; % output
%     epos = z - f(i); % error after update
    
    dxdt = (-x + J*H + JGz*z)/tau;
    x = x + dxdt*dt;
    t = t + dt;
    
%     % save data
%     tplot(i) = t;
%     xplot(:,i) = x(1:nplot);
%     Hplot(:,i) = H(1:nplot);
%     zplot(i) = z;
%     eplot(i) = epos - eneg;
%     dwplot(i) = mean(abs(dw(:)));

end
% Post training
for i=T_end+1:T_end+5*Tmax
    H = tanh(x); % firing rates
    eneg = z - f(i);
    z = W' * H; % output
%     epos = z - f(i);
    
    dxdt = (-x + J*H + JGz*z)/tau;
    x = x + dxdt*dt;
    t = t + dt;
    
    % save data
%     tplot(i) = t;
%     Hplot(:,i) = H(1:nplot);
%     zplot(i) = z;
    eplot(i) = eneg;
%     dwplot(i) = norm(dw);
end
maerr = mean(abs(eplot(T_end+4*Tmax+1:T_end+5*Tmax)));
rmserr = sqrt(mean(eplot(T_end+4*Tmax+1:T_end+5*Tmax).^2));
Wabs = norm(W);
% disp('run finished');

% T_start = T_start + nTmax;
% T_end = T_start + nTmax;
%     elseif con == 'N'
%         break;
%     else
%         disp('Wrong input');
%     end
%     con = input('continue? [Y/N]:','s');
%     if isempty(con)
%         con = 'dumb';
%     end
end
