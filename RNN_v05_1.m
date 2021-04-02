function RNN_v05_1(varargin)
% RNN_v05.1 A recurrent neural network with certain training phase
% Ref: Susillo and Abbott, 2009
% This version sets up the basic flow of the program, with FORCE training
% It plots the activity of nGN and actual output z.
% run by run_auto.m
% Update: from v04.3, apply FORCE to recurrent not Wout

% v01 by Emilio Salinas, January 2021
% Junda Zhu, 3-8-2021
tic
% clear all
%% parameters
para = varargin{1};
if length(para) ~= 5
    % network parameters
    nGN = 1000;     % number of generator (recurrent) neurons
    tau = 10;    % membrane time constant, in ms
    % run parameters
    Tmax = 1000;   % training time (in ms)
    dt = 1;      % integration time step (in ms)
    nplot = 7;   % number of generator neurons to plot;
else % parameters given by user input
    nGN = para(1);
    tau = para(2);
    Tmax = para(3);
    dt = para(4);
    nplot = para(5);
end
if nplot > nGN
    nplot = nGN;
end
whichfunc = 1; % which target function used (1-4)
p_z = 1; % p of non zero output
p_GG = 0.1; % p of non zero recurrence
alpha = 1;
%% initialize arrays
x = 2*rand(nGN,1) - 1;
H = tanh(x);
g = 1.5;
J = zeros(nGN);
J(randperm(length(J(:)),p_GG*length(J(:)))) = randn(p_GG*length(J(:)),1)*g/sqrt(p_GG*nGN); %recurrent weight matrix
% JGz = 2*rand(nGN,1)-1; %feedback weight matrix
W = randn(nGN,1)/sqrt(p_z*nGN); %output weight vector
P = eye(nGN)/alpha; %update matrix
z = 0; %output
f = 0; %target
eneg = 0;

% set space for data to be plotted
nTmax = Tmax/dt;
tplot = NaN(1, nTmax);
xplot = NaN(nplot, nTmax);
Hplot = NaN(nplot, nTmax);
zplot = NaN(1, nTmax);

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
con = 'Y';
T_start = 2000;
T_end = T_start + nTmax;
t=0;
for it=1:T_end+5*Tmax % precompute target function
    f(it) = func(it,peri);
    fplot = f;
end
while con ~= 'N'
    if con == 'Y'
        toc
        for i=1:T_start-1
            H = tanh(x); % firing rates
            z = W' * H; % output
            dw = eneg * P * H; %dw
            dxdt = (-x + J*H) / tau;
            x = x + dxdt*dt;
            t = t + dt;
            
            % save some data for plotting
            tplot(i) = t;
            Hplot(:,i) = H(1:nplot);
            zplot(i) = z;
            dwplot(i) = norm(dw);
        end
        toc
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
            epos = z - f(i); % error after update
            
            dxdt = (-x + J*H) / tau;
            x = x + dxdt*dt;
            t = t + dt;
            
            % save some data for plotting
            tplot(i) = t;
            xplot(:,i) = x(1:nplot);
            Hplot(:,i) = H(1:nplot);
            zplot(i) = z;
            eplot(i) = epos - eneg;
            dwplot(i) = norm(dw);
            dwplot(T_start) = 0;
        end
        toc
        % Post training
        for i=T_end+1:T_end+5*Tmax
            H = tanh(x); % firing rates
            eneg = z - f(i);
            z = W' * H; % output
            epos = z - f(i);
            
            dxdt = (-x + J*H) / tau;
            x = x + dxdt*dt;
            t = t + dt;
            
            % save some data for plotting
            tplot(i) = t;
            Hplot(:,i) = H(1:nplot);
            zplot(i) = z;
            eplot(i) = epos - eneg;
            dwplot(i) = 0;
        end
        toc
        disp('plotting');
        % graph the results
        clrGN = 'k';
        clrOut = 'r';
        clrF = 'g';
        clr_grid = 0.5*[1 1 1];
        % scale factor for plotting activity one neuron per row
        sfac = 0.5;
        clf
        subplot(3,1,1)
        hold on
%         xlim([0 T_end+2000])
        ylim([-1.2 1.2])
        set(gca, 'YTick', [-1, 0, 1])
        line([T_start T_start],[-1.2, 1.2])
        line([T_end T_end],[-1.2, 1.2])
        plot(tplot, fplot, '-', 'color', clrF, 'LineWidth', 2);
        plot(tplot, zplot, '-', 'color', clrOut, 'LineWidth', 2);
        ylabel('Output Unit');
        xlabel('Time (ms)');
        title(['RNN v04: ' num2str(nGN) ' neurons, with recurrence and training']);
        
        subplot(3,1,2)
        hold on
%         xlim([0 T_end+2000])
        ylim([0.25 nplot+0.75])
        set(gca, 'YTick', [1:nplot])
        for ii=1:nplot
            yoff = (ii-1) + 1;
            line([T_start T_start],[0.25, nplot+1])
            line([T_end T_end],[0.25, nplot+1])
            plot(xlim, yoff*[1 1], ':', 'color', clr_grid)
            plot(tplot, Hplot(ii,:)*sfac + yoff, '-', 'color', clrGN, 'LineWidth', 1);
        end
        ylabel('Recurrent neuron');
        xlabel('Time (ms)');
        
        subplot(3,1,3)
        hold on
%         xlim([0 T_end+2000])
        ylim([-0.15 0.15])
        set(gca, 'YTick', [-0.1, 0, 0.1])
        line([T_start T_start],[-1 1])
        line([T_end T_end],[-1 1])
        plot(tplot, eplot, '.', 'color', 'k');
        plot(tplot, dwplot/max(dwplot)*0.3-0.15, '.', 'color', 'c');
        ylabel('\delta Error');
        xlabel('Time (ms)');
        
        T_start = T_start + nTmax;
        T_end = T_start + nTmax;
    elseif con == 'N'
        break;
    else
        disp('Wrong input');
    end
    con = input('continue? [Y/N]:','s');
    if isempty(con)
        con = 'dumb';
    end
end