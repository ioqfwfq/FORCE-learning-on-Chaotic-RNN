function RNN_v02_2(varargin)
% RNN_v02.2 Skeleton for running a recurrent neural network
%
% This version sets up the basic flow of the program, with actual recurrent 
% interactions. It plots the activity of nGN
% "generator neurons" whose activites fall exponentially over time.
% run by run_auto.m
% Update: Continue plotting the same GNs with the same parameters.

% v01 by Emilio Salinas, January 2021
% J Zhu, 2-5-2021

% clear all
para = varargin{1};
if length(para) ~= 5
    %
    % network parameters
    %
    nGN = 150;     % number of generator (recurrent) neurons
    tau = 10;    % membrane time constant, in ms
    %
    % run parameters
    %
    Tmax = 1000;   % simulation time (in ms)
    dt = 1;      % integration time step (in ms)
    nplot = 5;   % number of generator neurons to plot;
else
    nGN = para(1);
    tau = para(2);
    Tmax = para(3);
    dt = para(4);
    nplot = para(5);
end
if nplot > nGN
    nplot = nGN;
end

%
% initialize arrays
%
x = -1 + 2*rand(nGN,1);
g = 1.5;
J = randn(nGN)*g/sqrt(nGN);

% set space for data to be plotted
nTmax = Tmax/dt;
tplot = NaN(1, nTmax);
xplot = NaN(nplot, nTmax);

%---------------
% loop over time
%---------------
con = 'Y';
T_start = 1;
T_end = T_start + nTmax;
t=0;
while con ~= 'N'
    if con == 'Y'
        for j=T_start:T_end
            t = t + dt;
            H = tanh(x); % firing rates H(x)
            % tau dx/dt = -x + J*H(x)
            dxdt = -x/tau + J*H/tau;
            x = x + dxdt*dt;
            
            % save some data for plotting
            tplot(j) = t;
            xplot(:,j) = x(1:nplot);
            Hplot(:,j) = H(1:nplot);
        end
        %
        % graph the results
        %
        clrGN = 'r';
        clr_grid = 0.5*[1 1 1];
        
        % scale factor for plotting activity one neuron per row
        sfac = 0.5;
        
        clf
        hold on
        xlim([T_start T_end+1])
        ylim([0.25 nplot+0.75])
        set(gca, 'YTick', [1:nplot])
        
        for j=1:nplot
            yoff = (j-1) + 1;
            plot(xlim, yoff*[1 1], ':', 'color', clr_grid)
            %     plot(tplot, xplot(j,:)*sfac + yoff, '-', 'color', clrGN);
            plot(tplot, Hplot(j,:)*sfac + yoff, '-', 'color', clrGN, 'LineWidth', 2);
        end
        ylabel('Recurrent neuron');
        xlabel('Time (ms)');
        title(['RNN v03: ' num2str(nGN) ' neurons, with recurrence']);
        T_start = T_start + nTmax;
        T_end = T_start + nTmax;
    elseif con == 'N'
        break;
    else
        disp('Wrong input');
    end
    con = input('continue? [Y/N]:','s');
end
