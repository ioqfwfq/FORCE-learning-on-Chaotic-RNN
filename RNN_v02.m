function RNN_v03_1(varargin)
% RNN_v03.1 A recurrent neural network without training
%
% This version sets up the basic flow of the program, with actual recurrent
% interactions and outputs. It plots the activity of nGN
% and actual output z.
% run by run_auto.m
% Update: added output z and target f.

% v01 by Emilio Salinas, January 2021
% Junda Zhu, 2-14-2021

% clear all
%%
para = varargin{1};
if length(para) ~= 5
    % network parameters
    nGN = 150;     % number of generator (recurrent) neurons
    tau = 10;    % membrane time constant, in ms
    % run parameters
    Tmax = 1000;   % simulation time (in ms)
    dt = 1;      % integration time step (in ms)
    nplot = 5;   % number of generator neurons to plot;
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
whichfunc = 1; % which target function used (1-5)
p_z = 
% initialize arrays
x = -1 + 2*rand(nGN,1);
g = 1.5;
J = randn(nGN)*g/sqrt(nGN); %recurrent weight matrix
W = randn(nGN,1)*g/sqrt(nGN); %output weight vector
z = 0;%output
f = 0;%target

% set space for data to be plotted
nTmax = Tmax/dt;
tplot = NaN(1, nTmax);
xplot = NaN(nplot, nTmax);
Hplot = NaN(nplot, nTmax);
zplot = NaN(1, nTmax);
fplot = NaN(1, nTmax);
%%
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
            z = W' * H; % output
            
            dxdt = -x/tau + J*H/tau; % tau dx/dt = -x + J*H(x)
            x = x + dxdt*dt;
            
            % save some data for plotting
            tplot(j) = t;
            xplot(:,j) = x(1:nplot);
            Hplot(:,j) = H(1:nplot);
            zplot(j) = z;
        end
        
        % target function
        switch whichfunc
            case 1
        f = sin(tplot*pi/3/tau); 
            case 2
                
            case 3
                
            case 4
                
            case 5
                fplot(j) = f;
        
        % graph the results        
        clrGN = 'k';
        clrOut = 'r';
        clrF = 'g';
        clr_grid = 0.5*[1 1 1];
        % scale factor for plotting activity one neuron per row
        sfac = 0.5;
       
        clf
        subplot(2,1,1)
        hold on
        xlim([T_start T_end+1])
        ylim([0.25 nplot+0.75])
        set(gca, 'YTick', [1:nplot])
        for j=1:nplot
            yoff = (j-1) + 1;
            plot(xlim, yoff*[1 1], ':', 'color', clr_grid)
            %     plot(tplot, xplot(j,:)*sfac + yoff, '-', 'color', clrGN);
            plot(tplot, Hplot(j,:)*sfac + yoff, '-', 'color', clrGN, 'LineWidth', 1.5);
        end
        ylabel('Recurrent neuron');
        xlabel('Time (ms)');
        title(['RNN v03: ' num2str(nGN) ' neurons, with recurrence']);
        
        subplot(2,1,2)
        hold on
        xlim([T_start T_end+1])
        ylim([-1 1])
        set(gca, 'YTick', 0)
        plot(tplot, zplot*sfac, '-', 'color', clrOut, 'LineWidth', 2);
        plot(tplot, fplot*sfac, '-', 'color', clrF, 'LineWidth', 2);
        ylabel('Output Unit');
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
