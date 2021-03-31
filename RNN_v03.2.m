function RNN_v03_2(varargin)
% RNN_v03.2 A recurrent neural network without training
%
% This version sets up the basic flow of the program, with actual recurrent
% interactions and outputs. It plots the activity of nGN
% and actual output z.
% run by run_auto.m
% Update: added 4 variation of target f and feedback from output z.

% v01 by Emilio Salinas, January 2021
% Junda Zhu, 2-14-2021

% clear all
%%
para = varargin{1};
if length(para) ~= 5
    % network parameters
    nGN = 1000;     % number of generator (recurrent) neurons
    tau = 10;    % membrane time constant, in ms
    % run parameters
    Tmax = 3000;   % simulation time (in ms)
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
whichfunc = 2; % which target function used (1-4)
p_z = 1; % p of non zero output
p_GG = 0.1; % p of non zero recurrence
% initialize arrays
x = 2*rand(nGN,1) - 1;
H = tanh(x);
g = 1.5;
J = zeros(nGN);
J(randperm(length(J(:)),p_GG*length(J(:)))) = randn(p_GG*length(J(:)),1)*g/sqrt(p_GG*nGN); %recurrent weight matrix
JGz = 2*rand(nGN,1)-1; %feedback weight matrix
W = randn(nGN,1)/sqrt(p_z*nGN); %output weight vector
z = 0;%output
f = 0;%target

% set space for data to be plotted
nTmax = Tmax/dt;
tplot = NaN(1, nTmax);
xplot = NaN(nplot, nTmax);
Hplot = NaN(nplot, nTmax);
zplot = NaN(1, nTmax);
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
        [tplot, xplot, Hplot, zplot, T_start, T_end, t, z, x, H, J, JGz, W] = RNN_main(tplot, xplot, Hplot, zplot, T_start, T_end, t, z, dt, tau, x, H, J, JGz, W, nplot);
        
        % target function
        switch whichfunc
            case 1 % triangular wave of period 600 ms
                peri = 600;
                f = 2*triangle(2*pi*(1/peri)*tplot)-1;
            case 2 % periodic function of period 1200 ms
                peri = 1200;
                f = sin(1.0*2*pi*(1/peri)*tplot) + ...
                    1/2*sin(2.0*2*pi*(1/peri)*tplot) + ...
                    1/6*sin(3.0*2*pi*(1/peri)*tplot) + ...
                    1/3*sin(4.0*2*pi*(1/peri)*tplot);
                f = f/((max(f)-min(f))/2);
            case 3 % square wave of period 600 ms
                peri = 600;
                f = 2*(sin(tplot/peri*2*pi)>0)-1;
            case 4 % sine wave of period 60 ms or 8000 ms
                peri = 6*tau;
                f = sin(tplot/peri*2*pi);
        end
        fplot = f;
        
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
        for ii=1:nplot
            yoff = (ii-1) + 1;
            plot(xlim, yoff*[1 1], ':', 'color', clr_grid)
            %     plot(tplot, xplot(j,:)*sfac + yoff, '-', 'color', clrGN);
            plot(tplot, Hplot(ii,:)*sfac + yoff, '-', 'color', clrGN, 'LineWidth', 1);
        end
        ylabel('Recurrent neuron');
        xlabel('Time (ms)');
        title(['RNN v03: ' num2str(nGN) ' neurons, with recurrence']);
        
        subplot(2,1,2)
        hold on
        xlim([T_start T_end+1])
        ylim([-1.2 1.2])
        set(gca, 'YTick', [-1, 0, 1])
        plot(tplot, zplot, '-', 'color', clrOut, 'LineWidth', 2);
        plot(tplot, fplot, '-', 'color', clrF, 'LineWidth', 2);
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
end
%% Main function
function [tplot, xplot, Hplot, zplot, T_start, T_end, t, z, x, H, J, JGz, W] = RNN_main(tplot, xplot, Hplot, zplot, T_start, T_end, t, z, dt, tau, x, H, J, JGz, W, nplot)
for i=T_start:T_end
    t = t + dt;
    H = tanh(x); % firing rates H(x)
    z = W' * H; % output
    
    dxdt = -x/tau + J*H/tau + JGz*z/tau;
    x = x + dxdt*dt;
    
    % save some data for plotting
    tplot(i) = t;
    xplot(:,i) = x(1:nplot);
    Hplot(:,i) = H(1:nplot);
    zplot(i) = z;
end
end
