%RNN_v0.1 Initial skeleton for running a recurrent neural network
%
% This version just sets up the basic flow of the program, without any
% actual recurrent interactions. It just plots the activity of nGN
% ``generator neurons'' whose activites fall exponentially over time.
%
% Here the plotting is all done at the end of the time loop, which is
% much more efficient.

% Emilio Salinas, January 2021

% 
% network parameters
%
nGN = 10;     % number of generator (recurrent) neurons
tau = 10;    % membrane time constant, in ms

% 
% run parameters
%
Tmax = 100;   % simulation time (in ms)
dt = 1;      % integration time step (in ms)
nplot = 5;   % number of generator neurons to plot;

if nplot > nGN
    nplot = nGN;
end

%
% initialize arrays
%
x = -1 + 2*rand(nGN,1);

% set space for data to be plotted
nTmax = Tmax/dt;
tplot = NaN(1, nTmax);
xplot = NaN(nplot, nTmax);

%---------------
% loop over time
%---------------

t=0;
for j=1:nTmax
    t = t + dt;
    % tau dx/dt = -x
    dxdt = -x/tau;
    x = x + dxdt*dt;
    
    % save some data for plotting 
    tplot(j) = t;
    xplot(:,j) = x(1:nplot);
end

%
% graph the results
%
clrGN = 'c';
clr_grid = 0.5*[1 1 1];

% scale factor for plotting activity one neuron per row
sfac = 0.5;     

clf
hold on
xlim([0 Tmax+1])
ylim([0.25 nplot+0.75])
set(gca, 'YTick', [1:nplot])

for j=1:nplot
    yoff = (j-1) + 1;
    plot(xlim, yoff*[1 1], ':', 'color', clr_grid)
    plot(tplot, xplot(j,:)*sfac + yoff, '-', 'color', clrGN);
end
ylabel('Recurrent neuron')
xlabel('Time (ms)')
title(['RNN v01: ' num2str(nGN) ' neurons, no recurrence'])



