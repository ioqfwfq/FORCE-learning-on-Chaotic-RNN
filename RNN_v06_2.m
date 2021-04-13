function RNN_v06_2(varargin)
% RNN_v06.2 A recurrent neural network with certain training phase
% Ref: Susillo and Abbott, 2009
% This version sets up the basic flow of the program, with FORCE training
% It plots the activity of nGN and actual output z.
% run by run_auto.m
% Update: from v06.1, noisy input function to control decision-like output

% v01 by Emilio Salinas, January 2021
% Junda Zhu, 3-29-2021
tic
% clear all
clf
%% parameters
para = varargin{1};
if length(para) ~= 8
    % network parameters
    nGN = 1200;     % number of generator (recurrent) neurons
    tau = 10;    % membrane time constant, in ms
    p_GG = 1; % p of non zero recurrence
    p_z = 1; % p of non zero output
    alpha = 1;
    g = 1.5;
    % run parameters
    Ttrain = 10000;   % training time (in ms)
    dt = 1;      % integration time step (in ms)
    
else % parameters given by user input
    nGN = para(1);
    tau = para(2);
    p_GG = para(3);
    p_z = para(4); % p of non zero output
    alpha = para(5);
    g = para(6);
    Ttrain = para(7);
    dt = para(8);
end
nplot = 5;
if nplot > nGN
    nplot = nGN;
end

numinput = 1;% number of input
whichfunc = 2; % which target function used (1-4)
%% initialize arrays
x = gpuArray(2*rand(nGN,1,'single') - 1);
H = tanh(x);
J = gpuArray(zeros(nGN,'single'));
J(randperm(round(length(J(:))),round(p_GG*length(J(:))))) = randn(round(p_GG*length(J(:))),1)*g/sqrt(p_GG*nGN); %recurrent weight matrix
JGz = gpuArray(2*rand(nGN,1,'single')-1); %feedback weight matrix
I = gpuArray(zeros(numinput,1,'single'));
JGi = gpuArray(zeros(nGN,length(I),'single'));
% JGi(randperm(size(JGi,1),length(I))) = randn(length(I));
JGi(randperm(size(JGi,1),p_GG*size(JGi,1))) = randn(p_GG*size(JGi,1),1);
W = gpuArray(randn(nGN,1,'single')/sqrt(p_z*nGN)); %output weight vector
P = gpuArray(eye(nGN,'single')/alpha); %update matrix
z = 0; %output
f = 0; %target
eneg = 0;

% set space for data to be plotted
nTtrain = Ttrain/dt;
tplot = NaN(1, nTtrain);
% xplot = NaN(nplot, nTtrain);
Hplot = NaN(nplot, nTtrain);
zplot = NaN(1, nTtrain);
eplot = NaN(1, nTtrain);

% Target function
switch whichfunc
    case 1 % triangular wave of period 600 ms
        peri = 600;
        func = @(t,peri)(2*triangle(2*pi*(1/peri)*t)-1);
    case 2 % periodic function of period 1200 ms
        peri = 1200*rand(1)+600;
        func = @(t,peri)1/2*(sin(1.0*2*pi*(1/peri)*t) + ...
            1/4*sin(2.0*2*pi*(1/peri)*t) + ...
            1/12*sin(3.0*2*pi*(1/peri)*t) + ...
            1/6*sin(4.0*2*pi*(1/peri)*t));
    case 3 % square wave of period 600 ms
        peri = 600;
        func = @(t,peri)(2*(sin(t/peri*2*pi)>0)-1);
    case 4 % sine wave of period 60 ms or 8000 ms
        peri = 800*rand(1)+800;
        func = @(t,peri)(sin(t/peri*2*pi));
end


%% before training
T_start = 2001;
T_end = T_start + nTtrain -1;
t=0;

I(1:T_start-1) = func(1:T_start-1,peri);
f = zeros(size(I));
f(I<=0) = -1;
f(I>0) = 1;

for i=1:T_start
    H = tanh(x); % firing rates
    z = W' * H; % output
    dw = eneg * P * H; %dw
    dxdt = (-x + J*H + JGz*z) / tau;
    x = x + dxdt*dt;
    t = t + dt;
    
    % save some data for plotting
    tplot(i) = t;
    Hplot(:,i) = gather(H(1:nplot));
    zplot(i) = gather(z);
    dwplot(i) = norm(dw);
end
toc
%% training
con = 1;
while con
    % precompute target and input function
    peri = 1200*rand(1)+600;
    I(T_start:T_end) = func(T_start:T_end,peri);
    f = zeros(size(I));
    f(I<=0) = -0.5;
    f(I>0) = 0.5;
    
    % Main loop
    dwplot(T_start) = 0;
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
        dxdt = (-x + J*H + JGz*z + JGi*I(i)) / tau;
        x = x + dxdt*dt;
        t = t + dt;
        
        % save some data for plotting
        tplot(i) = t;
        Hplot(:,i) = gather(H(1:nplot));
        zplot(i) = gather(z);
        eplot(i) = gather(epos - eneg);
        dwplot(i) = norm(dw);
    end
    toc
    % testing
    peri = 1200*rand(1)+600;
    I(T_end+1:T_end+Ttrain) = func(T_end+1:T_end+Ttrain,peri);
    f = zeros(size(I));
    f(I<=0) = 0;
    f(I>0) = 1;
    dwplot(T_end+1:T_end+Ttrain) = 0;
    for i = T_end+1:T_end+Ttrain
        H = tanh(x); % firing rates
        eneg = z - f(i);
        z = W' * H; % output
        epos = z - f(i);
        dxdt = (-x + J*H + JGz*z + JGi*I(i)) / tau;
        x = x + dxdt*dt;
        t = t + dt;
        
        % save some data for plotting
        tplot(i) = t;
        Hplot(:,i) = gather(H(1:nplot));
        zplot(i) = gather(z);
        eplot(i) = gather(epos - eneg);
    end
    toc
    
    %% plot
    disp('plotting');
    % graph the results
    clrGN = 'k';
    clrOut = 'r';
    clrF = 'g';
    clr_grid = 0.5*[1 1 1];
    % scale factor for plotting activity one neuron per row
    sfac = 0.5;
    subplot(4,1,1)
    hold on
    %     xlim([0 T_end+Ttrain])
    ylim([-.2 1.2])
    set(gca, 'YTick', [-1, 0, 1])
    patch([T_start T_start T_end T_end],[-1.2, 1.2, 1.2, -1.2],'r', 'FaceAlpha',0.1,'EdgeAlpha',0.1)
    plot(tplot(T_start:T_end+Ttrain), f(T_start:T_end+Ttrain), '-', 'color', clrF, 'LineWidth', 2);
    plot(tplot(T_start:T_end+Ttrain), zplot(T_start:T_end+Ttrain), '-', 'color', clrOut, 'LineWidth', 2);
    ylabel('Output Unit');
    xlabel('Time (ms)');
    title(['RNN v06: ' num2str(nGN) ' neurons, with input']);
    
    subplot(4,1,2)
    hold on
    %     xlim([0 T_end+Ttrain])
    ylim([-1.2 1.2])
    set(gca, 'YTick', [0, 1])
    patch([T_start T_start T_end T_end],[-1.2, 1.2, 1.2, -1.2],'r', 'FaceAlpha',0.1,'EdgeAlpha',0.1)
    plot(tplot(T_start:T_end+Ttrain), I(T_start:T_end+Ttrain), '-b', 'LineWidth', 2);
    ylabel('Input Unit');
    xlabel('Time (ms)');
    
    subplot(4,1,3)
    hold on
    %     xlim([0 T_end+Ttrain])
    ylim([0.25 nplot+0.75])
    set(gca, 'YTick', 1:nplot)
    patch([T_start T_start T_end T_end],[0.25, nplot+1, nplot+1, 0.25],'r', 'FaceAlpha',0.1,'EdgeAlpha',0.1)
    for ii=1:nplot
        yoff = (ii-1) + 1;
        plot(xlim, yoff*[1 1], ':', 'color', clr_grid)
        plot(tplot(T_start:T_end+Ttrain), Hplot(ii,T_start:T_end+Ttrain)*sfac + yoff, '-', 'color', clrGN, 'LineWidth', 1);
    end
    ylabel('Recurrent neurons');
    xlabel('Time (ms)');
    
    subplot(4,1,4)
    hold on
    %     xlim([0 T_end+Ttrain])
    ylim([-0.15 0.15])
    set(gca, 'YTick', [-0.1, 0, 0.1])
    line([T_start T_start],[-1 1])
    line([T_end T_end],[-1 1])
    patch([T_start T_start T_end T_end],[-1, 1, 1, -1],'r', 'FaceAlpha',0.1,'EdgeAlpha',0.1)
    plot(tplot(T_start:T_end+Ttrain), eplot(T_start:T_end+Ttrain), '.', 'color', 'k');
    plot(tplot(T_start:T_end+Ttrain), dwplot(T_start:T_end+Ttrain)/max(dwplot)*0.3-0.15, '.', 'color', 'c');
    ylabel('\delta Error');
    xlabel('Time (ms)');
    
    T_start = T_end+Ttrain+1;
    T_end = T_start + nTtrain -1;
    
    % Control
    con = input('continue training? [1/0]:', 's');
    while con ~= '1' && con ~= '0'
        disp('Wrong input');
        con = input('continue training? [1/0]:','s');
    end
    con = str2double(con);
end
