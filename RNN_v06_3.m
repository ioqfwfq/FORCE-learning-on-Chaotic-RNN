function RNN_v06_3(varargin)
% RNN_v06.3 A recurrent neural network with certain training phase
% Ref: Susillo and Abbott, 2009
% This version sets up the basic flow of the program, with FORCE training
% It plots the activity of nGN and actual output z.
% Update: from v06.2, two pulse input function to control one output

% v01 by Emilio Salinas, January 2021
% Junda Zhu, 4-4-2021
tic
% clear all
clf
%% parameters
para = varargin{1};
if length(para) ~= 8
    % network parameters
    nGN = 200;     % number of generator (recurrent) neurons
    tau = 10;    % membrane time constant, in ms
    p_GG = 0.1; % p of non zero recurrence
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

numinput = 2;% number of input
%% initialize arrays
x = gpuArray(2*rand(nGN,1,'single') - 1);
J = gpuArray(zeros(nGN,'single'));
J(randperm(round(length(J(:))),round(p_GG*length(J(:))))) = randn(round(p_GG*length(J(:))),1)*g/sqrt(p_GG*nGN); %recurrent weight matrix
JGz = gpuArray(2*rand(nGN,1,'single')-1); %feedback weight matrix
I = gpuArray(zeros(numinput,1,'single'));
JGi = gpuArray(zeros(nGN,numinput,'single'));
JGi(randperm(length(JGi(:)),p_GG*length(JGi(:)))) = randn(p_GG*length(JGi(:)),1);
W = gpuArray(randn(nGN,1,'single')/sqrt(p_z*nGN)); %output weight vector
P = gpuArray(eye(nGN,'single')/alpha); %update matrix
z = 0; %output
eneg = 0;

% set space for data to be plotted
nTtrain = Ttrain/dt;
tplot = NaN(1, nTtrain);
% xplot = NaN(nplot, nTtrain);
Hplot = NaN(nplot, nTtrain);
zplot = NaN(1, nTtrain);
eplot = NaN(1, nTtrain);

%% before training
T_start = 2001;
T_end = T_start + nTtrain -1;
t=0;

wid = 20; % 20 ms
d = rand(6,1)*length(T_start:T_end);
pul = rectpuls(1:length(T_start:T_end),wid);
I = zeros(numinput,length(1:T_start-1));
f(1:T_start-1) = -0.5;

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
    tic
    % precompute target and input function
    wid = 20; % 20 ms
    d = rand(10,1)*length(T_start:T_end);
    pul = rectpuls(1:length(T_start:T_end),wid);
    I(1,T_start:T_end) = pulstran(1:length(T_start:T_end),d(1:5),pul)';
    I(2,T_start:T_end) = pulstran(1:length(T_start:T_end),d(6:10),pul)';
    f(T_start:T_end) = 0;
    
    % Main loop
    dwplot(T_start) = 0;
    for i=T_start:T_end
        f(i) = f(i-1);
        if I(1,i)
            f(i)=1;
        end
        if I(2,i)
            f(i)=0;
        end
        
        H = tanh(x); % firing rates
        PH = P*H;
        P = P - PH*PH'/(1+H'*PH); % update P
        eneg = z - f(i); % error
        dw = - eneg * P * H;
        W = W + dw; % update W
        J = J + repmat(dw', nGN, 1); %update J (recurrent)
        z = W' * H; % output
        epos = z - f(i); % error after update
        dxdt = (-x + J*H + JGz*z + JGi*I(:,i)) / tau;
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
    wid = 20; % 20 ms
    d = rand(6,1)*length(T_end+1:T_end+Ttrain);
    pul = rectpuls(1:length(T_end+1:T_end+Ttrain),wid);
    I(1,T_end+1:T_end+Ttrain) = pulstran(1:length(T_end+1:T_end+Ttrain),d(1:3),pul)';
    I(2,T_end+1:T_end+Ttrain) = pulstran(1:length(T_end+1:T_end+Ttrain),d(4:6),pul)';
    f(T_end+1:T_end+Ttrain) = 0;
    
    dwplot(T_end+1:T_end+Ttrain) = 0;
    for i = T_end+1:T_end+Ttrain
        f(i) = f(i-1);
        if I(1,i)
            f(i)=1;
        end
        if I(2,i)
            f(i)=0;
        end
        
        H = tanh(x); % firing rates
        eneg = z - f(i);
        z = W' * H; % output
        epos = z - f(i);
        dxdt = (-x + J*H + JGz*z + JGi*I(:,i)) / tau;
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
    plot(tplot(T_start:T_end+Ttrain), I(1,T_start:T_end+Ttrain), '-r', 'LineWidth', 2);
    plot(tplot(T_start:T_end+Ttrain), I(2,T_start:T_end+Ttrain), '-b', 'LineWidth', 2);
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
