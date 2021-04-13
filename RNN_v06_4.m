function RNN_v06_4(varargin)
% RNN_v06.4 A recurrent neural network with certain training phase
% Ref: Susillo and Abbott, 2009
% This version sets up the basic flow of the program, with FORCE training
% It plots the activity of nGN and actual output z.
% Update: from v06.3, now with two pairs of pulse input function to control two output

% v01 by Emilio Salinas, January 2021
% Junda Zhu, 4-6-2021
% clear all
clf
%% parameters
para = varargin{1};
if length(para) ~= 8
    % network parameters
    nGN = 200;     % number of generator (recurrent) neurons
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
nplot = 4;
if nplot > nGN
    nplot = nGN;
end

numoutput = 2;% number of output
%% initialize arrays
x = gpuArray(2*rand(nGN,1,'single') - 1);
J = gpuArray(zeros(nGN,'single'));
J(randperm(round(length(J(:))),round(p_GG*length(J(:))))) = randn(round(p_GG*length(J(:))),1)*g/sqrt(p_GG*nGN); %recurrent weight matrix
JGz = gpuArray(2*rand(nGN,numoutput,'single')-1); %feedback weight matrix
I = gpuArray(zeros(2*numoutput,1,'single'));
JGi = gpuArray(zeros(nGN,2*numoutput,'single'));
JGi(randperm(length(JGi(:)),p_GG*length(JGi(:)))) = randn(p_GG*length(JGi(:)),1);
W = gpuArray(randn(nGN,numoutput,'single')/sqrt(p_z*nGN)); %output weight vector
P = gpuArray(eye(nGN,'single')/alpha); %update matrix
z = gpuArray(zeros(numoutput,1,'single')); %output
eneg = gpuArray(zeros(numoutput,1,'single'));

% set space for data to be plotted
nTtrain = Ttrain/dt;
tplot = NaN(1, nTtrain);
% xplot = NaN(nplot, nTtrain);
Hplot = NaN(nplot, nTtrain);
zplot = NaN(numoutput, nTtrain);
eplot = NaN(1, nTtrain);

%% before training
T_start = 2001;
T_end = T_start + nTtrain -1;
t=0;

wid = 20; % 20 ms
d = rand(6,1)*length(T_start:T_end);
pul = rectpuls(1:length(T_start:T_end),wid);
I = zeros(2*numoutput,length(1:T_start-1));
f = zeros(numoutput,length(1:T_start-1)) -1;

for i=1:T_start
    H = tanh(x); % firing rates
    z = W' * H; % output
    dw = - P * H * eneg'; %dw
    dxdt = (-x + J*H) / tau;
    x = x + dxdt*dt;
    t = t + dt;
    
    % save some data for plotting
    tplot(i) = t;
    Hplot(:,i) = gather(H(1:nplot));
    zplot(:,i) = gather(z);
    dwplot(i) = norm(dw);
end
%% training
con = 1;
while con
    disp('Training Start');
    tic
    % precompute target and input function
    wid = 20; % 20 ms
    d = rand(40,1)*length(T_start:T_end);
    pul = rectpuls(1:length(T_start:T_end),wid);
    I(1,T_start:T_end) = pulstran(1:length(T_start:T_end),d(1:10),pul)';
    I(2,T_start:T_end) = pulstran(1:length(T_start:T_end),d(11:20),pul)';
    I(3,T_start:T_end) = pulstran(1:length(T_start:T_end),d(21:30),pul)';
    I(4,T_start:T_end) = pulstran(1:length(T_start:T_end),d(31:40),pul)';
    f = [f zeros(numoutput,length(T_start:T_end))] -1;
    
    % Main loop
    dwplot(T_start) = 0;
    for i=T_start:T_end
        f(:,i) = f(:,i-1);
        if I(1,i)
            f(1,i)=1;
            disp(['Training... ' num2str(i) ' / ' num2str(T_end)]);
        end
        if I(2,i)
            f(1,i)=-1;
        end
        if I(3,i)
            f(2,i)=1;
            disp(['Training... ' num2str(i) ' / ' num2str(T_end)]);
        end
        if I(4,i)
            f(2,i)=-1;
        end
        
        H = tanh(x); % firing rates
        PH = P*H;
        P = P - PH*PH'/(1+H'*PH); % update P
        eneg = z - f(:,i); % error
        dw = - P * H * eneg';
        W = W + dw; % update W
        J = J + repmat(dw(:,1)', nGN, 1) + repmat(dw(:,2)', nGN, 1); %update J (recurrent)
        z = W' * H; % output
        epos = z - f(:,i); % error after update
        dxdt = (-x + J*H + JGi*I(:,i)) / tau;
        x = x + dxdt*dt;
        t = t + dt;
        
        % save some data for plotting
        tplot(i) = t;
        Hplot(:,i) = gather(H(1:nplot));
        zplot(:,i) = gather(z);
        eplot(i) = gather(mean(epos - eneg));
        dwplot(i) = norm(dw);
    end
    disp('Training finished');
    toc
    % testing
    wid = 20; % 20 ms
    d = rand(12,1)*length(T_end+1:T_end+Ttrain);
    pul = rectpuls(1:length(T_end+1:T_end+Ttrain),wid);
    I(1,T_end+1:T_end+Ttrain) = pulstran(1:length(T_end+1:T_end+Ttrain),d(1:3),pul)';
    I(2,T_end+1:T_end+Ttrain) = pulstran(1:length(T_end+1:T_end+Ttrain),d(4:6),pul)';
    I(3,T_end+1:T_end+Ttrain) = pulstran(1:length(T_end+1:T_end+Ttrain),d(7:9),pul)';
    I(4,T_end+1:T_end+Ttrain) = pulstran(1:length(T_end+1:T_end+Ttrain),d(10:12),pul)';
    f = [f zeros(numoutput,length(T_end+1:T_end+Ttrain))] -1;
    
    dwplot(T_end+1:T_end+Ttrain) = 0;
    for i = T_end+1:T_end+Ttrain
        f(:,i) = f(:,i-1);
        if I(1,i)
            f(1,i)=1;
            disp(['Testing... ' num2str(i) 'ms / ' num2str(T_end+Ttrain) 'ms']);
        end
        if I(2,i)
            f(1,i)=-1;
        end
        if I(3,i)
            f(2,i)=1;
            disp(['Testing... ' num2str(i) 'ms / ' num2str(T_end+Ttrain) 'ms']);
        end
        if I(4,i)
            f(2,i)=-1;
        end
        
        H = tanh(x); % firing rates
        eneg = z - f(:,i);
        z = W' * H; % output
        epos = z - f(:,i);
        dxdt = (-x + J*H + JGi*I(:,i)) / tau;
        x = x + dxdt*dt;
        t = t + dt;
        
        % save some data for plotting
        tplot(i) = t;
        Hplot(:,i) = gather(H(1:nplot));
        zplot(:,i) = gather(z);
        eplot(i) = gather(mean(epos - eneg));
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
    %     ylim([-.2 1.2])
    %     set(gca, 'YTick', [-1, 0, 1])
    patch([T_start T_start T_end T_end],[-1.2, 6.2, 6.2, -1.2],'r', 'FaceAlpha',0.1,'EdgeAlpha',0.1);
    plot(tplot(T_start:T_end+Ttrain), f(1,T_start:T_end+Ttrain)+5, '-', 'color', clrF, 'LineWidth', 2);
    plot(tplot(T_start:T_end+Ttrain), zplot(1,T_start:T_end+Ttrain)+5, '-', 'color', clrOut, 'LineWidth', 2);
    plot(tplot(T_start:T_end+Ttrain), I(1,T_start:T_end+Ttrain)+2, '-r', 'LineWidth', 1);
    plot(tplot(T_start:T_end+Ttrain), I(2,T_start:T_end+Ttrain), '-b', 'LineWidth', 1);
    ylabel('Output 1');
    xlabel('Time (ms)');
    title(['RNN v06: ' num2str(nGN) ' neurons, with input']);
    
    subplot(4,1,2)
    hold on
    %     xlim([0 T_end+Ttrain])
    %     ylim([-1.2 1.2])
    %         set(gca, 'YTick', [0])
    patch([T_start T_start T_end T_end],[-1.2, 6.2, 6.2, -1.2],'r', 'FaceAlpha',0.1,'EdgeAlpha',0.1);
    plot(tplot(T_start:T_end+Ttrain), f(2,T_start:T_end+Ttrain)+5, '-', 'color', clrF, 'LineWidth', 2);
    plot(tplot(T_start:T_end+Ttrain), zplot(2,T_start:T_end+Ttrain)+5, '-', 'color', clrOut, 'LineWidth', 2);
    plot(tplot(T_start:T_end+Ttrain), I(3,T_start:T_end+Ttrain)+2, '-r', 'LineWidth', 1);
    plot(tplot(T_start:T_end+Ttrain), I(4,T_start:T_end+Ttrain), '-b', 'LineWidth', 1);
    ylabel('Output 2');
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
        plot(tplot(T_start:T_end+Ttrain), Hplot(ii,T_start:T_end+Ttrain)*sfac + yoff, '-', 'color', clrGN, 'LineWidth', 0.5);
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
