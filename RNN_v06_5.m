function RNN_v06_5(varargin)
% RNN_v06.5 A recurrent neural network with certain training phase
% Ref: Susillo and Abbott, 2009
% This version sets up the basic flow of the program, with FORCE training
% only on Wout!
% It plots the activity of nGN and actual output z.
% Update: from v06.4, now with any pairs of pulse input function to control any output

% v01 by Emilio Salinas, January 2021
% Junda Zhu, 4-12-2021
% clear all
clf
%% parameters
para = varargin{1};
if length(para) ~= 8
    % network parameters
    nGN = 1200;     % number of generator (recurrent) neurons
    tau = 10;    % membrane time constant, in ms
    p_GG = 0.8; % p of non zero recurrence
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
nplot = 8;
if nplot > nGN
    nplot = nGN;
end

numoutput = 8;% number of output
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
z = gpuArray(zeros(numoutput,1,'single')) -1; %output
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
    dxdt = (-x + J*H + JGz*z) / tau;
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
    % target and input function
    wid = 20; % 20 ms
    
    for ij = 1:numoutput
        d = rand(length(T_start:T_end)/500,1)*length(T_start:T_end); % random time points for the inputs
        pul = rectpuls(1:length(T_start:T_end),wid); % pulses
        I(2*ij-1,T_start:T_end) = pulstran(1:length(T_start:T_end),d(1:length(d)/2-1),pul)'; % On
        I(2*ij,T_start:T_end) = pulstran(1:length(T_start:T_end),d(length(d):-1:length(d)/2),pul)'; % Off
        
    end
    f = [f zeros(numoutput,length(T_start:T_end))-1]; % target function at -1
    % Main loop
    dwplot(T_start) = 0;
    for i=T_start:T_end
        f(:,i) = f(:,i-1);
        for ij = 1:numoutput
            if I(2*ij-1,i)
                f(ij,i) = 1;
                disp(['Training... ' num2str(i) 'ms / ' num2str(T_end) 'ms']);
            end
            if I(2*ij,i)
                f(ij,i) = -1;
            end
        end
        
        H = tanh(x); % firing rates
        PH = P*H;
        P = P - PH*PH'/(1+H'*PH); % update P
        eneg = z - f(:,i); % error
        dw = - P * H * eneg';
        W = W + dw; % update W
        %         J = J + repmat(dw(:,1)', nGN, 1) + repmat(dw(:,2)', nGN, 1); %update J (recurrent)
        z = W' * H; % output
        epos = z - f(:,i); % error after update
        dxdt = (-x + J*H + JGz*z + JGi*I(:,i)) / tau;
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
    for ij = 1:numoutput
        d = rand(length(T_end+1:T_end+Ttrain)/500,1)*length(T_end+1:T_end+Ttrain); % random time points for the inputs
        pul = rectpuls(1:length(T_end+1:T_end+Ttrain),wid); % pulses
        I(2*ij-1,T_end+1:T_end+Ttrain) = pulstran(1:length(T_end+1:T_end+Ttrain),d(1:length(d)/2-1),pul)'; % On
        I(2*ij,T_end+1:T_end+Ttrain) = pulstran(1:length(T_end+1:T_end+Ttrain),d(length(d):-1:length(d)/2),pul)'; % Off
    end
    f = [f zeros(numoutput,length(T_end+1:T_end+Ttrain))-1]; % target function at -1
    dwplot(T_end+1:T_end+Ttrain) = 0;
    
    for i = T_end+1:T_end+Ttrain
        f(:,i) = f(:,i-1);
        for ij = 1:numoutput
            if I(2*ij-1,i)
                f(ij,i) = 1;
                disp(['Testing... ' num2str(i) 'ms / ' num2str(T_end+Ttrain) 'ms']);
            end
            if I(2*ij,i)
                f(ij,i) = -1;
            end
        end
        
        H = tanh(x); % firing rates
        eneg = z - f(:,i);
        z = W' * H; % output
        epos = z - f(:,i);
        dxdt = (-x + J*H + JGz*z + JGi*I(:,i)) / tau;
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
    sfac = 0.5;% scale factor for plotting activity one neuron per row
    
    f1 = figure('Name', 'Results');
    title(['RNN v06: ' num2str(nGN) ' neurons, with input']);
    for ij = 1:numoutput
        subplot(2+numoutput,1,ij)
        ylim([-1.2, 7.2])
        hold on
        patch([T_start T_start T_end T_end],[-1.2, 7.2, 7.2, -1.2],'r', 'FaceAlpha',0.1,'EdgeAlpha',0.1);
        plot(tplot(T_start:T_end+Ttrain), I(2*ij-1,T_start:T_end+Ttrain)+2, '-m', 'LineWidth', 1);
        plot(tplot(T_start:T_end+Ttrain), I(2*ij,T_start:T_end+Ttrain), '-b', 'LineWidth', 1);
        plot(tplot(T_start:T_end+Ttrain), f(ij,T_start:T_end+Ttrain)+5, '-', 'color', clrF, 'LineWidth', 2);
        plot(tplot(T_start:T_end+Ttrain), zplot(ij,T_start:T_end+Ttrain)+5, '-', 'color', clrOut, 'LineWidth', 2);
        
        ylabel(['Output ' num2str(ij)]);
        xlabel('Time (ms)');
    end
    
    subplot(2+numoutput,1,[2+numoutput-1 2+numoutput])
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
    
    f2 = figure('Name', 'Performance');
    subplot(2+numoutput,1,2+numoutput)
    hold on
    %     xlim([0 T_end+Ttrain])
    ylim([-0.15 0.15])
    set(gca, 'YTick', [-0.1, 0, 0.1])
    line([T_start T_start],[-1 1])
    line([T_end T_end],[-1 1])
    patch([T_start T_start T_end T_end],[-1, 1, 1, -1],'r', 'FaceAlpha',0.1,'EdgeAlpha',0.1)
    plot(tplot(T_start:T_end+Ttrain), eplot(T_start:T_end+Ttrain), '.','MarkerSize',2, 'color', 'k');
    plot(tplot(T_start:T_end+Ttrain), dwplot(T_start:T_end+Ttrain)/max(dwplot)*0.3-0.15, '.','MarkerSize',3, 'color', 'c');
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
