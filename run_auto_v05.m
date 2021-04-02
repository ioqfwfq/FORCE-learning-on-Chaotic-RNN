function run_auto_v05
%% g is the vari
clc
clear all
g_temp = 0.8:0.1:0.8;
for g = 1:length(g_temp)
    param = [100 10 800 1 g_temp(g)];
    tic
    for c = 1:1
        [maerr_w(g, c),rmserr_w(g, c),Wabs_w(g,c)] = RNN_v04_5(param);
        [maerr_r(g, c),rmserr_r(g, c),Wabs_r(g,c)] = RNN_v05_2(param);
    end
    toc
    disp(['Group ' num2str(g) ' finished']);
end
%% visualization
clf
maerrplot1 = mean(maerr_w,2);
rmserrplot1 = mean(rmserr_w,2);
Wabsplot1 = mean(Wabs_w,2);
maerrplot2 = mean(maerr_r,2);
rmserrplot2 = mean(rmserr_r,2);
Wabsplot2 = mean(Wabs_r,2);

subplot 311
hold on
plot(g_temp,maerrplot1,'.-', 'color', 'k');
plot(g_temp,maerrplot2,'.-', 'color', 'r');
ylabel('MAE');
xlabel('g');
subplot 312
hold on
plot(g_temp,rmserrplot1,'.-', 'color', 'k');
plot(g_temp,rmserrplot2,'.-', 'color', 'r');
ylabel('RMS');
xlabel('g');
subplot 313
hold on
plot(g_temp,Wabsplot1,'.-', 'color', 'k');
plot(g_temp,Wabsplot2,'.-', 'color', 'r');
ylabel('|W|');
xlabel('g');
%% save files
diffg.parameter.nGN = param(1);
diffg.parameter.tau = param(2);
diffg.parameter.training_time = param(3);
diffg.parameter.dt = param(4);
diffg.parameter.g = g_temp;
diffg.parameter.p_GG = 0.1;
diffg.code.FORCEonJ = 'RNN_v05_2.m';
diffg.code.FORCEonW = 'RNN_v04_5.m';
diffg.maerr_w = maerr_w;
diffg.rmserr_w = rmserr_w;
diffg.Wabs_w = Wabs_w;
diffg.maerr_r = maerr_r;
diffg.rmserr_r = rmserr_r;
diffg.Wabs_r = Wabs_r;
save('diffgvalue.mat','-struct','diffg');
savefig('diffgvalue.fig');
clear all
