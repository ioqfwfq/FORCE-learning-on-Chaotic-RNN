function [maerr, rmserr, Wabs] = run_auto_v06
%% p_GG is the vari
clc
clear all
pGG_temp = 0.1:0.1:1;
for g = 1:length(pGG_temp)
    param = [500 10 8000 1 1.5 pGG_temp(g)];
    tic
    for c = 1:10
        [maerr_w(g, c),rmserr_w(g, c),Wabs_w(g,c)] = RNN_v04_6(param);
        [maerr_r(g, c),rmserr_r(g, c),Wabs_r(g,c)] = RNN_v05_3(param);
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
plot(pGG_temp,maerrplot1,'.-', 'color', 'k');
plot(pGG_temp,maerrplot2,'.-', 'color', 'r');
ylabel('MAE');
xlabel('p_GG');
subplot 312
plot(pGG_temp,rmserrplot1,'.-', 'color', 'k');
plot(pGG_temp,rmserrplot2,'.-', 'color', 'r');
ylabel('RMS');
xlabel('p_GG');
subplot 313
plot(pGG_temp,Wabsplot1,'.-', 'color', 'k');
plot(pGG_temp,Wabsplot2,'.-', 'color', 'r');
ylabel('|W|');
xlabel('p_GG');
%% save files
diffp.parameter.nGN = param(1);
diffp.parameter.tau = param(2);
diffp.parameter.training_time = param(3);
diffp.parameter.dt = param(4);
diffp.parameter.g = param(5);
diffp.parameter.p_GG = pGG_temp;
diffp.code.FORCEonJ = 'RNN_v05_3.m';
diffp.code.FORCEonW = 'RNN_v04_6.m';
diffp.maerr_w = maerr_w;
diffp.rmserr_w = rmserr_w;
diffp.Wabs_w = Wabs_w;
diffp.maerr_r = maerr_r;
diffp.rmserr_r = rmserr_r;
diffp.Wabs_r = Wabs_r;
save('diffp_GG.mat','-struct','diffp');
savefig('diffp_GG.fig');
clear all
