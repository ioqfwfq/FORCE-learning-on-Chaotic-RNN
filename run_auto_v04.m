function [maerr, rmserr, Wabs] = run_auto_v04
%% g is the vari
clc
clear all
g_temp = 0.8:0.1:1.5;
for g = 1:length(g_temp)
    param = [1000 10 12000 1 g_temp(g)];
    tic
    for c = 1:2
        [maerr(g, c),rmserr(g, c),Wabs(g,c)] = RNN_v04_5(param);
    end
    toc
    disp(['Group ' num2str(g) ' finished']);
end
%% visualization
maerrplot = mean(maerr,2);
rmserrplot = mean(rmserr,2);
Wabsplot = mean(Wabs,2);

subplot 311
hold on
plot(g_temp,maerrplot,'.-', 'color', 'k');
ylabel('MAE');
xlabel('g');
subplot 312
plot(g_temp,rmserrplot,'.', 'color', 'r');
ylabel('RMS');
xlabel('g');
subplot 313
plot(g_temp,Wabsplot,'.', 'color', 'r');
ylabel('|W|');
xlabel('g');