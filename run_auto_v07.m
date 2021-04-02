function [] = run_auto_v07(vari)
switch vari
    case 1 %p_GG
        %% p_GG is the vari
        clear all
        clc
        vari_temp = 0.1:0.1:1;
        for g = 1:length(vari_temp)
            param = [500, 10, vari_temp(g), 1, 1, 1.5, 8000, 1]; % [nGN tau p_GG p_z alpha g Tmax dt]
            tic
            for c = 1:16
                [maerr_w(g, c),rmserr_w(g, c),Wabs_w(g,c)] = RNN_v04_7(param);
                [maerr_r(g, c),rmserr_r(g, c),Wabs_r(g,c)] = RNN_v05_4(param);
            end
            toc
            disp(['Group ' num2str(g) ' finished']);
        end
    case 2 %g
        %% g is the vari
        clc
        vari_temp = 0.8:0.1:1.5;
        for g = 1:length(vari_temp)
            param = [500, 10, 0.1, 1, 1, vari_temp(g), 8000, 1]; % [nGN tau p_GG p_z alpha g Tmax dt]
            tic
            for c = 1:16
                [maerr_w(g, c),rmserr_w(g, c),Wabs_w(g, c)] = RNN_v04_7(param);
                [maerr_r(g, c),rmserr_r(g, c),Wabs_r(g, c)] = RNN_v05_4(param);
            end
            toc
            disp(['Group ' num2str(g) ' finished']);
        end
    case 3 % p_z
        %% p_z is the vari
        clc
        vari_temp = 0.2:0.2:1;
        for g = 1:length(vari_temp)
            param = [500, 10, 0.1, vari_temp(g), 1, 1.5, 8000, 1]; % [nGN tau p_GG p_z alpha g Tmax dt]
            tic
            for c = 1:20
                [maerr_w(g, c),rmserr_w(g, c),Wabs_w(g, c)] = RNN_v04_7(param);
                [maerr_r(g, c),rmserr_r(g, c),Wabs_r(g, c)] = RNN_v05_4(param);
            end
            toc
            disp(['Group ' num2str(g) ' finished']);
        end
    case 4 % training time Tmax
        %% Tmax is the vari
        clc
        vari_temp = 0:2000:24000;
        for g = 1:length(vari_temp)
            param = [500, 10, 0.1, 1, 1, 1.5, vari_temp(g), 1]; % [nGN tau p_GG p_z alpha g Tmax dt]
            tic
            for c = 1:20
                [maerr_w(g, c),rmserr_w(g, c),Wabs_w(g, c)] = RNN_v04_7(param);
                [maerr_r(g, c),rmserr_r(g, c),Wabs_r(g, c)] = RNN_v05_4(param);
            end
            toc
            disp(['Group ' num2str(g) ' finished']);
        end
    case 5 % learning rate
        %% alpha is the vari
        clc
        vari_temp = 1:9:100;
        for g = 1:length(vari_temp)
            param = [500, 10, 0.1, 1, vari_temp(g), 1.5, 2000, 1]; % [nGN tau p_GG p_z alpha g Tmax dt]
            tic
            for c = 1:16
                [maerr_w(g, c),rmserr_w(g, c),Wabs_w(g, c)] = RNN_v04_7(param);
                [maerr_r(g, c),rmserr_r(g, c),Wabs_r(g, c)] = RNN_v05_4(param);
            end
            toc
            disp(['Group ' num2str(g) ' finished']);
        end
end

maerrplot1 = mean(maerr_w,2);
rmserrplot1 = mean(rmserr_w,2);
Wabsplot1 = mean(Wabs_w,2);
maerrplot2 = mean(maerr_r,2);
rmserrplot2 = mean(rmserr_r,2);
Wabsplot2 = mean(Wabs_r,2);

%% save files
diffp.maerr_w = maerr_w;
diffp.rmserr_w = rmserr_w;
diffp.Wabs_w = Wabs_w;
diffp.maerr_r = maerr_r;
diffp.rmserr_r = rmserr_r;
diffp.Wabs_r = Wabs_r;
diffp.parameter.nGN = param(1);
diffp.parameter.tau = param(2);
diffp.parameter.p_GG = param(3);
diffp.parameter.p_z = param(4);
diffp.parameter.alpha = param(5);
diffp.parameter.g = param(6);
diffp.parameter.training_time = param(7);
diffp.parameter.dt = param(8);
diffp.code.FORCEonJ = 'RNN_v05_4.m';
diffp.code.FORCEonW = 'RNN_v04_7.m';
fname = string(vari);
save(fname,'-struct','diffp');

%% visualization
clf
subplot 311
hold on
plot(vari_temp,maerrplot1,'.-', 'color', 'k');
plot(vari_temp,maerrplot2,'.-', 'color', 'r');
ylabel('MAE');
% xlabel('g');
subplot 312
hold on
plot(vari_temp,rmserrplot1,'.-', 'color', 'k');
plot(vari_temp,rmserrplot2,'.-', 'color', 'r');
ylabel('RMS');
% xlabel('g');
subplot 313
hold on
plot(vari_temp,Wabsplot1,'.-', 'color', 'k');
plot(vari_temp,Wabsplot2,'.-', 'color', 'r');
ylabel('|W|');
xlabel('pZ');
savefig(fname);
clear all
