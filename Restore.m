clear;
close all;
clc;
tic;

% 相关需要设置的参数取值cases
InitialFluxes = {'[0.7 -0.7 -0.7]','[0.6 0 -0.6]','[0.5 -0.3 -0.3]','[0.5 0 0]'};
Winding1Connections = {'Y','Yn','Yg','Delta (D1)','Delta (D11)'};
Winding2Connections = {'Y','Yn','Yg','Delta (D1)','Delta (D11)'};
NominalPowers = {'[100e6 50]','[300e6 50]'};
PhaseAngles = {'-180','-150','-120','-90','-60','-30','0','30','60','90','120','150','180'};
view = 1; % 是否显示仿真数据波形和差流波形
% ********** 空载合闸时的励磁涌流 ********** %
% 影响因素：是否同期合闸、合闸角、容量、剩磁、接线方式、铁芯材料、系统阻抗等，目前暂未考虑铁芯材料和系统阻抗
% 与影响因素对应的相关模块：Breaker——是否同期合闸  Em——合闸角  Transformer——容量、剩磁、接线方式
if 0
    fprintf('Start simulation for InrushCurrent_NoLoadSwitch...\n');
    SwitchingTimes.Breaker = {'[0.1 1]','[0.2 1]','[0.3 1]'}; % 设置一种同期合闸和一种非同期合闸
    
    Total = length(Winding1Connections)*length(Winding2Connections)*length(InitialFluxes)*length(NominalPowers)*length(PhaseAngles)*2;
    RootPath = 'E:\关于项目\江苏涌流识别\仿真\bsh\2019\';
    Model = 'ExciterFlowCurrent_NoLoadSwitch.slx';
    CaseName = '空投涌流';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    open_system(Model);
    
    flag = 0;
    for No_Winding1Connection = 1:length(Winding1Connections)
        for No_Winding2Connection = 1:length(Winding2Connections)
            for No_InitialFluxes = 1:length(InitialFluxes)
                for No_NominalPower = 1:length(NominalPowers)
                    % 变压器调参:接线方式、剩磁、容量
                    set_param('ExciterFlowCurrent_NoLoadSwitch/Transformer','InitialFluxes',InitialFluxes{No_InitialFluxes},...
                        'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                        'Winding2Connection',Winding2Connections{No_Winding2Connection},...
                        'NominalPower',NominalPowers{No_NominalPower});
                    for No_PhaseAngle = 1:length(PhaseAngles)
                        % 电源调参：合闸角
                        set_param('ExciterFlowCurrent_NoLoadSwitch/Em','PhaseAngle',PhaseAngles{No_PhaseAngle});
                        for No_Breaker = 1:2 % 设置一种同期合闸和一种非同期合闸
                            % 断路器调参
                            if No_Breaker == 1 % 同期合闸
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{1});
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{1});
                                FileName = strcat('空投涌流_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_',NominalPowers{No_NominalPower},'VA_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','同期');
                            else % 非同期合闸
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{2});
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{3});
                                FileName = strcat('空投涌流_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_',NominalPowers{No_NominalPower},'VA_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','非同期');
                            end
                            try
                            % 运行slx仿真模型
                                sim(strcat(RootPath,Model)); % 当前模型设置电压等级为35/220kV
                            catch
                                fprintf(strcat('Failure:',FileName));
                                continue;
                            end
                            % 将一次侧、二次侧数据保存在txt中，Data1为一次侧数据，Data2为二次测数据
                            Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中 
                            Data2 = importdata('Iabc_2.mat');
                            FullName = strcat(SavePath,FileName,'.txt');
                            Signal = [Data1.Time Data1.Data Data2.Data]; % 时间、一次侧Iabc、二次测Iabc
                            dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                            flag = flag+1;
                            if view
                                hf=figure;
                                subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                                subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right'); 

                                subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                                subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right'); 
                                
                                bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                                saveas(gcf,strrep(FullName,'.txt','.png'));
                                set(hf,'Visible','off');close all;
                            end
                            % 打印进度条
                            if mod(flag,1000) == 0
                                percentage=sprintf('%.2f%%', flag/Total*100);
                                disp(['Completed ',percentage]);
                            end
                        end
                    end
                end
            end
        end
    end
    save_system(Model);close_system(Model);
end

% ********** 外部故障切除时的恢复性涌流 ********** %
% 影响因素：接线方式、故障发生时刻、故障切除时刻、故障严重程度、故障类型、负荷类型等，目前暂未考虑故障发生时刻、负荷类型
if 0
    fprintf('Start simulation for RecoveryCurrent_ExternalFault...\n');
    FaultTypes = {{'on','off','off','on'},{'off','on','off','on'},{'off','off','on','on'},...
        {'on','on','off','on'},{'off','on','on','on'},{'on','off','on','on'},{'on','on','on','on'},...
        {'on','on','off','off'},{'off','on','on','off'},{'on','off','on','off'}};
    FaultNames = {'AN','BN','CN','ABN','BCN','ACN','ABCN','AB','BC','AC'};
    SwitchTypes = {{'on','off','off'},{'off','on','off'},{'off','off','on'},...
        {'on','on','off'},{'off','on','on'},{'on','off','on'},{'on','on','on'},...
        {'on','on','off'},{'off','on','on'},{'on','off','on'}}; % 断路器采用分相动作机制
    Line1.Length = {'120','30','60','90','120','150','180','210','120'}; % 假设线路总长240km
    Line2.Length = {'120','210','180','150','120','90','60','30','120'};
    PositionNames = {'始端故障','距离系统侧12.5%处故障','距离系统侧25%处故障','距离系统侧37.5%处故障','距离系统侧50%处故障',...
        '距离系统侧62.5%处故障','距离系统侧75%处故障','距离系统侧87.5%处故障','终端故障'};
    SwitchingTimes.Breaker = {'[0.35 0.85]','[0.36 0.86]'}; % 假设0.3s故障发生持续60ms，断路器跳开0.5s后重合闸
    SwitchNames = {'故障后50ms切除','故障后60ms切除'};
    
    Total = length(Winding1Connections)*length(Winding2Connections)*(2*length(FaultTypes)-1)*length(Line1.Length)*length(SwitchingTimes.Breaker);
    RootPath = 'E:\关于项目\江苏涌流识别\仿真\bsh\2019\';
    Model = 'RecoveryCurrent_ExternalFault.slx';
    CaseName = '恢复涌流';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    open_system(Model);
    
    flag = 0;
    for No_Winding1Connection = 1:length(Winding1Connections)
        for No_Winding2Connection = 1:length(Winding2Connections)
            % 变压器调参:接线方式
            set_param('RecoveryCurrent_ExternalFault/Transformer1',...
                'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                'Winding2Connection',Winding2Connections{No_Winding2Connection});           
            for No_LineLength = 1:length(Line1.Length)
                % 线路调参:线路分段，模拟线路上的故障点位置
                set_param('RecoveryCurrent_ExternalFault/Line1','Length',Line1.Length{No_LineLength});
                set_param('RecoveryCurrent_ExternalFault/Line2','Length',Line2.Length{No_LineLength});                        
                for No_FaultType = 1:length(FaultTypes)
                    for No_SwitchTime = 1:length(SwitchingTimes.Breaker)
                        % 故障模块调参
                        if No_LineLength==1 %始端故障，设置Fault，Fault_S，Fault_T时间
                            set_param('RecoveryCurrent_ExternalFault/Fault_S',...
                                'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                                'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                                'SwitchTimes','[0.3 0.36]');               
                            set_param('RecoveryCurrent_ExternalFault/Fault','SwitchTimes','[10.3 10.36]');
                            set_param('RecoveryCurrent_ExternalFault/Fault_T','SwitchTimes','[10.3 10.36]');                        
                        elseif No_LineLength==length(Line1.Length) %终端故障，设置Fault，Fault_S，Fault_T时间
                            set_param('RecoveryCurrent_ExternalFault/Fault_T',...
                                'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                                'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                                'SwitchTimes','[0.3 0.36]');
                            set_param('RecoveryCurrent_ExternalFault/Fault_S','SwitchTimes','[10.3 10.36]');
                            set_param('RecoveryCurrent_ExternalFault/Fault','SwitchTimes','[10.3 10.36]');
                        else
                            set_param('RecoveryCurrent_ExternalFault/Fault',...
                                'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                                'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                                'SwitchTimes','[0.3 0.36]');
                            set_param('RecoveryCurrent_ExternalFault/Fault_S','SwitchTimes','[10.3 10.36]');
                            set_param('RecoveryCurrent_ExternalFault/Fault_T','SwitchTimes','[10.3 10.36]');
                        end
                        
                        % 断路器模块调参
                        if No_FaultType~=7 % 对于非对称性故障，考虑全切和分相切
                            for cut=1:2
                                if cut==1 % 分相切
                                    set_param('RecoveryCurrent_ExternalFault/Breaker1',...
                                        'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                        'SwitchA',SwitchTypes{No_FaultType}{1},'SwitchB',SwitchTypes{No_FaultType}{2},...
                                        'SwitchC',SwitchTypes{No_FaultType}{3});
                                    set_param('RecoveryCurrent_ExternalFault/Breaker2',...
                                        'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                        'SwitchA',SwitchTypes{No_FaultType}{1},'SwitchB',SwitchTypes{No_FaultType}{2},...
                                        'SwitchC',SwitchTypes{No_FaultType}{3});
                                    FileName = strcat('恢复涌流_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                        '_',FaultNames{No_FaultType},'_',PositionNames{No_LineLength},'_',SwitchNames{No_SwitchTime},'_分相切');
                                    try
                                        sim(strcat(RootPath,Model)); % 当前模型设置电压等级为220/500kV
                                    catch
                                        fprintf(strcat('Failure:',FileName));
                                        continue;
                                    end
                                    
                                    % Data1为一次侧数据，Data2为二次测数据
                                    Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中
                                    Data2 = importdata('Iabc_2.mat');
                                    FullName = strcat(SavePath,FileName,'.txt');
                                    Signal = [Data1.Time Data1.Data Data2.Data]; % 时间、一次侧Iabc、二次测Iabc
                                    dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                                    flag = flag+1;
                                    if view
                                        hf=figure;
                                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right'); 

                                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right'); 
                                
                                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                                        saveas(gcf,strrep(FullName,'.txt','.png'));
                                        set(hf,'Visible','off');close all;
                                    end
                                elseif cut==2  % 全切
                                    set_param('RecoveryCurrent_ExternalFault/Breaker1',...
                                        'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                        'SwitchA','on','SwitchB','on','SwitchC','on');
                                    set_param('RecoveryCurrent_ExternalFault/Breaker2',...
                                        'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                        'SwitchA','on','SwitchB','on','SwitchC','on');
                                    FileName = strcat('恢复涌流_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                        '_',FaultNames{No_FaultType},'_',PositionNames{No_LineLength},'_',SwitchNames{No_SwitchTime},'_全切');
                                end
                            end
                        else
                            set_param('RecoveryCurrent_ExternalFault/Breaker1',...
                                'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                'SwitchA','on','SwitchB','on','SwitchC','on');
                            set_param('RecoveryCurrent_ExternalFault/Breaker2',...
                                'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                'SwitchA','on','SwitchB','on','SwitchC','on');
                            FileName = strcat('恢复涌流_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_',FaultNames{No_FaultType},'_',PositionNames{No_LineLength},'_',SwitchNames{No_SwitchTime},'_全切');
                        end  
                        % 运行仿真模型
                        try
                            sim(strcat(RootPath,Model)); % 当前模型设置电压等级为35/220kV
                        catch
                            fprintf(strcat('Failure:',FileName));
                            continue;
                        end
                        % Data1为一次侧数据，Data2为二次测数据
                        Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中
                        Data2 = importdata('Iabc_2.mat');
                        FullName = strcat(SavePath,FileName,'.txt');
                        Signal = [Data1.Time Data1.Data Data2.Data]; % 时间、一次侧Iabc、二次测Iabc
                        dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                        flag = flag+1;
                        if view
                            hf=figure;
                            subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                            subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                            
                            subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                            subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                                
                            bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                            saveas(gcf,strrep(FullName,'.txt','.png'));
                            set(hf,'Visible','off');close all;
                        end
                        % 打印进度条
                        if mod(flag,1000) == 0
                            percentage=sprintf('%.2f%%', flag/Total*100);
                            disp(['Completed ',percentage]);
                        end
                    end
                end
            end
        end
    end
    save_system(Model);close_system(Model);
end
                            

% ********** 空载合闸串/并联变压器时相邻变压器的和应涌流 ********** %
% 影响因素：是否同期合闸、合闸角、剩磁、接线方式、容量、系统阻抗等，目前暂未考虑容量和系统阻抗
if 0 % 串联和应
    fprintf('Start simulation for SympatheticCurrent_Series...\n');
    SwitchingTimes.Breaker = {'[1.1 2]','[1.2 2]','[1.3 2]'}; % 设置一种同期合闸和一种非同期合闸
    
    Total = length(Winding1Connections)*length(Winding2Connections)*length(InitialFluxes)*length(PhaseAngles)*2;
    RootPath = 'E:\关于项目\江苏涌流识别\仿真\bsh\2019\';
    Model = 'SympatheticCurrent_Series.slx';
    CaseName = '和应涌流'; % 和应涌流和变压器接线方式密切相关，部分接线方式下无和应特征
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');        
    open_system(Model);
    
    flag = 0; % 此工况暂时么考虑变压器容量参数的变化
    for No_Winding1Connection = 1:length(Winding1Connections)
        for No_Winding2Connection = 1:length(Winding2Connections)
            for No_InitialFluxes = 1:length(InitialFluxes)
                % 变压器调参:接线方式、剩磁
                set_param('SympatheticCurrent_Series/Transformer1','InitialFluxes',InitialFluxes{No_InitialFluxes},...
                    'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                    'Winding2Connection',Winding2Connections{No_Winding2Connection});
                set_param('SympatheticCurrent_Series/Transformer2',...
                    'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                    'Winding2Connection',Winding2Connections{No_Winding2Connection});%'InitialFluxes',InitialFluxes{No_InitialFluxes},
                for No_PhaseAngle = 1:length(PhaseAngles)
                    % 电源调参：合闸角
                    set_param('SympatheticCurrent_Series/Em','PhaseAngle',PhaseAngles{No_PhaseAngle});                     
                    for No_Breaker = 1:2 %设置一种同期合闸和一种非同期合闸
                        % 是否同期合闸
                        if No_Breaker == 1 % 同期合闸
                            set_param('SympatheticCurrent_Series/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Series/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Series/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{1});
                            FileName = strcat('和应涌流_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_','串联_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','同期');
                        else % 非同期合闸
                            set_param('SympatheticCurrent_Series/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Series/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{2});
                            set_param('SympatheticCurrent_Series/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{3});
                            FileName = strcat('和应涌流_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_','串联_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','非同期');
                        end
                        % 运行仿真模型
                        try
                            sim(strcat(RootPath,Model)); % 当前模型设置电压等级为35/220kV
                        catch
                            fprintf(strcat('Failure:',FileName));
                            continue;
                        end
                        % Data1为一次侧数据，Data2为二次测数据
                        Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中
                        Data2 = importdata('Iabc_2.mat');
                        FullName = strcat(SavePath,FileName,'.txt');
                        select=(Data1.Time>=1.0); % 截取后一段数据，因前一段数据是空投涌流数据
                        Signal = [Data1.Time(select)-1 Data1.Data(select,:) Data2.Data(select,:)]; % 时间、一次侧Iabc、二次测Iabc
                        dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                        flag = flag+1;
                        if view
                            hf=figure;
                            subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                            subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                            
                            subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                            subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                             
                            bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                            saveas(gcf,strrep(FullName,'.txt','.png'));
                            set(hf,'Visible','off');close all;
                        end
                        % 打印进度条
                        if mod(flag,1000) == 0
                            percentage=sprintf('%.2f%%', flag/Total*100);
                            disp(['Completed ',percentage]);
                        end
                    end
                end
            end
        end
    end
    save_system(Model);close_system(Model);
end

if 0 % 并联和应
    fprintf('Start simulation for SympatheticCurrent_Parallel...\n');
    SwitchingTimes.Breaker = {'[1.1 2]','[1.2 2]','[1.3 2]'}; % 设置一种同期合闸和一种非同期合闸
    
    Total = length(Winding1Connections)*length(Winding2Connections)*length(InitialFluxes)*length(PhaseAngles)*2;
    RootPath = 'E:\关于项目\江苏涌流识别\仿真\bsh\2019\';
    Model = 'SympatheticCurrent_Parallel.slx';
    CaseName = '和应涌流';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    open_system(Model);
    
    flag = 0;
    for No_Winding1Connection = 1:length(Winding1Connections)
        for No_Winding2Connection = 1:length(Winding2Connections)
            for No_InitialFluxes = 1:length(InitialFluxes)
                % 变压器调参：接线方式、剩磁
                set_param('SympatheticCurrent_Parallel/Transformer1','InitialFluxes',InitialFluxes{No_InitialFluxes},...
                    'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                    'Winding2Connection',Winding2Connections{No_Winding2Connection});
                set_param('SympatheticCurrent_Parallel/Transformer2',...
                    'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                    'Winding2Connection',Winding2Connections{No_Winding2Connection});                        
                for No_PhaseAngle = 1:length(PhaseAngles)
                    % 电源调参：合闸角
                    set_param('SympatheticCurrent_Parallel/Em','PhaseAngle',PhaseAngles{No_PhaseAngle});                        
                    for No_Breaker = 1:2 % 设置一种同期合闸和一种非同期合闸
                        if No_Breaker == 1 % 同期合闸
                            set_param('SympatheticCurrent_Parallel/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Parallel/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Parallel/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{1});
                            FileName = strcat('和应涌流_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_','并联_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','同期');
                        else % 非同期合闸
                            set_param('SympatheticCurrent_Parallel/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Parallel/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{2});
                            set_param('SympatheticCurrent_Parallel/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{3});
                            FileName = strcat('和应涌流_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_','并联_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','非同期');
                        end
                        % 运行仿真模型
                        try
                            sim(strcat(RootPath,Model)); % 当前模型设置电压等级为35/220kV
                        catch
                            fprintf(strcat('Failure:',FileName));
                            continue;
                        end
                        
                        % Data1为一次侧数据，Data2为二次测数据
                        Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中
                        Data2 = importdata('Iabc_2.mat');
                        FullName = strcat(SavePath,FileName,'.txt');
                        select=(Data1.Time>=1.0);
                        Signal = [Data1.Time(select)-1 Data1.Data(select,:) Data2.Data(select,:)]; % 时间、一次侧Iabc、二次测Iabc
                        dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                        flag = flag+1;
                        if view
                            hf=figure;
                            subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                            subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                            
                            subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                            subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                             
                            bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                            saveas(gcf,strrep(FullName,'.txt','.png'));
                            set(hf,'Visible','off');close all;
                        end
                        % 打印进度条
                        if mod(flag,1000) == 0
                            percentage=sprintf('%.2f%%', flag/Total*100);
                            disp(['Completed ',percentage]);
                        end
                    end
                end
            end
        end
    end
    save_system(Model);close_system(Model);
end

% ********** 内部故障(包括引线故障和绕组故障) ********** %
% 影响因素：故障持续时间、故障类型、故障位置等
if 0 % 内部故障_引线故障
    fprintf('Start simulation for InternalFault_Lead...\n');
    FaultTypes = {{'on','off','off','on'},{'off','on','off','on'},{'off','off','on','on'},...
        {'on','on','off','on'},{'off','on','on','on'},{'on','off','on','on'},{'on','on','on','on'},...
        {'on','on','off','off'},{'off','on','on','off'},{'on','off','on','off'}};
    FaultNames = {'AN','BN','CN','ABN','BCN','ACN','ABCN','AB','BC','AC'};
    SwitchTimes.Fault = {'[0.3 0.34]','[0.3 0.36]','[0.3 0.38]',...
        '[0.31 0.35]','[0.31 0.37]','[0.31 0.39]'}; % 设置故障时刻和故障持续时间
    
    RootPath = 'E:\关于项目\江苏涌流识别\仿真\bsh\2019\';
    CaseName = '引线故障';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    Models = dir(strcat(RootPath,'InternalFaults_Lead_*.slx'));
    Total = length(FaultTypes)*length(SwitchTimes.Fault)*2*length(Models);
    
    flag = 0;
    for No_Model = 1:length(Models)
        Model = Models(No_Model).name; % 选择某个模型
        open_system(Model);
        for No_FaultType = 1:length(FaultTypes)
            for No_SwitchTime = 1:length(SwitchTimes.Fault)
                for No_FaultPosition =1:2 % 一次侧故障还是二次侧故障
                    if No_FaultPosition == 1 % 一次侧引线故障
                        % 故障调参：故障类型、故障时间
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),...
                            'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                            'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                            'SwitchTimes',SwitchTimes.Fault{No_SwitchTime});
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),'SwitchTimes','[10 11]');
                        % 根据故障位置修改
                        PositionName = '一次侧';
                    else
                        % 故障调参：故障类型、故障时间
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),...
                            'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                            'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                            'SwitchTimes',SwitchTimes.Fault{No_SwitchTime});
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),'SwitchTimes','[10 11]');
                        % 根据故障位置修改
                        PositionName = '二次侧';
                    end
                    
                    [str1,str2] = strsplit(Models(No_Model).name,'Lead_');
                    WindConnection = strrep(str1{2},'.slx','');
                    FileName = strcat('引线故障_',WindConnection,'_',PositionName,'_',FaultNames{No_FaultType},'_',SwitchTimes.Fault{No_SwitchTime},'s');
                    % 运行仿真模型
                    try
                        sim(strcat(RootPath,Model)); % 当前模型设置电压等级为35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1为一次侧数据，Data2为二次测数据
                    Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中
                    Data2 = importdata('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % 时间、一次侧Iabc、二次测Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                        
                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                        saveas(gcf,strrep(FullName,'.txt','.png'));
                        set(hf,'Visible','off');close all;
                    end
                    % 打印进度条
                    if mod(flag,500) == 0
                        percentage=sprintf('%.2f%%', flag/Total*100);
                        disp(['Completed ',percentage]);
                    end
                end
            end
        end
        save_system(Model);close_system(Model);
    end
end

%%

% FaultTypes = {{'on','off','off','on'},{'off','on','off','on'},{'off','off','on','on'},...
%     {'on','on','off','on'},{'off','on','on','on'},{'on','off','on','on'},{'on','on','on','on'},...
%     {'on','on','off','off'},{'off','on','on','off'},{'on','off','on','off'}};
% FaultNames = {'AN','BN','CN','ABN','BCN','ACN','ABCN','AB','BC','AC'};

%%
% 影响因素：故障持续时间、故障类型、故障位置等
if 0 % 内部故障_绕组故障_匝地  内部故障_绕组故障_相间
    try
        delete('Iabc_1.mat');delete('Iabc_2.mat');
    catch
        fprintf('No output files currently');
    end
    
    view = 0;
    fprintf('Start simulation for InternalFault_Winding...\n');
    FaultTypes = {{'on','off','on','off'}};
    FaultNames = {'AC'};
    SwitchTimes.Fault = {'[0.3 0.36]','[0.31 0.39]'}; % 设置故障时刻和故障持续时间
    
    RootPath = 'E:\关于项目\江苏涌流识别\仿真\bsh\2019\';
    CaseName = '绕组故障';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    Models = dir(strcat(RootPath,'InternalFaults_Winding_*.slx'));
    Total = length(FaultTypes)*length(SwitchTimes.Fault)*2*length(Models);
    
    flag = 0;
    for No_Model = 1:length(Models)
        Model = Models(No_Model).name; % 选择某个模型
        open_system(Model);
        for No_FaultType = 1:length(FaultTypes)
            for No_SwitchTime = 1:length(SwitchTimes.Fault)
                for No_FaultPosition =1:2 % 一次侧故障还是二次侧故障
                    if No_FaultPosition == 1 % 一次侧绕组故障
                        % 故障调参：故障类型、故障时间
                        % 一次侧相关控制开关状态
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','close','SwitchA','on',...
                            'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % 模拟一次侧线圈匝模型
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),...
                            'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                            'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                            'SwitchTimes',SwitchTimes.Fault{No_SwitchTime});
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        % 二次侧相关控制开关状态
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','open','SwitchA','off',...
                            'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % 模拟二次侧线圈匝模型
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),'SwitchTimes','[10 11]');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        % 根据故障位置修改
                        PositionName = '一次侧40%A50%C';
                    else
                        % 故障调参：故障类型、故障时间
                        % 二次侧相关控制开关状态
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','close','SwitchA','on',...
                            'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % 模拟二次侧线圈匝模型
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),...
                            'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                            'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                            'SwitchTimes',SwitchTimes.Fault{No_SwitchTime});
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        % 一次侧相关控制开关状态
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','open','SwitchA','off',...
                            'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % 模拟一次侧线圈匝模型
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),'SwitchTimes','[10 11]');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        % 根据故障位置修改
                        PositionName = '二次侧50%A40%C';
                    end
                    
                    [str1,str2] = strsplit(Models(No_Model).name,'Winding_');
                    WindConnection = strrep(str1{2},'.slx','');
                    FileName = strcat('绕组故障相间_',PositionName,'_',WindConnection,'_',FaultNames{No_FaultType},'_',SwitchTimes.Fault{No_SwitchTime},'s');
                    % 运行仿真模型
                    try
                        sim(strcat(RootPath,Model)); % 当前模型设置电压等级为35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1为一次侧数据，Data2为二次测数据
                    Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中
                    Data2 = importdata('Iabc_2.mat');
                    delete('Iabc_1.mat');delete('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % 时间、一次侧Iabc、二次测Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                        
                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                        saveas(gcf,strrep(FullName,'.txt','.png'));
                        set(hf,'Visible','on');%close all;
                    end
                    % 打印进度条
                    if mod(flag,1000) == 0
                        percentage=sprintf('%.2f%%', flag/Total*100);
                        disp(['Completed ',percentage]);
                    end
                end
            end
        end
        save_system(Model);%close_system(Model);
    end
end


% 影响因素：故障持续时间、故障类型、故障位置等
if 0 % 内部故障_绕组故障_匝间_单相匝间
    try
        delete('Iabc_1.mat');delete('Iabc_2.mat');
    catch
        fprintf('No output files currently');
    end
    
    view = 0;
    fprintf('Start simulation for InternalFault_Winding...\n');
    SwitchTimes.Fault = {'[0.3 0.36]','[0.31 0.39]'}; % 设置故障时刻和故障持续时间
    
    RootPath = 'E:\关于项目\江苏涌流识别\仿真\bsh\2019\';
    CaseName = '绕组故障';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    Models = dir(strcat(RootPath,'InternalFaults_Winding_*.slx'));
    Total = 3*2*length(Models);
    
    flag = 0;
    for No_Model = 1:length(Models)
        Model = Models(No_Model).name; % 选择某个模型
        open_system(Model);
        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),'SwitchTimes','[10 11]'); % 不发生匝地和匝间故障
        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),'SwitchTimes','[10 11]');
        for No_FaultPosition =1:2 % 一次侧故障还是二次侧故障
            if No_FaultPosition == 1 % 一次侧绕组故障
                % 故障调参：故障类型、故障时间
                % 一次侧相关控制开关状态
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','close','SwitchA','on',...
                    'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % 模拟一次侧线圈匝模型
                
                % 二次侧相关控制开关状态
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','open','SwitchA','off',...
                    'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % 模拟二次侧线圈匝模型
                
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                
                for NoCase=1:3
                    if NoCase==1
                        % 根据故障位置修改
                        PositionName = '一次侧80%A';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % 匝间故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                    elseif NoCase==2
                        PositionName = '一次侧80%B';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                    elseif NoCase==3
                        PositionName = '一次侧80%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                    end
                    [str1,str2] = strsplit(Models(No_Model).name,'Winding_');
                    WindConnection = strrep(str1{2},'.slx','');
                    FileName = strcat('绕组故障匝间_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},'s');
                    % 运行仿真模型
                    try
                        sim(strcat(RootPath,Model)); % 当前模型设置电压等级为35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1为一次侧数据，Data2为二次测数据
                    Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中
                    Data2 = importdata('Iabc_2.mat');
                    delete('Iabc_1.mat');delete('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % 时间、一次侧Iabc、二次测Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                        
                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                        saveas(gcf,strrep(FullName,'.txt','.png'));
                        set(hf,'Visible','off');close all;
                    end
                end
            else
                % 故障调参：故障类型、故障时间
                % 二次侧相关控制开关状态
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','close','SwitchA','on',...
                    'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % 模拟二次侧线圈匝模型
                % 一次侧相关控制开关状态
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','open','SwitchA','off',...
                    'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % 模拟一次侧线圈匝模型
                
                
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                
                for NoCase=1:3
                    if NoCase==1
                        % 根据故障位置修改
                        PositionName = '二次侧80%A';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                    elseif NoCase==2
                        % 根据故障位置修改
                        PositionName = '二次侧80%B';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                    elseif NoCase==3
                        % 根据故障位置修改
                        PositionName = '二次侧80%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                    end
                    [str1,str2] = strsplit(Models(No_Model).name,'Winding_');
                    WindConnection = strrep(str1{2},'.slx','');
                    FileName = strcat('绕组故障匝间_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},'s');
                    % 运行仿真模型
                    try
                        sim(strcat(RootPath,Model)); % 当前模型设置电压等级为35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1为一次侧数据，Data2为二次测数据
                    Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中
                    Data2 = importdata('Iabc_2.mat');
                    delete('Iabc_1.mat');delete('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % 时间、一次侧Iabc、二次测Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                        
                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                        saveas(gcf,strrep(FullName,'.txt','.png'));
                        set(hf,'Visible','off');close all;
                    end                    
                end
            end
        end
        save_system(Model);%close_system(Model);
    end
    % 打印进度条
    if mod(flag,1000) == 0
        percentage=sprintf('%.2f%%', flag/Total*100);
        disp(['Completed ',percentage]);
    end
end

if 0 % 内部故障_绕组故障_匝间_多相匝间
    try
        delete('Iabc_1.mat');delete('Iabc_2.mat');
    catch
        fprintf('No output files currently');
    end
    
    view = 0;
    fprintf('Start simulation for InternalFault_Winding...\n');
    SwitchTimes.Fault = {'[0.3 0.36]','[0.31 0.39]','[0.41 0.49]'}; % 设置故障时刻和故障持续时间
    
    RootPath = 'E:\关于项目\江苏涌流识别\仿真\bsh\2019\';
    CaseName = '绕组故障';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    Models = dir(strcat(RootPath,'InternalFaults_Winding_*.slx'));
    Total = 4*length(Models);
    
    flag = 0;
    for No_Model = 1:length(Models)
        Model = Models(No_Model).name; % 选择某个模型
        open_system(Model);
        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),'SwitchTimes','[10 11]'); % 不发生匝地和匝间故障
        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),'SwitchTimes','[10 11]');
        for No_FaultPosition =1:2 % 一次侧故障还是二次侧故障
            if No_FaultPosition == 1 % 一次侧绕组故障
                % 故障调参：故障类型、故障时间
                % 一次侧相关控制开关状态
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','close','SwitchA','on',...
                    'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % 模拟一次侧线圈匝模型
                
                % 二次侧相关控制开关状态
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','open','SwitchA','off',...
                    'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % 模拟二次侧线圈匝模型
                
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                
                [str1,str2] = strsplit(Models(No_Model).name,'Winding_');
                WindConnection = strrep(str1{2},'.slx','');
                for NoCase=1:4
                    if NoCase==1 % AB故障
                        % 根据故障位置修改
                        PositionName = '一次侧60%A80%B';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % 匝间故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        FileName = strcat('绕组故障匝间_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},'s');
                    elseif NoCase==2 % BC故障
                        PositionName = '一次侧80%B70%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        FileName = strcat('绕组故障匝间_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},'s');
                    elseif NoCase==3 % AC故障
                        PositionName = '一次侧60%A70%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % 匝间故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{3},'External','off');
                        FileName = strcat('绕组故障匝间_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},SwitchTimes.Fault{3},'s');
                    elseif NoCase==4 % ABC故障
                        PositionName = '一次侧60%A80%B70%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % 匝间故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{2},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{3},'External','off');
                        FileName = strcat('绕组故障匝间_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},SwitchTimes.Fault{2},SwitchTimes.Fault{3},'s');
                    end
                    
                    
                    % 运行仿真模型
                    try
                        sim(strcat(RootPath,Model)); % 当前模型设置电压等级为35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1为一次侧数据，Data2为二次测数据
                    Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中
                    Data2 = importdata('Iabc_2.mat');
                    delete('Iabc_1.mat');delete('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % 时间、一次侧Iabc、二次测Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                        
                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                        %saveas(gcf,strrep(FullName,'.txt','.png'));
                        set(hf,'Visible','on');%close all;
                    end
                end
            else
                % 故障调参：故障类型、故障时间
                % 二次侧相关控制开关状态
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','close','SwitchA','on',...
                    'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % 模拟二次侧线圈匝模型
                % 一次侧相关控制开关状态
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','open','SwitchA','off',...
                    'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % 模拟一次侧线圈匝模型
                
                
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                
                [str1,str2] = strsplit(Models(No_Model).name,'Winding_');
                WindConnection = strrep(str1{2},'.slx','');
                
                for NoCase=1:4
                    if NoCase==1 % AB故障
                        % 根据故障位置修改
                        PositionName = '二次侧70%A80%B';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{2},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        FileName = strcat('绕组故障匝间_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},SwitchTimes.Fault{2},'s');
                    elseif NoCase==2 % BC故障
                        % 根据故障位置修改
                        PositionName = '二次侧80%B60%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{2},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{3},'External','off');
                        FileName = strcat('绕组故障匝间_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{2},SwitchTimes.Fault{3},'s');
                    elseif NoCase==3 % AC故障
                        % 根据故障位置修改
                        PositionName = '二次侧70%A60%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        FileName = strcat('绕组故障匝间_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},'s');
                    elseif NoCase==4 % ABC故障
                        % 根据故障位置修改
                        PositionName = '二次侧70%A80%B60%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % 匝间不故障
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{2},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{3},'External','off');
                        FileName = strcat('绕组故障匝间_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},SwitchTimes.Fault{2},SwitchTimes.Fault{3},'s');
                    end
                    
                    
                    % 运行仿真模型
                    try
                        sim(strcat(RootPath,Model)); % 当前模型设置电压等级为35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1为一次侧数据，Data2为二次测数据
                    Data1 = importdata('Iabc_1.mat'); % 存储仿真数据至txt中
                    Data2 = importdata('Iabc_2.mat');
                    delete('Iabc_1.mat');delete('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % 时间、一次侧Iabc、二次测Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('一次侧/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('二次侧/pu');
                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                        
                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                        %saveas(gcf,strrep(FullName,'.txt','.png'));
                        set(hf,'Visible','on');%close all;
                    end                    
                end
            end
        end
        save_system(Model);%close_system(Model);
    end
    % 打印进度条
    if mod(flag,1000) == 0
        percentage=sprintf('%.2f%%', flag/Total*100);
        disp(['Completed ',percentage]);
    end
end

%% 备份
if 0
    model = 'E:\关于项目\江苏涌流识别\仿真\bsh\ExciterFlowCurrent_NoLoadSwitch.slx';
    sim(model);
    Data = importdata('Iabc.mat');
    filename='E:\关于项目\江苏涌流识别\仿真\bsh\test.txt';
    Signal = [Data.Time Data.Data];
    dlmwrite(filename,Signal,'delimiter',' '); % 保存运行结果至txt文件中   ,'precision',6
end

if 0
    Addr = 'E:\关于项目\江苏涌流识别\仿真\bsh\内部故障\';
    ToAddr = 'E:\关于项目\江苏涌流识别\仿真\bsh\ML试验\';
    Lists = dir(strcat(Addr,'*.txt'));
    for i=1:length(Lists)
        filename = Lists(i).name;
        SOURCE = strcat(Addr,filename);
        DESTINATION = strcat(ToAddr,'内部故障_',filename);
        [SUCCESS,MESSAGE,MESSAGEID] = copyfile(SOURCE,DESTINATION,'f');
    end
end

toc;