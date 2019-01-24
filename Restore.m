clear;
close all;
clc;
tic;

% �����Ҫ���õĲ���ȡֵcases
InitialFluxes = {'[0.7 -0.7 -0.7]','[0.6 0 -0.6]','[0.5 -0.3 -0.3]','[0.5 0 0]'};
Winding1Connections = {'Y','Yn','Yg','Delta (D1)','Delta (D11)'};
Winding2Connections = {'Y','Yn','Yg','Delta (D1)','Delta (D11)'};
NominalPowers = {'[100e6 50]','[300e6 50]'};
PhaseAngles = {'-180','-150','-120','-90','-60','-30','0','30','60','90','120','150','180'};
view = 1; % �Ƿ���ʾ�������ݲ��κͲ�������
% ********** ���غ�բʱ������ӿ�� ********** %
% Ӱ�����أ��Ƿ�ͬ�ں�բ����բ�ǡ�������ʣ�š����߷�ʽ����о���ϡ�ϵͳ�迹�ȣ�Ŀǰ��δ������о���Ϻ�ϵͳ�迹
% ��Ӱ�����ض�Ӧ�����ģ�飺Breaker�����Ƿ�ͬ�ں�բ  Em������բ��  Transformer����������ʣ�š����߷�ʽ
if 0
    fprintf('Start simulation for InrushCurrent_NoLoadSwitch...\n');
    SwitchingTimes.Breaker = {'[0.1 1]','[0.2 1]','[0.3 1]'}; % ����һ��ͬ�ں�բ��һ�ַ�ͬ�ں�բ
    
    Total = length(Winding1Connections)*length(Winding2Connections)*length(InitialFluxes)*length(NominalPowers)*length(PhaseAngles)*2;
    RootPath = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\2019\';
    Model = 'ExciterFlowCurrent_NoLoadSwitch.slx';
    CaseName = '��Ͷӿ��';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    open_system(Model);
    
    flag = 0;
    for No_Winding1Connection = 1:length(Winding1Connections)
        for No_Winding2Connection = 1:length(Winding2Connections)
            for No_InitialFluxes = 1:length(InitialFluxes)
                for No_NominalPower = 1:length(NominalPowers)
                    % ��ѹ������:���߷�ʽ��ʣ�š�����
                    set_param('ExciterFlowCurrent_NoLoadSwitch/Transformer','InitialFluxes',InitialFluxes{No_InitialFluxes},...
                        'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                        'Winding2Connection',Winding2Connections{No_Winding2Connection},...
                        'NominalPower',NominalPowers{No_NominalPower});
                    for No_PhaseAngle = 1:length(PhaseAngles)
                        % ��Դ���Σ���բ��
                        set_param('ExciterFlowCurrent_NoLoadSwitch/Em','PhaseAngle',PhaseAngles{No_PhaseAngle});
                        for No_Breaker = 1:2 % ����һ��ͬ�ں�բ��һ�ַ�ͬ�ں�բ
                            % ��·������
                            if No_Breaker == 1 % ͬ�ں�բ
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{1});
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{1});
                                FileName = strcat('��Ͷӿ��_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_',NominalPowers{No_NominalPower},'VA_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','ͬ��');
                            else % ��ͬ�ں�բ
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{2});
                                set_param('ExciterFlowCurrent_NoLoadSwitch/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{3});
                                FileName = strcat('��Ͷӿ��_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_',NominalPowers{No_NominalPower},'VA_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','��ͬ��');
                            end
                            try
                            % ����slx����ģ��
                                sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ35/220kV
                            catch
                                fprintf(strcat('Failure:',FileName));
                                continue;
                            end
                            % ��һ�βࡢ���β����ݱ�����txt�У�Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                            Data1 = importdata('Iabc_1.mat'); % �洢����������txt�� 
                            Data2 = importdata('Iabc_2.mat');
                            FullName = strcat(SavePath,FileName,'.txt');
                            Signal = [Data1.Time Data1.Data Data2.Data]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                            dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                            flag = flag+1;
                            if view
                                hf=figure;
                                subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                                subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right'); 

                                subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
                                subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right'); 
                                
                                bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                                saveas(gcf,strrep(FullName,'.txt','.png'));
                                set(hf,'Visible','off');close all;
                            end
                            % ��ӡ������
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

% ********** �ⲿ�����г�ʱ�Ļָ���ӿ�� ********** %
% Ӱ�����أ����߷�ʽ�����Ϸ���ʱ�̡������г�ʱ�̡��������س̶ȡ��������͡��������͵ȣ�Ŀǰ��δ���ǹ��Ϸ���ʱ�̡���������
if 0
    fprintf('Start simulation for RecoveryCurrent_ExternalFault...\n');
    FaultTypes = {{'on','off','off','on'},{'off','on','off','on'},{'off','off','on','on'},...
        {'on','on','off','on'},{'off','on','on','on'},{'on','off','on','on'},{'on','on','on','on'},...
        {'on','on','off','off'},{'off','on','on','off'},{'on','off','on','off'}};
    FaultNames = {'AN','BN','CN','ABN','BCN','ACN','ABCN','AB','BC','AC'};
    SwitchTypes = {{'on','off','off'},{'off','on','off'},{'off','off','on'},...
        {'on','on','off'},{'off','on','on'},{'on','off','on'},{'on','on','on'},...
        {'on','on','off'},{'off','on','on'},{'on','off','on'}}; % ��·�����÷��ද������
    Line1.Length = {'120','30','60','90','120','150','180','210','120'}; % ������·�ܳ�240km
    Line2.Length = {'120','210','180','150','120','90','60','30','120'};
    PositionNames = {'ʼ�˹���','����ϵͳ��12.5%������','����ϵͳ��25%������','����ϵͳ��37.5%������','����ϵͳ��50%������',...
        '����ϵͳ��62.5%������','����ϵͳ��75%������','����ϵͳ��87.5%������','�ն˹���'};
    SwitchingTimes.Breaker = {'[0.35 0.85]','[0.36 0.86]'}; % ����0.3s���Ϸ�������60ms����·������0.5s���غ�բ
    SwitchNames = {'���Ϻ�50ms�г�','���Ϻ�60ms�г�'};
    
    Total = length(Winding1Connections)*length(Winding2Connections)*(2*length(FaultTypes)-1)*length(Line1.Length)*length(SwitchingTimes.Breaker);
    RootPath = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\2019\';
    Model = 'RecoveryCurrent_ExternalFault.slx';
    CaseName = '�ָ�ӿ��';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    open_system(Model);
    
    flag = 0;
    for No_Winding1Connection = 1:length(Winding1Connections)
        for No_Winding2Connection = 1:length(Winding2Connections)
            % ��ѹ������:���߷�ʽ
            set_param('RecoveryCurrent_ExternalFault/Transformer1',...
                'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                'Winding2Connection',Winding2Connections{No_Winding2Connection});           
            for No_LineLength = 1:length(Line1.Length)
                % ��·����:��·�ֶΣ�ģ����·�ϵĹ��ϵ�λ��
                set_param('RecoveryCurrent_ExternalFault/Line1','Length',Line1.Length{No_LineLength});
                set_param('RecoveryCurrent_ExternalFault/Line2','Length',Line2.Length{No_LineLength});                        
                for No_FaultType = 1:length(FaultTypes)
                    for No_SwitchTime = 1:length(SwitchingTimes.Breaker)
                        % ����ģ�����
                        if No_LineLength==1 %ʼ�˹��ϣ�����Fault��Fault_S��Fault_Tʱ��
                            set_param('RecoveryCurrent_ExternalFault/Fault_S',...
                                'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                                'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                                'SwitchTimes','[0.3 0.36]');               
                            set_param('RecoveryCurrent_ExternalFault/Fault','SwitchTimes','[10.3 10.36]');
                            set_param('RecoveryCurrent_ExternalFault/Fault_T','SwitchTimes','[10.3 10.36]');                        
                        elseif No_LineLength==length(Line1.Length) %�ն˹��ϣ�����Fault��Fault_S��Fault_Tʱ��
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
                        
                        % ��·��ģ�����
                        if No_FaultType~=7 % ���ڷǶԳ��Թ��ϣ�����ȫ�кͷ�����
                            for cut=1:2
                                if cut==1 % ������
                                    set_param('RecoveryCurrent_ExternalFault/Breaker1',...
                                        'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                        'SwitchA',SwitchTypes{No_FaultType}{1},'SwitchB',SwitchTypes{No_FaultType}{2},...
                                        'SwitchC',SwitchTypes{No_FaultType}{3});
                                    set_param('RecoveryCurrent_ExternalFault/Breaker2',...
                                        'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                        'SwitchA',SwitchTypes{No_FaultType}{1},'SwitchB',SwitchTypes{No_FaultType}{2},...
                                        'SwitchC',SwitchTypes{No_FaultType}{3});
                                    FileName = strcat('�ָ�ӿ��_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                        '_',FaultNames{No_FaultType},'_',PositionNames{No_LineLength},'_',SwitchNames{No_SwitchTime},'_������');
                                    try
                                        sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ220/500kV
                                    catch
                                        fprintf(strcat('Failure:',FileName));
                                        continue;
                                    end
                                    
                                    % Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                                    Data1 = importdata('Iabc_1.mat'); % �洢����������txt��
                                    Data2 = importdata('Iabc_2.mat');
                                    FullName = strcat(SavePath,FileName,'.txt');
                                    Signal = [Data1.Time Data1.Data Data2.Data]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                                    dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                                    flag = flag+1;
                                    if view
                                        hf=figure;
                                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right'); 

                                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
                                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right'); 
                                
                                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                                        saveas(gcf,strrep(FullName,'.txt','.png'));
                                        set(hf,'Visible','off');close all;
                                    end
                                elseif cut==2  % ȫ��
                                    set_param('RecoveryCurrent_ExternalFault/Breaker1',...
                                        'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                        'SwitchA','on','SwitchB','on','SwitchC','on');
                                    set_param('RecoveryCurrent_ExternalFault/Breaker2',...
                                        'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                        'SwitchA','on','SwitchB','on','SwitchC','on');
                                    FileName = strcat('�ָ�ӿ��_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                        '_',FaultNames{No_FaultType},'_',PositionNames{No_LineLength},'_',SwitchNames{No_SwitchTime},'_ȫ��');
                                end
                            end
                        else
                            set_param('RecoveryCurrent_ExternalFault/Breaker1',...
                                'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                'SwitchA','on','SwitchB','on','SwitchC','on');
                            set_param('RecoveryCurrent_ExternalFault/Breaker2',...
                                'SwitchTimes',SwitchingTimes.Breaker{No_SwitchTime},...
                                'SwitchA','on','SwitchB','on','SwitchC','on');
                            FileName = strcat('�ָ�ӿ��_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_',FaultNames{No_FaultType},'_',PositionNames{No_LineLength},'_',SwitchNames{No_SwitchTime},'_ȫ��');
                        end  
                        % ���з���ģ��
                        try
                            sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ35/220kV
                        catch
                            fprintf(strcat('Failure:',FileName));
                            continue;
                        end
                        % Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                        Data1 = importdata('Iabc_1.mat'); % �洢����������txt��
                        Data2 = importdata('Iabc_2.mat');
                        FullName = strcat(SavePath,FileName,'.txt');
                        Signal = [Data1.Time Data1.Data Data2.Data]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                        dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                        flag = flag+1;
                        if view
                            hf=figure;
                            subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                            subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                            
                            subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
                            subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                                
                            bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                            saveas(gcf,strrep(FullName,'.txt','.png'));
                            set(hf,'Visible','off');close all;
                        end
                        % ��ӡ������
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
                            

% ********** ���غ�բ��/������ѹ��ʱ���ڱ�ѹ���ĺ�Ӧӿ�� ********** %
% Ӱ�����أ��Ƿ�ͬ�ں�բ����բ�ǡ�ʣ�š����߷�ʽ��������ϵͳ�迹�ȣ�Ŀǰ��δ����������ϵͳ�迹
if 0 % ������Ӧ
    fprintf('Start simulation for SympatheticCurrent_Series...\n');
    SwitchingTimes.Breaker = {'[1.1 2]','[1.2 2]','[1.3 2]'}; % ����һ��ͬ�ں�բ��һ�ַ�ͬ�ں�բ
    
    Total = length(Winding1Connections)*length(Winding2Connections)*length(InitialFluxes)*length(PhaseAngles)*2;
    RootPath = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\2019\';
    Model = 'SympatheticCurrent_Series.slx';
    CaseName = '��Ӧӿ��'; % ��Ӧӿ���ͱ�ѹ�����߷�ʽ������أ����ֽ��߷�ʽ���޺�Ӧ����
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');        
    open_system(Model);
    
    flag = 0; % �˹�����ʱô���Ǳ�ѹ�����������ı仯
    for No_Winding1Connection = 1:length(Winding1Connections)
        for No_Winding2Connection = 1:length(Winding2Connections)
            for No_InitialFluxes = 1:length(InitialFluxes)
                % ��ѹ������:���߷�ʽ��ʣ��
                set_param('SympatheticCurrent_Series/Transformer1','InitialFluxes',InitialFluxes{No_InitialFluxes},...
                    'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                    'Winding2Connection',Winding2Connections{No_Winding2Connection});
                set_param('SympatheticCurrent_Series/Transformer2',...
                    'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                    'Winding2Connection',Winding2Connections{No_Winding2Connection});%'InitialFluxes',InitialFluxes{No_InitialFluxes},
                for No_PhaseAngle = 1:length(PhaseAngles)
                    % ��Դ���Σ���բ��
                    set_param('SympatheticCurrent_Series/Em','PhaseAngle',PhaseAngles{No_PhaseAngle});                     
                    for No_Breaker = 1:2 %����һ��ͬ�ں�բ��һ�ַ�ͬ�ں�բ
                        % �Ƿ�ͬ�ں�բ
                        if No_Breaker == 1 % ͬ�ں�բ
                            set_param('SympatheticCurrent_Series/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Series/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Series/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{1});
                            FileName = strcat('��Ӧӿ��_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_','����_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','ͬ��');
                        else % ��ͬ�ں�բ
                            set_param('SympatheticCurrent_Series/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Series/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{2});
                            set_param('SympatheticCurrent_Series/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{3});
                            FileName = strcat('��Ӧӿ��_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_','����_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','��ͬ��');
                        end
                        % ���з���ģ��
                        try
                            sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ35/220kV
                        catch
                            fprintf(strcat('Failure:',FileName));
                            continue;
                        end
                        % Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                        Data1 = importdata('Iabc_1.mat'); % �洢����������txt��
                        Data2 = importdata('Iabc_2.mat');
                        FullName = strcat(SavePath,FileName,'.txt');
                        select=(Data1.Time>=1.0); % ��ȡ��һ�����ݣ���ǰһ�������ǿ�Ͷӿ������
                        Signal = [Data1.Time(select)-1 Data1.Data(select,:) Data2.Data(select,:)]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                        dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                        flag = flag+1;
                        if view
                            hf=figure;
                            subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                            subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                            
                            subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
                            subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                             
                            bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                            saveas(gcf,strrep(FullName,'.txt','.png'));
                            set(hf,'Visible','off');close all;
                        end
                        % ��ӡ������
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

if 0 % ������Ӧ
    fprintf('Start simulation for SympatheticCurrent_Parallel...\n');
    SwitchingTimes.Breaker = {'[1.1 2]','[1.2 2]','[1.3 2]'}; % ����һ��ͬ�ں�բ��һ�ַ�ͬ�ں�բ
    
    Total = length(Winding1Connections)*length(Winding2Connections)*length(InitialFluxes)*length(PhaseAngles)*2;
    RootPath = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\2019\';
    Model = 'SympatheticCurrent_Parallel.slx';
    CaseName = '��Ӧӿ��';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    open_system(Model);
    
    flag = 0;
    for No_Winding1Connection = 1:length(Winding1Connections)
        for No_Winding2Connection = 1:length(Winding2Connections)
            for No_InitialFluxes = 1:length(InitialFluxes)
                % ��ѹ�����Σ����߷�ʽ��ʣ��
                set_param('SympatheticCurrent_Parallel/Transformer1','InitialFluxes',InitialFluxes{No_InitialFluxes},...
                    'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                    'Winding2Connection',Winding2Connections{No_Winding2Connection});
                set_param('SympatheticCurrent_Parallel/Transformer2',...
                    'Winding1Connection',Winding1Connections{No_Winding1Connection},...
                    'Winding2Connection',Winding2Connections{No_Winding2Connection});                        
                for No_PhaseAngle = 1:length(PhaseAngles)
                    % ��Դ���Σ���բ��
                    set_param('SympatheticCurrent_Parallel/Em','PhaseAngle',PhaseAngles{No_PhaseAngle});                        
                    for No_Breaker = 1:2 % ����һ��ͬ�ں�բ��һ�ַ�ͬ�ں�բ
                        if No_Breaker == 1 % ͬ�ں�բ
                            set_param('SympatheticCurrent_Parallel/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Parallel/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Parallel/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{1});
                            FileName = strcat('��Ӧӿ��_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_','����_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','ͬ��');
                        else % ��ͬ�ں�բ
                            set_param('SympatheticCurrent_Parallel/BreakerA','SwitchingTimes',SwitchingTimes.Breaker{1});
                            set_param('SympatheticCurrent_Parallel/BreakerB','SwitchingTimes',SwitchingTimes.Breaker{2});
                            set_param('SympatheticCurrent_Parallel/BreakerC','SwitchingTimes',SwitchingTimes.Breaker{3});
                            FileName = strcat('��Ӧӿ��_',Winding1Connections{No_Winding1Connection},Winding2Connections{No_Winding2Connection},...
                                '_','����_',InitialFluxes{No_InitialFluxes},'pu_',...
                                PhaseAngles{No_PhaseAngle},'deg_','��ͬ��');
                        end
                        % ���з���ģ��
                        try
                            sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ35/220kV
                        catch
                            fprintf(strcat('Failure:',FileName));
                            continue;
                        end
                        
                        % Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                        Data1 = importdata('Iabc_1.mat'); % �洢����������txt��
                        Data2 = importdata('Iabc_2.mat');
                        FullName = strcat(SavePath,FileName,'.txt');
                        select=(Data1.Time>=1.0);
                        Signal = [Data1.Time(select)-1 Data1.Data(select,:) Data2.Data(select,:)]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                        dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                        flag = flag+1;
                        if view
                            hf=figure;
                            subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                            subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                            
                            subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
                            subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                             
                            bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                            saveas(gcf,strrep(FullName,'.txt','.png'));
                            set(hf,'Visible','off');close all;
                        end
                        % ��ӡ������
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

% ********** �ڲ�����(�������߹��Ϻ��������) ********** %
% Ӱ�����أ����ϳ���ʱ�䡢�������͡�����λ�õ�
if 0 % �ڲ�����_���߹���
    fprintf('Start simulation for InternalFault_Lead...\n');
    FaultTypes = {{'on','off','off','on'},{'off','on','off','on'},{'off','off','on','on'},...
        {'on','on','off','on'},{'off','on','on','on'},{'on','off','on','on'},{'on','on','on','on'},...
        {'on','on','off','off'},{'off','on','on','off'},{'on','off','on','off'}};
    FaultNames = {'AN','BN','CN','ABN','BCN','ACN','ABCN','AB','BC','AC'};
    SwitchTimes.Fault = {'[0.3 0.34]','[0.3 0.36]','[0.3 0.38]',...
        '[0.31 0.35]','[0.31 0.37]','[0.31 0.39]'}; % ���ù���ʱ�̺͹��ϳ���ʱ��
    
    RootPath = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\2019\';
    CaseName = '���߹���';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    Models = dir(strcat(RootPath,'InternalFaults_Lead_*.slx'));
    Total = length(FaultTypes)*length(SwitchTimes.Fault)*2*length(Models);
    
    flag = 0;
    for No_Model = 1:length(Models)
        Model = Models(No_Model).name; % ѡ��ĳ��ģ��
        open_system(Model);
        for No_FaultType = 1:length(FaultTypes)
            for No_SwitchTime = 1:length(SwitchTimes.Fault)
                for No_FaultPosition =1:2 % һ�β���ϻ��Ƕ��β����
                    if No_FaultPosition == 1 % һ�β����߹���
                        % ���ϵ��Σ��������͡�����ʱ��
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),...
                            'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                            'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                            'SwitchTimes',SwitchTimes.Fault{No_SwitchTime});
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),'SwitchTimes','[10 11]');
                        % ���ݹ���λ���޸�
                        PositionName = 'һ�β�';
                    else
                        % ���ϵ��Σ��������͡�����ʱ��
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),...
                            'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                            'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                            'SwitchTimes',SwitchTimes.Fault{No_SwitchTime});
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),'SwitchTimes','[10 11]');
                        % ���ݹ���λ���޸�
                        PositionName = '���β�';
                    end
                    
                    [str1,str2] = strsplit(Models(No_Model).name,'Lead_');
                    WindConnection = strrep(str1{2},'.slx','');
                    FileName = strcat('���߹���_',WindConnection,'_',PositionName,'_',FaultNames{No_FaultType},'_',SwitchTimes.Fault{No_SwitchTime},'s');
                    % ���з���ģ��
                    try
                        sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                    Data1 = importdata('Iabc_1.mat'); % �洢����������txt��
                    Data2 = importdata('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                        
                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                        saveas(gcf,strrep(FullName,'.txt','.png'));
                        set(hf,'Visible','off');close all;
                    end
                    % ��ӡ������
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
% Ӱ�����أ����ϳ���ʱ�䡢�������͡�����λ�õ�
if 0 % �ڲ�����_�������_�ѵ�  �ڲ�����_�������_���
    try
        delete('Iabc_1.mat');delete('Iabc_2.mat');
    catch
        fprintf('No output files currently');
    end
    
    view = 0;
    fprintf('Start simulation for InternalFault_Winding...\n');
    FaultTypes = {{'on','off','on','off'}};
    FaultNames = {'AC'};
    SwitchTimes.Fault = {'[0.3 0.36]','[0.31 0.39]'}; % ���ù���ʱ�̺͹��ϳ���ʱ��
    
    RootPath = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\2019\';
    CaseName = '�������';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    Models = dir(strcat(RootPath,'InternalFaults_Winding_*.slx'));
    Total = length(FaultTypes)*length(SwitchTimes.Fault)*2*length(Models);
    
    flag = 0;
    for No_Model = 1:length(Models)
        Model = Models(No_Model).name; % ѡ��ĳ��ģ��
        open_system(Model);
        for No_FaultType = 1:length(FaultTypes)
            for No_SwitchTime = 1:length(SwitchTimes.Fault)
                for No_FaultPosition =1:2 % һ�β���ϻ��Ƕ��β����
                    if No_FaultPosition == 1 % һ�β��������
                        % ���ϵ��Σ��������͡�����ʱ��
                        % һ�β���ؿ��ƿ���״̬
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','close','SwitchA','on',...
                            'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % ģ��һ�β���Ȧ��ģ��
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),...
                            'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                            'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                            'SwitchTimes',SwitchTimes.Fault{No_SwitchTime});
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        % ���β���ؿ��ƿ���״̬
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','open','SwitchA','off',...
                            'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % ģ����β���Ȧ��ģ��
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),'SwitchTimes','[10 11]');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        % ���ݹ���λ���޸�
                        PositionName = 'һ�β�40%A50%C';
                    else
                        % ���ϵ��Σ��������͡�����ʱ��
                        % ���β���ؿ��ƿ���״̬
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','close','SwitchA','on',...
                            'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % ģ����β���Ȧ��ģ��
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),...
                            'FaultA',FaultTypes{No_FaultType}{1},'FaultB',FaultTypes{No_FaultType}{2},...
                            'FaultC',FaultTypes{No_FaultType}{3},'GroundFault',FaultTypes{No_FaultType}{4},...
                            'SwitchTimes',SwitchTimes.Fault{No_SwitchTime});
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        % һ�β���ؿ��ƿ���״̬
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','open','SwitchA','off',...
                            'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % ģ��һ�β���Ȧ��ģ��
                        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),'SwitchTimes','[10 11]');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        % ���ݹ���λ���޸�
                        PositionName = '���β�50%A40%C';
                    end
                    
                    [str1,str2] = strsplit(Models(No_Model).name,'Winding_');
                    WindConnection = strrep(str1{2},'.slx','');
                    FileName = strcat('����������_',PositionName,'_',WindConnection,'_',FaultNames{No_FaultType},'_',SwitchTimes.Fault{No_SwitchTime},'s');
                    % ���з���ģ��
                    try
                        sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                    Data1 = importdata('Iabc_1.mat'); % �洢����������txt��
                    Data2 = importdata('Iabc_2.mat');
                    delete('Iabc_1.mat');delete('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                        
                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                        saveas(gcf,strrep(FullName,'.txt','.png'));
                        set(hf,'Visible','on');%close all;
                    end
                    % ��ӡ������
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


% Ӱ�����أ����ϳ���ʱ�䡢�������͡�����λ�õ�
if 0 % �ڲ�����_�������_�Ѽ�_�����Ѽ�
    try
        delete('Iabc_1.mat');delete('Iabc_2.mat');
    catch
        fprintf('No output files currently');
    end
    
    view = 0;
    fprintf('Start simulation for InternalFault_Winding...\n');
    SwitchTimes.Fault = {'[0.3 0.36]','[0.31 0.39]'}; % ���ù���ʱ�̺͹��ϳ���ʱ��
    
    RootPath = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\2019\';
    CaseName = '�������';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    Models = dir(strcat(RootPath,'InternalFaults_Winding_*.slx'));
    Total = 3*2*length(Models);
    
    flag = 0;
    for No_Model = 1:length(Models)
        Model = Models(No_Model).name; % ѡ��ĳ��ģ��
        open_system(Model);
        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),'SwitchTimes','[10 11]'); % �������ѵغ��Ѽ����
        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),'SwitchTimes','[10 11]');
        for No_FaultPosition =1:2 % һ�β���ϻ��Ƕ��β����
            if No_FaultPosition == 1 % һ�β��������
                % ���ϵ��Σ��������͡�����ʱ��
                % һ�β���ؿ��ƿ���״̬
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','close','SwitchA','on',...
                    'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % ģ��һ�β���Ȧ��ģ��
                
                % ���β���ؿ��ƿ���״̬
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','open','SwitchA','off',...
                    'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % ģ����β���Ȧ��ģ��
                
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                
                for NoCase=1:3
                    if NoCase==1
                        % ���ݹ���λ���޸�
                        PositionName = 'һ�β�80%A';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % �Ѽ����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                    elseif NoCase==2
                        PositionName = 'һ�β�80%B';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                    elseif NoCase==3
                        PositionName = 'һ�β�80%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                    end
                    [str1,str2] = strsplit(Models(No_Model).name,'Winding_');
                    WindConnection = strrep(str1{2},'.slx','');
                    FileName = strcat('��������Ѽ�_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},'s');
                    % ���з���ģ��
                    try
                        sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                    Data1 = importdata('Iabc_1.mat'); % �洢����������txt��
                    Data2 = importdata('Iabc_2.mat');
                    delete('Iabc_1.mat');delete('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                        
                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                        saveas(gcf,strrep(FullName,'.txt','.png'));
                        set(hf,'Visible','off');close all;
                    end
                end
            else
                % ���ϵ��Σ��������͡�����ʱ��
                % ���β���ؿ��ƿ���״̬
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','close','SwitchA','on',...
                    'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % ģ����β���Ȧ��ģ��
                % һ�β���ؿ��ƿ���״̬
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','open','SwitchA','off',...
                    'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % ģ��һ�β���Ȧ��ģ��
                
                
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                
                for NoCase=1:3
                    if NoCase==1
                        % ���ݹ���λ���޸�
                        PositionName = '���β�80%A';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                    elseif NoCase==2
                        % ���ݹ���λ���޸�
                        PositionName = '���β�80%B';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                    elseif NoCase==3
                        % ���ݹ���λ���޸�
                        PositionName = '���β�80%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                    end
                    [str1,str2] = strsplit(Models(No_Model).name,'Winding_');
                    WindConnection = strrep(str1{2},'.slx','');
                    FileName = strcat('��������Ѽ�_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},'s');
                    % ���з���ģ��
                    try
                        sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                    Data1 = importdata('Iabc_1.mat'); % �洢����������txt��
                    Data2 = importdata('Iabc_2.mat');
                    delete('Iabc_1.mat');delete('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
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
    % ��ӡ������
    if mod(flag,1000) == 0
        percentage=sprintf('%.2f%%', flag/Total*100);
        disp(['Completed ',percentage]);
    end
end

if 0 % �ڲ�����_�������_�Ѽ�_�����Ѽ�
    try
        delete('Iabc_1.mat');delete('Iabc_2.mat');
    catch
        fprintf('No output files currently');
    end
    
    view = 0;
    fprintf('Start simulation for InternalFault_Winding...\n');
    SwitchTimes.Fault = {'[0.3 0.36]','[0.31 0.39]','[0.41 0.49]'}; % ���ù���ʱ�̺͹��ϳ���ʱ��
    
    RootPath = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\2019\';
    CaseName = '�������';
    [SUCCESS,MESSAGE,MESSAGEID] = mkdir(CaseName);
    SavePath = strcat(RootPath,CaseName,'\');
    Models = dir(strcat(RootPath,'InternalFaults_Winding_*.slx'));
    Total = 4*length(Models);
    
    flag = 0;
    for No_Model = 1:length(Models)
        Model = Models(No_Model).name; % ѡ��ĳ��ģ��
        open_system(Model);
        set_param(strcat(strrep(Model,'.slx',''),'/Fault_S'),'SwitchTimes','[10 11]'); % �������ѵغ��Ѽ����
        set_param(strcat(strrep(Model,'.slx',''),'/Fault_T'),'SwitchTimes','[10 11]');
        for No_FaultPosition =1:2 % һ�β���ϻ��Ƕ��β����
            if No_FaultPosition == 1 % һ�β��������
                % ���ϵ��Σ��������͡�����ʱ��
                % һ�β���ؿ��ƿ���״̬
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','close','SwitchA','on',...
                    'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % ģ��һ�β���Ȧ��ģ��
                
                % ���β���ؿ��ƿ���״̬
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','open','SwitchA','off',...
                    'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % ģ����β���Ȧ��ģ��
                
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                
                [str1,str2] = strsplit(Models(No_Model).name,'Winding_');
                WindConnection = strrep(str1{2},'.slx','');
                for NoCase=1:4
                    if NoCase==1 % AB����
                        % ���ݹ���λ���޸�
                        PositionName = 'һ�β�60%A80%B';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % �Ѽ����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        FileName = strcat('��������Ѽ�_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},'s');
                    elseif NoCase==2 % BC����
                        PositionName = 'һ�β�80%B70%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        FileName = strcat('��������Ѽ�_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},'s');
                    elseif NoCase==3 % AC����
                        PositionName = 'һ�β�60%A70%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % �Ѽ����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{3},'External','off');
                        FileName = strcat('��������Ѽ�_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},SwitchTimes.Fault{3},'s');
                    elseif NoCase==4 % ABC����
                        PositionName = 'һ�β�60%A80%B70%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % �Ѽ����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{2},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{3},'External','off');
                        FileName = strcat('��������Ѽ�_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},SwitchTimes.Fault{2},SwitchTimes.Fault{3},'s');
                    end
                    
                    
                    % ���з���ģ��
                    try
                        sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                    Data1 = importdata('Iabc_1.mat'); % �洢����������txt��
                    Data2 = importdata('Iabc_2.mat');
                    delete('Iabc_1.mat');delete('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
                        subplot(224);plot(Signal(:,1),Signal(:,5:7)-Signal(:,[6,7,5]));legend('Iab','Ibc','Ica'); set(gca,'YAxisLocation','right');
                        
                        bb=strrep(FileName,'_','\_');suptitle(bb);clear bb;
                        %saveas(gcf,strrep(FullName,'.txt','.png'));
                        set(hf,'Visible','on');%close all;
                    end
                end
            else
                % ���ϵ��Σ��������͡�����ʱ��
                % ���β���ؿ��ƿ���״̬
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker1'),'InitialState','close','SwitchA','on',...
                    'SwitchB','on','SwitchC','on','SwitchTimes','[10 11]','External','off'); % ģ����β���Ȧ��ģ��
                % һ�β���ؿ��ƿ���״̬
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2'),'InitialState','open','SwitchA','off',...
                    'SwitchB','off','SwitchC','off','SwitchTimes','[10 11]','External','off'); % ģ��һ�β���Ȧ��ģ��
                
                
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                set_param(strcat(strrep(Model,'.slx',''),'/Breaker2_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                
                [str1,str2] = strsplit(Models(No_Model).name,'Winding_');
                WindConnection = strrep(str1{2},'.slx','');
                
                for NoCase=1:4
                    if NoCase==1 % AB����
                        % ���ݹ���λ���޸�
                        PositionName = '���β�70%A80%B';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{2},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        FileName = strcat('��������Ѽ�_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},SwitchTimes.Fault{2},'s');
                    elseif NoCase==2 % BC����
                        % ���ݹ���λ���޸�
                        PositionName = '���β�80%B60%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes','[10 11]','External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{2},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{3},'External','off');
                        FileName = strcat('��������Ѽ�_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{2},SwitchTimes.Fault{3},'s');
                    elseif NoCase==3 % AC����
                        % ���ݹ���λ���޸�
                        PositionName = '���β�70%A60%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes','[10 11]','External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off');
                        FileName = strcat('��������Ѽ�_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},'s');
                    elseif NoCase==4 % ABC����
                        % ���ݹ���λ���޸�
                        PositionName = '���β�70%A80%B60%C';
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_A'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{1},'External','off'); % �Ѽ䲻����
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_B'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{2},'External','off');
                        set_param(strcat(strrep(Model,'.slx',''),'/Breaker1_C'),'InitialState','0','SwitchingTimes',SwitchTimes.Fault{3},'External','off');
                        FileName = strcat('��������Ѽ�_',PositionName,'_',WindConnection,'_',SwitchTimes.Fault{1},SwitchTimes.Fault{2},SwitchTimes.Fault{3},'s');
                    end
                    
                    
                    % ���з���ģ��
                    try
                        sim(strcat(RootPath,Model)); % ��ǰģ�����õ�ѹ�ȼ�Ϊ35/220kV
                    catch
                        fprintf(strcat('Failure:',FileName));
                        continue;
                    end
                    % Data1Ϊһ�β����ݣ�Data2Ϊ���β�����
                    Data1 = importdata('Iabc_1.mat'); % �洢����������txt��
                    Data2 = importdata('Iabc_2.mat');
                    delete('Iabc_1.mat');delete('Iabc_2.mat');
                    FullName = strcat(SavePath,FileName,'.txt');
                    Signal = [Data1.Time Data1.Data Data2.Data]; % ʱ�䡢һ�β�Iabc�����β�Iabc
                    dlmwrite(FullName,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
                    flag = flag+1;
                    if view
                        hf=figure;
                        subplot(221);plot(Signal(:,1),Signal(:,2:4));legend('Ia','Ib','Ic');ylabel('һ�β�/pu');
                        subplot(222);plot(Signal(:,1),Signal(:,2:4)-Signal(:,[3,4,2]));legend('Iab','Ibc','Ica');set(gca,'YAxisLocation','right');
                        
                        subplot(223);plot(Signal(:,1),Signal(:,5:7));legend('Ia','Ib','Ic');ylabel('���β�/pu');
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
    % ��ӡ������
    if mod(flag,1000) == 0
        percentage=sprintf('%.2f%%', flag/Total*100);
        disp(['Completed ',percentage]);
    end
end

%% ����
if 0
    model = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\ExciterFlowCurrent_NoLoadSwitch.slx';
    sim(model);
    Data = importdata('Iabc.mat');
    filename='E:\������Ŀ\����ӿ��ʶ��\����\bsh\test.txt';
    Signal = [Data.Time Data.Data];
    dlmwrite(filename,Signal,'delimiter',' '); % �������н����txt�ļ���   ,'precision',6
end

if 0
    Addr = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\�ڲ�����\';
    ToAddr = 'E:\������Ŀ\����ӿ��ʶ��\����\bsh\ML����\';
    Lists = dir(strcat(Addr,'*.txt'));
    for i=1:length(Lists)
        filename = Lists(i).name;
        SOURCE = strcat(Addr,filename);
        DESTINATION = strcat(ToAddr,'�ڲ�����_',filename);
        [SUCCESS,MESSAGE,MESSAGEID] = copyfile(SOURCE,DESTINATION,'f');
    end
end

toc;