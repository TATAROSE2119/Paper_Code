clear
close
clc

%% ����ʯ�� fault4
fault_point=181;
fault=1;
Data=load('C:\Paper_code\matlab\PCA\group1\group1\test1.mat');  %% ���ϵ���data_sample�ĵ�2580����
data_sample=Data.Data;
X_train=cell2mat(data_sample(1441:2400,[82 83 164 165 213 214 215 248 249 175 176 177 178 210 211 274 275]));
X_test=cell2mat(data_sample(2401:2880,[82 83 164 165 213 214 215 248 249 175 176 177 178 210 211 274 275]));
train_norm=(X_train-min(X_train))./(max(X_train)-min(X_train));
X_test=(X_test-min(X_train))./(max(X_train)-min(X_train));
t=16;
%% FAULT1
% ��������2023-01-14 17:00:07.000��¼��·FCC3_FT-1108������λ�ã��������������������Ŵ��������������ϣ����ɲ��������Ǳ���ѹ�ܽ�ͷ��©�����˲���ͱ���
% �������ڣ�1min
% ���Ϸ����ڷ�Ӧ����װ��
% ����ѡ��:12 13 20 21 22 23 24 25 52 53 90 91 205 238 239 272 273 36 37
% fault_point=181;
% fault=2;
% Data=load('���״߻�/����1/Data.mat');  %% ���ϵ���data_sample�ĵ�2580����
% data_sample=Data.Data;
% X_train=cell2mat(data_sample(2881:3720,[12 13 20 21 22 23 24 25 52 53 90 91 205 238 239 272 273 36 37]));
% X_test=cell2mat(data_sample(3721:4020,[12 13 20 21 22 23 24 25 52 53 90 91 205 238 239 272 273 36 37]));
% [train_num,dim]=size(X_train);
% train_norm=(X_train-min(X_train))./(max(X_train)-min(X_train));
% X_test=(X_test-min(X_train))./(max(X_train)-min(X_train));
% t=14;
q=3;
[num,dim]=size(train_norm);
train_block=cell(1,q);
train_data=zeros(num-q+1,dim*q); %%q��ʾ��̬ϵͳ�Ľ״�
    for k=1:q
        train_block{1,k}=train_norm(k:num-(q-k),:); 
        train_data(:,(k-1)*dim+1:k*dim)=train_block{1,k};
    end
[~,Xmean,Xstd] = zscore(train_data);
[num_train,dim_train]=size(train_data);
Xtrain=(train_data-repmat(Xmean,num_train,1))./repmat(Xstd,num_train,1);
% [L,S]=rpca(Xtrain);
lambda = 1 / sqrt(max(size(Xtrain))); % ���򻯲���
tol = 1e-4; % �����ݲ�
max_iter = 1000; % ����������
% ���ģ��
[L, S] = RobustPCA_GPT(Xtrain, lambda, tol, max_iter);
%%
Xstandard=Xtrain;
control_level=0.9999;
[U,D_t,V]=svd(L);
for j=1:dim*q
   if(D_t(j,j)<=1e-10)
      num_pc=j; 
   end
end
W=D_t(1:num_pc,1:num_pc)*D_t(1:num_pc,1:num_pc);
P=V(:,1:num_pc);
T2_limit = zeros(num_train,1);
Q_limit = zeros(num_train,1);  
I_Q=eye(size(P,1));
for i = 1:num_train
    T2_limit(i)=(Xstandard(i,:)*(P/W*P')*Xstandard(i,:)');  
%   Q_limit(i) = norm((I_Q - (P*P'))*U_standard(i,:)','fro');  
    Q_limit(i) = Xstandard(i,:)*(I_Q-P*P')*(I_Q-P*P')*Xstandard(i,:)';
end
T2UCL1 = ksdensity(T2_limit,control_level,'function','icdf');
QUCL = ksdensity(Q_limit,control_level,'function','icdf');
%% ���߼��
testdata_orig=X_test;
% test_original=X_test;
[num_test_orig,~]=size(testdata_orig);
test_block=cell(1,q);
test_data=zeros(num_test_orig-q+1,dim*q); %%q��ʾ��̬ϵͳ�Ľ״�
for k=1:q
    test_block{1,k}=testdata_orig(k:num_test_orig-(q-k),:); 
    test_data(:,(k-1)*dim+1:k*dim)=test_block{1,k};
end
[num_test,dim_test]=size(test_data);
Xtest=(test_data-repmat(Xmean,num_test,1))./repmat(Xstd,num_test,1);
Xtest_standard=Xtest;
[r,y]=size(P*P');
I=eye(r,y);
T2=zeros(num_test,1);
Q=zeros(num_test,1);
%         u_new=Xtest_standard(i,:)*V'/(D_t(1:dim*q,1:dim*q)+tao*eye(dim*q));
%         t_new=u_new
for i=1:num_test
   T2(i)=Xtest_standard(i,:)*P/W*P'*Xtest_standard(i,:)';
   Q(i)=Xtest_standard(i,:)*(I-P*P')*(I-P*P')*Xtest_standard(i,:)';
end      % 
%%����ͳ����������� 
    figure;
    tsubplot(2,1,1);
    plot(log(T2),'linewidth',1.5,'color','b');%�����߿�
    hold on
    % plot([fault_point,fault_point],[min(((S2)))-0.1,max(((S2)))+0.1],'r-.','linewidth',1.5);%����
    % hold on
    plot(repmat((log(T2UCL1)),1,size(T2,1)),'r-.','linewidth',1.5);  %%%������Ϊ��ɫ
    xlabel('Sample number');
    ylabel('log(T^2)');
    legend('log(T^2)','Control limit');
    hold on
    title('RPCA');
    set(gca,'FontName','Times New Roman','FontSize',16);
    tsubplot(2,1,2);
    plot(log(Q),'linewidth',1.5,'color','b');%�����߿�
    hold on
    % plot([fault_point,fault_point],[min(((NS_S2)))-0.1,max(((NS_S2)))+0.1],'r-.','linewidth',1.5);%����
    % hold on
    plot(repmat(log((QUCL)),1,size(Q,1)),'r-.','linewidth',1.5);  %%%������Ϊ��ɫ
    xlabel('Sample number');
    ylabel('log(Q)');
    legend('log(Q)','Control limit');
    hold on
    title('RPCA');
    set(gca,'FontName','Times New Roman','FontSize',16);
%%����FAR FDR
FDR_T2=length(find(T2(fault_point:end)>T2UCL1))/(length(T2)-fault_point+1);
FAR_T2=length(find(T2(1:fault_point-1)>T2UCL1))/(fault_point-1); 
FDR_Q=length(find(Q(fault_point:end)>QUCL))/(length(Q)-fault_point+1);
FAR_Q=length(find(Q(1:fault_point-1)>QUCL))/(fault_point-1);
FDR_T2=100*FDR_T2;
FAR_T2=100*FAR_T2;
FDR_Q=100*FDR_Q;
FAR_Q=100*FAR_Q;
save(['5RPCA_T2_Q_fault=',num2str(fault),'.mat'],'FDR_T2','FAR_T2','FDR_Q','FAR_Q','T2','T2UCL1','Q','QUCL');
