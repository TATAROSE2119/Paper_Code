clc;clear;
%% 1.导入数据
%产生训练数据
num_sample=100;
a=10*randn(num_sample,1);
x1=a+randn(num_sample,1);
x2=1*sin(a)+randn(num_sample,1);
x3=5*cos(5*a)+randn(num_sample,1);
x4=0.8*x2+0.1*x3+randn(num_sample,1);
xx_train=[x1,x2,x3,x4];

% 产生测试数据
a=10*randn(num_sample,1);
x1=a+randn(num_sample,1);
x2=1*sin(a)+randn(num_sample,1);
x3=5*cos(5*a)+randn(num_sample,1);
x4=0.8*x2+0.1*x3+randn(num_sample,1);
xx_test=[x1,x2,x3,x4];
xx_test(51:100,2)=xx_test(51:100,2)+15*ones(50,1);

%% 2.数据处理
Xtrain=xx_train;
Xtest=xx_test;
X_mean=mean(Xtrain);
X_std=std(Xtrain);
[X_row, X_col]=size(Xtrain);
Xtrain=(Xtrain-repmat(X_mean,X_row,1))./repmat(X_std,X_row,1); %标准化处理

%% 3.PCA降维
SXtrain = cov(Xtrain);%求协方差矩阵
[T,lm]=eig(SXtrain);%求特征值及特征向量,特征值排列顺序为从小到大
D=flipud(diag(lm));%将特征值从大到小排列
% 确定降维后的数量
num=1;
while sum(D(1:num))/sum(D)<0.85
    num = num+1;
end
P = T(:,X_col-num+1:X_col); %取对应的向量
P_=fliplr(P); %特征向量由大到小排列


%% 4.计算T2和Q的限值
%求置信度为99%时的T2统计控制限,T=k*(n^2-1)/n(n-k)*F(k,n-k)
%其中k对应num,n对应X_row
T2UCL1=num*(X_row-1)*(X_row+1)*finv(0.99,num,X_row - num)/(X_row*(X_row - num));%求置信度为99%时的T2统计控制限 

%求置信度为99%的Q统计控制限
for i = 1:3
    th(i) = sum((D(num+1:X_col)).^i);
end
h0 = 1 - 2*th(1)*th(3)/(3*th(2)^2);
ca = norminv(0.99,0,1);
QU = th(1)*(h0*ca*sqrt(2*th(2))/th(1) + 1 + th(2)*h0*(h0 - 1)/th(1)^2)^(1/h0); %置信度为99%的Q统计控制限 

%% 5.模型测试
n = size(Xtest,1);
Xtest=(Xtest-repmat(X_mean,n,1))./repmat(X_std,n,1);%标准化处理
%求T2统计量，Q统计量
[r,y] = size(P*P');
I = eye(r,y); 
T2 = zeros(n,1);
Q = zeros(n,1);
lm_=fliplr(flipud(lm));
%T2的计算公式Xtest.T*P_*inv(S)*P_*Xtest
for i = 1:n
    T2(i)=Xtest(i,:)*P_*inv(lm_(1:num,1:num))*P_'*Xtest(i,:)';    
    Q(i) = Xtest(i,:)*(I - P*P')*Xtest(i,:)';                                                                                    
end

%% 6.绘制T2和SPE图
figure('Name','PCA');
subplot(2,1,1);
plot(1:i,T2(1:i),'k');
hold on;
plot(i:n,T2(i:n),'k');
title('统计量变化图');
xlabel('采样数');
ylabel('T2');
hold on;
line([0,n],[T2UCL1,T2UCL1],'LineStyle','--','Color','r');

subplot(2,1,2);
plot(1:i,Q(1:i),'k');
hold on;
plot(i:n,Q(i:n),'k');
title('统计量变化图');
xlabel('采样数');
ylabel('SPE');
hold on;
line([0,n],[QU,QU],'LineStyle','--','Color','r');

%% 7.绘制贡献图
%7.1.确定造成失控状态的得分
S = Xtest(51,:)*P(:,1:num);
r = [ ];
for i = 1:num
    if S(i)^2/lm_(i) > T2UCL1/num
        r = cat(2,r,i);
    end
end
%7.2.计算每个变量相对于上述失控得分的贡献
cont = zeros(length(r),X_col);
for i = length(r)
    for j = 1:X_col
        cont(i,j) = abs(S(i)/D(i)*P(j,i)*Xtest(51,j));
    end
end
%7.3.计算每个变量的总贡献
CONTJ = zeros(X_col,1);
for j = 1:X_col
    CONTJ(j) = sum(cont(:,j));
end
%7.4.计算每个变量对Q的贡献
e = Xtest(51,:)*(I - P*P');%选取第60个样本来检测哪个变量出现问题。
contq = e.^2;
%5. 绘制贡献图
figure
subplot(2,1,1);
bar(contq,'g');
xlabel('变量号');
ylabel('SPE贡献率 %');
hold on;
subplot(2,1,2);
bar(CONTJ,'r');
xlabel('变量号');
ylabel('T^2贡献率 %');