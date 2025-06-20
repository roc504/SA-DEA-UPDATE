% 获取相对路径
filePath1 = fullfile('code', 'SADEA_1best1.csv');
data=readmatrix(filePath1);%test_curve6%
x=data(:,1);
y1=data(:,2);
filePath2 = fullfile('code', 'BDE_res4031.csv');
data2=readmatrix(filePath2);
y2=data2(:,2);
filePath3 = fullfile('code', 'GA_res11.csv');
data3=readmatrix(filePath3);
y3=data3(:,2);
filePath4 = fullfile('code', 'BDE_res02241.csv');
data4=readmatrix(filePath4);
y4=data4(:,2);
% 颜色设置
color1 = [31, 119, 180] / 255; % 深蓝色 #1f77b4
color2 = [255, 127, 14] / 255;  % 橙色 #ff7f0e
color3 = [44, 160, 44] / 255;   % 绿色 #2ca02c
color4 =[0.4940, 0.1840, 0.5560];
figure;
hold on;
plot(x, y1, 'Color', color1, 'LineWidth', 1.5, 'MarkerFaceColor', color1);
plot(x, y2, 'Color', color2, 'LineWidth',1.5, 'MarkerFaceColor', color2);
plot(x, y3, 'Color', color3, 'LineWidth', 1.5, 'MarkerFaceColor', color3);
plot(x, y4, 'Color', color4, 'LineWidth', 1.5, 'MarkerFaceColor', color4);
% plot(x,y1*100+1,'r-','LineWidth',1);
% plot(x,y2*100,'b-','LineWidth',1);
% plot(x,y3*100,'black-','LineWidth',1);
hold off;

legend({'SA-DEA','GA','PSO','DE'},'FontName','Microsft YaHei UI');
xlabel({'Times'},'FontName','Microsft YaHei UI');
ylabel({'Fitness'},'FontName','Microsft YaHei UI');