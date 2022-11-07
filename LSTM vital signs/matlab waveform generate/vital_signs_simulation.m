clear;
clc;
file_num=50;   %%生成文件总数
folder_path='D:\program\matlab\bin\dataset\';
rmdir(folder_path, 's');
mkdir(folder_path);
for i=1:file_num
    dt=0.01;    %%时间间隔/s
    T=10;       %%仿真总时长/s
    Fb=0.1+0.4*rand;
    Fh=0.8+1.2*rand;
    Ah=0.23;
    Ab=1;
    t=[0:dt:T];
    t(1)=[];
    phi0_h=2*pi*rand();
    phi0_b=2*pi*rand();
    s_t=Ah*sin(2*pi*Fh*t+phi0_h)+Ab*sin(2*pi*Fb*t+phi0_b);
    s_t=s_t+noise(s_t);


    %%plot
    plot(t,s_t);
    grid on 
    xlabel('Time/s')
    ylabel('Amplitude/mm')
    title('Vital Signs Simulation')

    %%save to file
    fid=fopen([[folder_path],[num2str(i)],['.txt']],'w');
    fprintf(fid,'%.3f ',Fb);
    fprintf(fid,'%.3f ',Fh);
    for j=1:length(s_t)
        fprintf(fid,'%.3f ',s_t(j));
    end
    fclose(fid);
end
