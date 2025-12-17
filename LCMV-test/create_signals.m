clear all
close all
load handel.mat
addpath('C:\project\fillline.m');
%% variens

clear all
%this is the radios of the table.

c_k = 340;                                        % Sound velocity (m/s)
c = 340; 
fs = 16000;
n = 4096;                                         % Number of samples
mtype = 'omnidirectional';                        % Type of microphone
order = -1;                                       % -1 equals maximum reflection order!
dim = 3;                                          % Room dimension
orientation = 0;                                  % Microphone orientation (rad)
hp_filter = 1;                                    % Disable high-pass filter
% define analysis parameters
NO_S_i=10;
lottery=2000;
R=1.3;
R_small = 1.2;
pad=25000;
start=1;
max1=100;
noise_R=0.2;
flag=0;
flag2=0;
nfft=2048; 
hop=1024;
num_jumps=9;
high=1;
angle=360;
distance_from_woll=0.5;
radius_mics = 0.1;
%% rir generation & covariens sounds 

close all
% define room dimension
L1_temp=randi([1,20]);
L1=4+0.1*L1_temp;
L2_temp=randi([1,20]);
L2=4+0.1*L2_temp;
L = [L1 L2 3];
room_x = L(1);
room_y = L(2);
SNR_difuse=20;
beta=0.3;
SIR = 0;

% calc snr
SNR_direction = 20;
SNR_mic = 30;
%% create circle & line & mic location % speaker location

distance_total=R+distance_from_woll+noise_R;
end_point_x=room_x-(R+distance_from_woll+noise_R);
end_point_y=room_y-(R+distance_from_woll+noise_R);
Radius_X = (end_point_x-distance_total).*rand + distance_total;
Radius_Y = (end_point_y-distance_total).*rand + distance_total;

R_angle=randi([1,angle]); % take  rand 180 degrees of circle to create rand orientation of Microphone array
t = linspace(-pi,2*pi,angle+angle/2); % 3 times 180 to create option to rand whole circ
t=t(R_angle:R_angle+angle/2-1);
x = R*sin(t)+Radius_X;
y = R*cos(t)+Radius_Y;
x_small = R_small*sin(t)+Radius_X;
y_small = R_small*cos(t)+Radius_Y;

z=0*t+high;
circ_mics_x = radius_mics*sin(t)+Radius_X;
circ_mics_y = radius_mics*cos(t)+Radius_Y;
[line_x,line_y]=fillline([x(1) y(1)], [x(angle/2) y(angle/2)],R*2*100);

r= [circ_mics_x(1) circ_mics_y(1) high; circ_mics_x(60) circ_mics_y(60) high;...
    circ_mics_x(120) circ_mics_y(120) high; circ_mics_x(180) circ_mics_y(180) high];

%% create locations   
    
center=[Radius_X Radius_Y];
start_circ_vec=[line_x(1) line_y(1)]-center;
labels_location = 5:10:175;
list_locations = [];
next_speech = 1;
while next_speech
    next_speech = 0;
    rand1 = randi(angle/2); 
    x1 = x(rand1);
    y1 = y(rand1);
    x1_temp=x1;
    y1_temp=y1;
    w=0.01*randi([1,314]);
    x1=x1+noise_R*sin(w);
    y1=y1+noise_R*cos(w);

    rand2 = randi(angle/2); 
    x2 = x(rand2);
    y2 = y(rand2);
    x2_temp=x2;
    y2_temp=y2;
    w=0.01*randi([1,314]);
    x2=x2+noise_R*sin(w);
    y2=y2+noise_R*cos(w);

    loc_xy = [x1,y1;x2,y2];
    dist = pdist(loc_xy,'euclidean');
    if dist<0.5
        next_speech = 1;
    end
end

s_first = [x1 ; y1 ; high];
s_second = [x2 ; y2 ; high];

%% create speakers clean

male_female_check=randi([1,2]);

if male_female_check==1
   male_female1='male'; 
   NO_S=112;
else
   male_female1='female'; 
   NO_S=56;
end

source=randi([1,NO_S]);
source1 = int2str(source);

male_female_check=randi([1,2]);

if male_female_check==1
   male_female2='male'; 
   NO_S=112;
else
   male_female2='female'; 
   NO_S=56;
end

source=randi([1,NO_S]);
source2 = int2str(source);    


path1 = strcat('C:\project\TIMIT CD\TIMIT\test\',male_female1,'\',male_female1,'_',source1);
path2 = strcat('C:\project\TIMIT CD\TIMIT\test\',male_female2,'\',male_female2,'_',source2);

f1=dir([path1 '\*.wav']);
f2=dir([path2 '\*.wav']);
folder1=f1.folder;
folder2=f2.folder;

file11=f1(3).name;
file12=f1(4).name;

file21=f1(5).name;
file22=f1(6).name;

source11 = strcat(folder1,'\',file11);
source12 = strcat(folder1,'\',file12);
source21 = strcat(folder1,'\',file21);
source22 = strcat(folder1,'\',file22);

speech_11_temp = audioread(source11);
speech_12_temp = audioread(source12);
speech_21_temp = audioread(source21);
speech_22_temp = audioread(source22);

speech_11=speech_11_temp(:,1);
speech_12=speech_12_temp(:,1);
speech_21=speech_21_temp(:,1);
speech_22=speech_22_temp(:,1);


pad_zeros1=zeros(1,fs)';
pad_zeros2=zeros(1,fs*2)';

in1 = [pad_zeros2 ; speech_11 ; pad_zeros1 ; zeros(1,length(speech_21))' ; pad_zeros1 ; speech_12];
vad1 = [pad_zeros2 ; ones(1,length(speech_11))' ; pad_zeros1 ; zeros(1,length(speech_21))' ; pad_zeros1 ; ones(1,length(speech_12))'];

in2 = [pad_zeros2 ; zeros(1,length(speech_11))' ; pad_zeros1 ; speech_21 ; pad_zeros1 ; speech_22];
vad2 = [pad_zeros2 ; zeros(1,length(speech_11))' ; pad_zeros1 ; ones(1,length(speech_21))' ; pad_zeros1 ; ones(1,length(speech_22))'];

maxlen = max(length(in1), length(in2));
in1(end+1:maxlen,:) = 0;
in2(end+1:maxlen,:) = 0;
vad1(end+1:maxlen,:) = 0;
vad2(end+1:maxlen,:) = 0;

in1 = in1.';
in2 = in2.';

vad1 = vad1.';
vad2 = vad2.';

vad = vad1+vad2;
 

%% create RIRs

h_first = rir_generator(c_k, fs, r, s_first', L, beta, n, mtype, order, dim, orientation, hp_filter);    
receiver_first = conv2(in1,h_first);

h_second = rir_generator(c_k, fs, r, s_second', L, beta, n, mtype, order, dim, orientation, hp_filter);    
receiver_second = conv2(in2,h_second);

%% create noise

middle = [Radius_X Radius_Y high];
s_noise = [Radius_X Radius_Y high];
d_noise = norm(s_noise-middle);
while d_noise<2
    x_noise=distance_from_woll+0.01*randi(100)*(room_x-2*distance_from_woll);
    y_noise=distance_from_woll+0.01*randi(100)*(room_y-2*distance_from_woll);
    s_noise = [x_noise y_noise high];
    d_noise = norm(s_noise-middle);
end

p=randi([1,5]);
noise_file=sprintf('C:\\project\\noise\\noise_%d.wav',p);
noise_temp = audioread(noise_file);
noise_temp = [noise_temp.' noise_temp.' noise_temp.'].';
noise_temp=noise_temp(1:length(receiver_first)-n+1)';
h_noise = rir_generator(c_k, fs, r, s_noise, L, beta, n, mtype, order, dim, orientation, hp_filter);
Receivers_noise = conv2(noise_temp,h_noise);    
Receivers_noise = Receivers_noise/mean(std(Receivers_noise'));
%% create input

M=size(receiver_first,1);
length_receives=size(receiver_first,2);
difuse_noise = create_difuse_noise(length_receives);
difuse_noise = difuse_noise/mean(std(difuse_noise));
receiver_first=receiver_first.'/max(max(abs(receiver_first.')));
receiver_second=receiver_second.'/max(max(abs(receiver_second.'))); 

A_x_before=(mean(std(receiver_first))+mean(std(receiver_second)))/2;
Ax_SIR = A_x_before/(10^(SIR/20));
receivers = Ax_SIR*receiver_first.'+receiver_second.';
%clac An
A_x=mean(std(receivers'));
std_n=mean(std(Receivers_noise'));
A_n_difuse = A_x/(10^(SNR_difuse/20));
A_n_diraction = A_x/(10^(SNR_direction/20));
A_n_mic = A_x/(10^(SNR_mic/20));

%create mic moise
mic_noise = A_n_mic*randn(M,length_receives);    
% create difuse noise
receivers=(receivers+mic_noise+A_n_difuse*difuse_noise.'+A_n_diraction*Receivers_noise).'; 
noise_only = (mic_noise+A_n_difuse*difuse_noise.'+A_n_diraction*Receivers_noise).';
receivers=receivers/max(max(abs(receivers)));     

path = sprintf('C:\\Users\\user\\Desktop\\master\\Thesis\\sharon\\pini\\');
first_clean_file=strcat(path,'first_clean.wav');
second_clean_file=strcat(path,'second_clean.wav');
first_reverb_file=strcat(path,'first_reverb.wav');
second_reverb_file=strcat(path,'second_reverb.wav');
data_noise=strcat(path,'noise.wav');

audiowrite(first_clean_file,in1,fs) 
audiowrite(second_clean_file,in2,fs) 
audiowrite(first_reverb_file,receiver_first,fs) 
audiowrite(second_reverb_file,receiver_second,fs) 
audiowrite(data_noise,noise_only,fs) 


