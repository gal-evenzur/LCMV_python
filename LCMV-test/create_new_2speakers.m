%% create signal for LCMV
clear all
dir = '//fs/lab-acoustic/WSJ1/Train/';
%noise_background_file = strcat(dir,'noise_floor.wav');
;speaker_a_1_file = strcat(dir,'48Z/48ZA010A.wav');
speaker_a_2_file = strcat(dir,'48Z/48ZA010B.wav');
speaker_b_1_file = strcat(dir,'49A/49AA010C.wav');
speaker_b_2_file = strcat(dir,'49A/49AA010D.wav');

%noise_background = audioread(noise_background_file)';

speaker_a_1 = audioread(speaker_a_1_file)';
speaker_a_2 =  audioread(speaker_a_2_file)';
speaker_b_1 = audioread(speaker_b_1_file)';
speaker_b_2 = audioread(speaker_b_2_file)';

%noise_background= noise_background(2:5,:);

speaker_a_1=speaker_a_1(2:5,:);
speaker_a_2=speaker_a_2(2:5,:);
speaker_b_1=speaker_b_1(2:5,:);
speaker_b_2=speaker_b_2(2:5,:);

%padding for same length
len=length(speaker_a_1);
len_all= 4*len;

%noise_air_cond=[noise_air_cond noise_air_cond];
%noise_air_cond=noise_air_cond(:,1:len_all);
%noise_background_a= noise_background(:,1:3*len);
%noise_background_a_pad=[noise_background_a zeros(4,len)];
%noise_background_b=noise_background(:,len_all+1:len_all+len);
%noise_background_b_pad=[noise_background_b zeros(4,3*len)];
%speaker_a_1_pad=[zeros(4,len) speaker_a_1 zeros(4,2*len)];
%speaker_a_2_pad=speaker_a_2(2:5,:);
speakers_pad=[zeros(4,len) speaker_a_1 speaker_b_1 speaker_a_2+speaker_b_2];
%speaker_b_2_pad=speaker_b_2(2:5,:);



signal_all= noise_air_cond + speakers_pad; %+ noise_background_a_pad + noise_background_b_pad;
dir_res = '/Users/sharon/Dropbox (BIU)/ARI/records/exp1_static/results/450_lcmv/';
mixed_signal_file = strcat(dir_res,'mixed_signal.wav');
fs=16000;
audiowrite(mixed_signal_file,signal_all',fs)