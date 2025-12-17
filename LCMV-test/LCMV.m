%% create signal for LCMV
clear all
% %dir = '/Users/sharon/Dropbox (BIU)/ARI/records/exp1_static/exp_1_static_rec_T60_450ms/';
% 
% %noise_background_file = strcat(dir,'noise_floor.wav');
% noise_air_cond_file = strcat(dir,'air_conditinar.wav');
% speaker_a_1_file = strcat(dir,'4_plus50deg/1r.wav');
% speaker_a_2_file = strcat(dir,'4_plus50deg/2r.wav');
% speaker_b_1_file = strcat(dir,'5_minus50deg/1r.wav');
% speaker_b_2_file = strcat(dir,'5_minus50deg/2r.wav');
% 
% %noise_background = audioread(noise_background_file)';
% noise_air_cond = audioread(noise_air_cond_file)';
% speaker_a_1 = audioread(speaker_a_1_file)';
% speaker_a_2 =  audioread(speaker_a_2_file)';
% speaker_b_1 = audioread(speaker_b_1_file)';
% speaker_b_2 = audioread(speaker_b_2_file)';
% 
% %noise_background= noise_background(2:5,:);
% noise_air_cond= noise_air_cond(2:5,:);
% speaker_a_1=speaker_a_1(2:5,:);
% speaker_a_2=speaker_a_2(2:5,:);
% speaker_b_1=speaker_b_1(2:5,:);
% speaker_b_2=speaker_b_2(2:5,:);
% 
% %padding for same length
% len=length(speaker_a_1);
% len_all= 4*len;
% 
% noise_air_cond=[noise_air_cond noise_air_cond];
% noise_air_cond=noise_air_cond(:,1:len_all);
% %noise_background_a= noise_background(:,1:3*len);
% %noise_background_a_pad=[noise_background_a zeros(4,len)];
% %noise_background_b=noise_background(:,len_all+1:len_all+len);
% %noise_background_b_pad=[noise_background_b zeros(4,3*len)];
% %speaker_a_1_pad=[zeros(4,len) speaker_a_1 zeros(4,2*len)];
% %speaker_a_2_pad=speaker_a_2(2:5,:);
% speakers_pad=[zeros(4,len) speaker_a_1 speaker_b_1 speaker_a_2+speaker_b_2];
% %speaker_b_2_pad=speaker_b_2(2:5,:);

root_mac = '/Users/sharon/Dropbox (BIU)/ARI/'
exp_dir = 'records/exp2_staic_2_Speakers_Array_In_and_Out_T60_120ms/tst2/ARI'
dir = 'C:\Users\user\Dropbox (BIU)\ARI\records\exp2_staic_2_Speakers_Array_In_and_Out_T60_120ms\tst2\ARI\Additional_Recording_Positions_NEW_Firmware\';
sess1 = strcat(dir,'0_plus50.wav');
sess1 = audioread(sess1)';
%signal_all= noise_air_cond + speakers_pad; %+ noise_background_a_pad + noise_background_b_pad;
sess1=sess1(2:5,:);
signal_all= sess1;
%dir_res = '/Users/sharon/Dropbox (BIU)/ARI/records/exp1_static/results/450_lcmv/';
dir_res = 'C:\Users\user\Dropbox (BIU)\ARI\records\exp2_staic_2_Speakers_Array_In_and_Out_T60_120ms\results_lcmv\sess1-0_plus50\';

mixed_signal_file = strcat(dir_res,'mixed_signal.wav');
fs=16000;
audiowrite(mixed_signal_file,signal_all',fs)

%soundsc(signal_all(1,:), fs)
%% create z_k

M=4;

%speech_file1 = strcat(dir,'4_plus50deg\1r.wav');
%speach1 = audioread(speech_file1)';
%speach1=speach1(2:5,:)
%speech_file2 = strcat(dir,'5_minus50deg\1r.wav');
%speach2 = audioread(speech_file2)';
%speach2=speach2(2:5,:);

%noise_file = strcat('C:\Users\user\Dropbox (BIU)\ARI\records\exp1_static\exp_1_static_rec_T60_120ms\air_conditinar.wav');
%noise = audioread(noise_file)';
%noise=noise(:,1:length(speach2));
%noise=noise(2:5,:);

Receivers=signal_all;
wlen = 1024%4096; % 2048
hop = wlen/4;  
nfft = wlen;   
NUP = ceil((nfft+1)/2);
win=hamming(wlen, 'periodic');
index=fix(1+((length(Receivers(1,:))-wlen)/hop));


z_k=ones(M,NUP,index);
for i = 1:M
    z_k(i,:,:)=my_stft(squeeze(Receivers(i,:)), win, hop, nfft);
end

s = spectrogram(squeeze(Receivers(1,:)),win,hop*3,nfft,'yaxis');
figure();imagesc(flipud(log(abs(s))))
%% create phi_noise & cholsky
noise_frames=100; %50;
epsilon=0.01;
noise_frame=z_k(:,:,1:noise_frames);   % end of the audio
noise_cor=zeros(NUP,M,M);
noise_cor_chol=zeros(NUP,M,M);
inv_chol=zeros(NUP,M,M);

for i = 1:NUP
    temp_noise=squeeze(noise_frame(:,i,:));
    noise_cor(i,:,:)=temp_noise*temp_noise'/noise_frames;
    noise_cor_chol(i,:,:)=chol(squeeze(noise_cor(i,:,:)));
    inv_chol(i,:,:)=squeeze(noise_cor_chol(i,:,:))+epsilon*eye(M)*(norm(squeeze(noise_cor_chol(i,:,:))));
end

%% create z_n and G
z_n_cor=zeros(NUP,M,M);
G1=zeros(NUP,M);
G2=zeros(NUP,M);

for i = 1:NUP
    %z_n=squeeze(z_k(:,i,400:450));
    z_n=squeeze(z_k(:,i,380:480));
    z_n=squeeze(inv_chol(i,:,:))\z_n;
    z_n_cor=z_n*z_n'/length(z_n);
    [w,v]=eig(z_n_cor);
    [~, idx] = max(max(v));
    fi=w(:,idx);
    temp=squeeze(noise_cor_chol(i,:,:))*fi;
    G1(i,:)=temp/temp(1);
end

for i = 1:NUP
    %z_n=squeeze(z_k(:,i,750:800));
    z_n=squeeze(z_k(:,i,730:830));
    z_n=squeeze(inv_chol(i,:,:))\z_n;
    z_n_cor=z_n*z_n'/length(z_n);
    [w,v]=eig(z_n_cor);
    [~, idx] = max(max(v));
    fi=w(:,idx);
    temp=squeeze(noise_cor_chol(i,:,:))*fi;
    G2(i,:)=temp/temp(1);
end

G=cat(3,G1,G2);

%% Create W and multiple

W=zeros(M,2,NUP);
z_total=zeros(NUP,2,index);
for i = 1:NUP
    g=squeeze(G(i,:,:));
    b=squeeze(noise_cor(i,:,:));
    inv_b=b+epsilon*eye(M)*norm(b);
    c=inv_b\g;
    inv_temp=g'*c+epsilon*eye(2)*norm(g'*c);
    W(:,:,i)=c/inv_temp;
    z_total(i,:,:)=squeeze(W(:,:,i))'*squeeze(z_k(:,i,:));
end

%% ISTFT

[first_channel, t1] = my_istft(squeeze(z_total(:,1,:)), win, win, hop, nfft, fs);
[second_channel, t2] = my_istft(squeeze(z_total(:,2,:)), win, win, hop, nfft, fs);


first_channel_file=strcat(dir_res,'first_channel_120_short.wav');
second_channel_file=strcat(dir_res,'second_channel_120_short.wav');
            
audiowrite(first_channel_file,first_channel,fs) 
audiowrite(second_channel_file,second_channel,fs) 
