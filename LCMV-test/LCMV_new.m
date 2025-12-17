%% Prepare
clear all
close all

%% Upload signals and set parameters

root_mac = '/Users/sharon/Dropbox (BIU)/ARI/';
%exp_dir = 'records/exp2_staic_2_Speakers_Array_In_and_Out_T60_120ms/tst2/ARI/Additional_Recording_Positions_NEW_Firmware/';


Array = 'Medal'; % 'ARI'; % 'Medal' % 'External' % Simulation
 
        switch Array
          
            case {'ARI'}
            disp(Array)
            %exp_dir = 'Recordings/exp2_staic_2_Speakers_Array_In_and_Out_T60_120ms/tst3/ARI_array/';
            %exp_dir = 'Recordings/exp3_T60_120/ARI/';
            %exp_dir = 'Recordings/exp4_T60_120/ARI_B/';
            %exp_dir = 'Recordings/exp5_comp_os/ROS/';
            %exp_dir = 'Recordings/exp5_comp_os/LINUX/';
            %exp_dir = 'Recordings/exp5_comp_os/ROS/';
            %exp_dir = 'Recordings/Shell/Without/';
            exp_dir = 'Recordings/Reverb-340/ARI/';
            
            %rec_sig_file = strcat(root_mac,exp_dir,'ARI_6CH_ON.wav'); 
            rec_sig_file = strcat(root_mac,exp_dir,'ARI_rev_C.wav'); % 'ARI_minus30_plus30_A' 'ARI_0_40.' 'ARI_ROS_A' 'ARI_linux_A' 'ARI_python_A' 'ARI_ros_A' 'python_no_shell_C'
            
            noise_tim_st = 0; noise_tim_fn = 4; % Sec
            first_tim_st = 6.0; first_tim_fn = 9.5; % Sec
            second_tim_st = 11.0;  second_tim_fn = 15.5;% Sec
        
            
%           case {'ARI_OFF'}
%             disp(Array)
%             exp_dir = 'records/exp2_staic_2_Speakers_Array_In_and_Out_T60_120ms/tst3/ARI_array/';
%             exp_dir = 'Recordings/exp3_T60_120/ARI/';
%             rec_sig_file = strcat(root_mac,exp_dir,'ARI_6CH_OFF.wav');
%             
%             noise_tim_st = 0; noise_tim_fn = 2; % Sec
%             first_tim_st = 6.5; first_tim_fn = 9; % Sec
%             second_tim_st = 12.3;  second_tim_fn = 15.4;% Sec
          
            case 'Medal'
            disp(Array)
            %exp_dir = 'records/exp2_staic_2_Speakers_Array_In_and_Out_T60_120ms/tst3/External/';
            %exp_dir = 'Recordings/exp3_T60_120/Medal/';
            %exp_dir = 'Recordings/exp4_T60_120/Medal_B/';
            exp_dir = 'Recordings/Reverb-340/Medal/'; 
            %rec_sig_file = strcat(root_mac,exp_dir,'ext_6CH_ON.wav'); 
            rec_sig_file = strcat(root_mac,exp_dir,'medal_rev_C.wav'); % 'medal_0_40.wav' 'medal_minus30_plus30_B'
            
            noise_tim_st = 0; noise_tim_fn = 4; % Sec
            first_tim_st = 6.0; first_tim_fn = 10.0; % Sec
            second_tim_st = 11.0;  second_tim_fn = 15.0;% Sec
            
          case 'External'
            disp(Array)
            %exp_dir = 'records/exp2_staic_2_Speakers_Array_In_and_Out_T60_120ms/tst3/Stand/';
            %exp_dir = 'Recordings/exp3_T60_120/Horizon/';
            %exp_dir = 'Recordings/exp4_T60_120/Horizon_B/'; 
            exp_dir = 'Recordings/Reverb-340/External/'; 
            
            %rec_sig_file = strcat(root_mac,exp_dir,'hor_6CH_ON.wav');
            rec_sig_file = strcat(root_mac,exp_dir,'horizon_rev_B.wav'); % 'horizon_0_40' 'horizon_minus30_plus30_B' 'horizon_minus30_plus30_B'
            
            noise_tim_st = 0; noise_tim_fn = 4; % Sec
            first_tim_st = 5.5; first_tim_fn = 10.0; % Sec
            second_tim_st = 11.0;  second_tim_fn = 15.0;% Sec
        
            
          case 'Simulation'
            disp(Array)
            exp_dir = 'Recordings/Sim_SNR_10_SIR_0_T60_300/9/';
            rec_sig_file = strcat(root_mac,exp_dir,'signal_9.wav');
            
            noise_tim_st = 0; noise_tim_fn = 2; % Sec
            first_tim_st = 2.3; first_tim_fn = 4.3; % Sec
            second_tim_st = 6;  second_tim_fn = 9.5;% Sec
            
          otherwise
            disp('Unknown array')
        end

[rec_sig,fs] = audioread(rec_sig_file);


res_dir = strcat(root_mac,exp_dir,'Results/');
ref = 1;

signal_proc = rec_sig(:,1);
signal_mics = rec_sig(:,2:5);
%signal_mics = rec_sig(:,1:4);
signal_ref = rec_sig(:,ref+1);


signal_mics_file = strcat(res_dir,'signal_mics.wav');
signal_proc_file = strcat(res_dir,'signal_proc.wav');
signal_ref_file = strcat(res_dir,'signal_ref.wav');

audiowrite(signal_mics_file,signal_mics,fs)
audiowrite(signal_proc_file,0.9*signal_proc/max(abs(signal_proc)),fs)
audiowrite(signal_ref_file,0.9*signal_ref/max(abs(signal_ref)),fs)



%% STFT of mic. signals

M = size(signal_mics,2);

z = signal_mics';

% Parameters

wlen = 1024; % 4096; % 2048
R = wlen/4;  
nfft = wlen;   
K = nfft/2+1;
win = hamming(wlen, 'periodic');
L = floor(1+((length(z(1,:))-wlen)/R));


F_L = K-1; %K-1; % 100; % Noncausal Causal part of the RTF % maximum K-1
F_R = K-1; % K-1; % 100; % Causal part of the RTF % maximum K-1


z_k=ones(M,K,L);
for i = 1:M
    z_k(i,:,:) = my_stft(squeeze(z(i,:)), win, R, nfft);
end

T = (0:L-1)/fs*R;
F = (0:K-1)*fs/2/(K-1);
figure(1)
imagesc(T,F,20*log10(abs(squeeze(z_k(1,1:K,:))+eps)))
axis xy
xlabel('Time[Sec]','fontsize',14);
ylabel('Frequency[Hz]','fontsize',14);
set(gca,'fontsize',14);
colorbar

%% Generate noise correlation matrix & apply Cholsky decomposition

noise_frm_st = ceil(noise_tim_st*fs/R+1); noise_frm_fn = floor(noise_tim_fn*fs/R);




z_n = z_k(:,:,noise_frm_st:noise_frm_fn);   

T = (0:size(z_n,3)-1)/fs*R;
F = (0:K-1)*fs/2/(K-1);
figure(2)
imagesc(T,F,20*log10(abs(squeeze(z_n(1,1:K,:))+eps)))
axis xy
xlabel('Time[Sec]','fontsize',14);
ylabel('Frequency[Hz]','fontsize',14);
set(gca,'fontsize',14);
colorbar

epsilon = 0.01;

noise_cor = zeros(K,M,M);
noise_cor_chol = zeros(K,M,M);
inv_chol = zeros(K,M,M);

for k = 1:K % choleski decomposition for noise whitening
    temp_noise = squeeze(z_n(:,k,:));
    noise_cor(k,:,:) = temp_noise*temp_noise'/length(z_n);
    noise_cor_chol(k,:,:) = chol(squeeze(noise_cor(k,:,:)));
    inv_chol(k,:,:) = squeeze(noise_cor_chol(k,:,:))+epsilon*eye(M)*(norm(squeeze(noise_cor_chol(k,:,:))));
end



%% RTF of first speaker

first_frm_st = ceil(first_tim_st*fs/R+1); first_frm_fn = floor(first_tim_fn*fs/R);

z_f = z_k(:,:,first_frm_st:first_frm_fn);   

T = (0:size(z_f,3)-1)/fs*R;
F = (0:K-1)*fs/2/(K-1);
figure(3)
imagesc(T,F,20*log10(abs(squeeze(z_f(1,1:K,:))+eps)))
axis xy
xlabel('Time[Sec]','fontsize',14);
ylabel('Frequency[Hz]','fontsize',14);
set(gca,'fontsize',14);
colorbar

z_f_cor=zeros(K,M,M);
G_f=zeros(K,M);

for k = 1:K
    temp_first = squeeze(z_f(:,k,:));
    temp_first = squeeze(inv_chol(k,:,:))\temp_first;
    z_f_cor = temp_first*temp_first'/length(temp_first);
    [w,v] = eig(z_f_cor);
    [~, idx] = max(max(v));
    fi = w(:,idx);
    temp = squeeze(noise_cor_chol(k,:,:))*fi;
    G_f(k,:) = temp/temp(ref);
end

for m = 1:M % add small noise to avoid devision by zero
    ind = find(abs(G_f(:,m))>3*mean(abs(G_f(:,m))));
    G_f(ind,m) = 2*binornd(1,0.5,length(ind),1)-1;
    
end

G_f_full = [G_f(1:K,:) ; conj(G_f(K-1:-1:2,:))];
g_f = ifft(G_f_full,[],1);

% g_f_c = ifftshift(g_f,1);
% g_f_c(1:K-1-F_L,:) = 0; g_f_c(K-1+F_R:end,:) = 0;
% g_f_c = fftshift(g_f_c,1);
%figure(5), plot(-K+1:K-2,g_f_c(:,1))

g_f_trc = zeros(size(g_f));
g_f_trc(1:F_R,:) = g_f(1:F_R,:);
g_f_trc(nfft-F_L+1:end,:) = g_f(nfft-F_L+1:end,:);
figure(4)
plot(ifftshift(g_f_trc))
title('ReIR-1st')


G_f_trc_full = fft(g_f_trc);
G_f = G_f_trc_full(1:K,:);

%% RTF of second speaker

second_frm_st = ceil(second_tim_st*fs/R+1); second_frm_fn = floor(second_tim_fn*fs/R);

z_s = z_k(:,:,second_frm_st:second_frm_fn);  

T = (0:size(z_s,3)-1)/fs*R;
F = (0:K-1)*fs/2/(K-1);
figure(5)
imagesc(T,F,20*log10(abs(squeeze(z_s(1,1:K,:))+eps)))
axis xy
xlabel('Time[Sec]','fontsize',14);
ylabel('Frequency[Hz]','fontsize',14);
set(gca,'fontsize',14);
colorbar

z_s_cor = zeros(K,M,M);
G_s = zeros(K,M);


for k = 1:K
    temp_second = squeeze(z_s(:,k,:));
    temp_second = squeeze(inv_chol(k,:,:))\temp_second;
    z_s_cor = temp_second*temp_second'/length(temp_second);
    [w,v] = eig(z_s_cor);
    [~, idx] = max(max(v));
    fi = w(:,idx);
    temp = squeeze(noise_cor_chol(k,:,:))*fi;
    G_s(k,:) = temp/temp(ref);
end

for m = 1:M
    ind = find(abs(G_s(:,m))>3*mean(abs(G_s(:,m))));
    G_s(ind,m) = 2*binornd(1,0.5,length(ind),1)-1;
    
end
G_s_full = [G_s(1:K,:) ; conj(G_s(K-1:-1:2,:))];
g_s = ifft(G_s_full,[],1);

% g_s_c = ifftshift(g_s,1);
% g_s_c(1:K-1-F_L,:) = 0; g_s_c(K-1+F_R:end,:) = 0;
% g_s_c = fftshift(g_s_c,1);
%figure(6), plot(-K+1:K-2,g_s_c(:,2))


g_s_trc = zeros(size(g_s));
g_s_trc(1:F_R,:) = g_s(1:F_R,:);
g_s_trc(nfft-F_L+1:end,:) = g_s(nfft-F_L+1:end,:);

figure(6)
plot(ifftshift(g_s_trc))
title('ReIR-2nd')

G_s_trc_full = fft(g_s_trc);
G_s = G_s_trc_full(1:K,:);

%%

G = cat(3,G_f,G_s);

%% Generate W and apply the MVDR beamformer

W = zeros(M,2,K); % MVDR weights
z_out = zeros(K,2,L);

for k = 1:K
    g = squeeze(G(k,:,:));
    b = squeeze(noise_cor(k,:,:));
    inv_b = b+epsilon*norm(b)*eye(M);
    c = inv_b\g;
    inv_temp = g'*c+epsilon*norm(g'*c)*eye(2);
    W(:,:,k) = c/inv_temp;
    z_out(k,:,:) = squeeze(W(:,:,k))'*squeeze(z_k(:,k,:));
end

%% ISTFT

[first_channel, t1] = my_istft(squeeze(z_out(:,1,:)), win, win, R, nfft, fs);
[second_channel, t2] = my_istft(squeeze(z_out(:,2,:)), win, win, R, nfft, fs);


first_channel_file = strcat(res_dir,'first_out.wav');
second_channel_file = strcat(res_dir,'second_out.wav');
            
audiowrite(first_channel_file,0.9*first_channel/max(abs(first_channel)),fs) 
audiowrite(second_channel_file,0.9*second_channel/max(abs(second_channel)),fs) 
