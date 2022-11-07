function noise_out=noise(s_t)
SNR=10;
noise=randn(size(s_t));
noise=noise-mean(noise);
avg1=mean(s_t);
signal_power=mean((s_t-avg1).^2);
noise_variance=signal_power*10^(-SNR/10);
noise_out=sqrt(noise_variance)/std(noise)*noise;
end