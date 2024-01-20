% File reading
[y_male, fs_male] = audioread("heed_m.wav");
[y_female, fs_female] = audioread("heed_f.wav");

% 100ms vowel segments
vowel_segment_male = y_male(0.07 * fs_male : 0.17 * fs_male);
vowel_segment_female = y_female(0.07 * fs_female : 0.17 * fs_female);

% LPC modelling (male)
order = 18;
[A_male, G_male] = lpc(vowel_segment_male, order);

%LPC filter frequency response plot (male)
figure;
freqz(1, A_male, 1024, fs_male);
title('LPC Filter Frequency Response (Male Vowel)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');

%Plot male vowel amplitude spectrum & LPC filter response
figure;
fft_size = 1024;
speech_spectrum_male = 20 * log10(abs(fft(vowel_segment_male, fft_size)));
f = linspace(0, fs_male/2, fft_size/2);

plot(f, speech_spectrum_male(1:fft_size/2), 'b', 'LineWidth', 1.5);
hold on;
lpc_spectrum_male = 20 * log10(abs(freqz(1, A_male, f, fs_male)));
plot(f, lpc_spectrum_male, 'r--', 'LineWidth', 1.5);

legend('Speech Spectrum', 'LPC Filter Response');
title('Amplitude Spectrum and LPC Filter Response (Male Vowel)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');

% LPC modelling (female)
f_order = 30;
[A_female, G_female] = lpc(vowel_segment_female, f_order);

%LPC filter frequency response plot (female)
figure;
freqz(1, A_female, 1024, fs_female);
title('LPC Filter Frequency Response (Female Vowel)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');

%Plot female vowel amplitude spectrum & LPC filter response
figure;
fft_size = 1024;
speech_spectrum_female = 20 * log10(abs(fft(vowel_segment_female, fft_size)));
f = linspace(0, fs_female/2, fft_size/2);

plot(f, speech_spectrum_female(1:fft_size/2), 'b', 'LineWidth', 1.5);
hold on;
lpc_spectrum_female = 20 * log10(abs(freqz(1, A_female, f, fs_female)));
plot(f, lpc_spectrum_female, 'r--', 'LineWidth', 1.5);

legend('Speech Spectrum', 'LPC Filter Response');
title('Amplitude Spectrum and LPC Filter Response (Female Vowel)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');

%Pitch detection
% [f0_male, t_male] = pitch(vowel_segment_male, fs_male);
% [f0_female, t_female] = pitch(vowel_segment_female, fs_female);

% Autocorr-based pitch (male)
max_lag_male = round(fs_male / 80);
min_f0_male = 80; %Min pitch freq
max_f0_male = 400; %Max expected pitch freq

autocorr_male = xcorr(vowel_segment_male, max_lag_male);
autocorr_male = autocorr_male(max_lag_male+1:end); %only +ve lags

[~, locs_male] = findpeaks(autocorr_male, 'MinPeakDistance', round(fs_male / max_f0_male));

pitch_periods_male = diff(locs_male);
f0_male = fs_male ./ pitch_periods_male;

% Filter out invalid pitch values based on expected range
valid_indices_male = f0_male >= min_f0_male & f0_male <= max_f0_male;
f0_male = f0_male(valid_indices_male);

t_male = (locs_male(valid_indices_male) - 1) / fs_male;

% Autocorr-based pitch (female)
max_lag_female = round(fs_female / 80);
min_f0_female = 80;
max_f0_female = 400;

autocorr_female = xcorr(vowel_segment_female, max_lag_female);
autocorr_female = autocorr_female(max_lag_female+1:end);

[~, locs_female] = findpeaks(autocorr_female, 'MinPeakDistance', round(fs_female / max_f0_female));

pitch_periods_female = diff(locs_female);
f0_female = fs_female ./ pitch_periods_female;

% Filter out invalid pitch values
valid_indices_female = f0_female >= min_f0_female & f0_female <= max_f0_female;
f0_female = f0_female(valid_indices_female);

t_female = (locs_female(valid_indices_female) - 1) / fs_female;

%Mean fundamental freq (f0)
mean_f0_male = mean(f0_male);
mean_f0_female = mean(f0_female);

fprintf('Mean Fundamental Frequency (F0) for Male: %.2f Hz\n', mean_f0_male);
fprintf('Mean Fundamental Frequency (F0) for Female: %.2f Hz\n', mean_f0_female);

%Synthesis

%Generate periodic impulse train (male)
fundamental_frequency_synthesis_male = mean_f0_male;
duration_synthesis = 1;
impulse_train_male = impulse_train_generator(fundamental_frequency_synthesis_male, fs_male, duration_synthesis);

%Filter pulse train using LPC filter determined above
synthesised_speech_male = filter(1, A_male, impulse_train_male);

filename_male = sprintf('male_order%d_length%d.wav', order, 100);
audiowrite(filename_male, synthesised_speech_male, fs_male);
%sound(synthesised_speech_male, fs_male);

%Generate periodic impulse train (female)
fundamental_frequency_synthesis_female = mean_f0_female;
duration_synthesis = 1;
impulse_train_female = impulse_train_generator(fundamental_frequency_synthesis_female, fs_female, duration_synthesis);

synthesised_speech_female = filter(1, A_female, impulse_train_female);

filename_female = sprintf('female_order%d_length%d.wav', f_order, 100);
audiowrite(filename_female, synthesised_speech_female, fs_female);
%sound(synthesised_speech_female, fs_female);


function impulse_train = impulse_train_generator(frequency, fs, duration)
    %Generate a periodic impulse train
    % Calc. sample no.
    num_samples = round(duration * fs);

    % Gen. time vector
    t = (0:num_samples-1) / fs;

    % Gen. impulse train
    impulse_train = zeros(size(t));
    impulse_train(1:round(fs/frequency):end) = 1; %impulse every period
    impulse_train = impulse_train(1:num_samples);
end