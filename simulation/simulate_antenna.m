%% Standalone EM Simulation Script for AI Pixel Antennas
% Uses MATLAB Antenna Toolbox
% Computes full-wave S11 spectrum (10 - 20 GHz) and checks resonance.

clear; clc; close all;

%% 1. Load Antenna Mask
% By default, loads antenna_mask_14.25GHz.csv or specify your own CSV file:
maskFile = 'antenna_mask_14.25GHz.csv';

if ~isfile(maskFile)
    % Look in ../masks/ if running from simulation/ directory
    if isfile(fullfile('..', 'masks', maskFile))
        maskFile = fullfile('..', 'masks', maskFile);
    elseif isfile(fullfile('masks', maskFile))
        maskFile = fullfile('masks', maskFile);
    end
end

fprintf('Loading mask from: %s\n', maskFile);
raw = readmatrix(maskFile);
raw = raw(~all(isnan(raw), 2), :);
raw = raw(:, ~all(isnan(raw), 1));
if isequal(size(raw), [12 12])
    ant_design = raw;
else
    error('Expected 12x12 matrix, got %s', mat2str(size(raw)));
end

fprintf('Pixels ON: %d / 144 (%.1f%%)\n', sum(ant_design(:) ~= 0), 100 * sum(ant_design(:) ~= 0) / 144);

%% 2. Setup Frequency Grid (10 - 20 GHz, 81 points)
freq = linspace(10e9, 20e9, 81);
freqGHz = freq / 1e9;

%% 3. Construct Antenna Geometry & Mesh
fprintf('Building 12x12 pixel antenna structure...\n');
antenna = generateantenna_air(13, 13, ant_design);

%% 4. Run Electromagnetic S-Parameter Simulation
fprintf('Running Method of Moments (MoM) simulation (takes 1-3 mins)...\n');
tSim = tic;
s = sparameters(antenna, freq, 50);
simTime = toc(tSim);
fprintf('Simulation completed in %.1f seconds (%.2f minutes).\n', simTime, simTime / 60);

s11_dB = 20 * log10(max(abs(squeeze(s.Parameters(1, 1, :))), 1e-12));
[minS11, minIdx] = min(s11_dB);
resFreq = freqGHz(minIdx);

fprintf('===============================================\n');
fprintf('Resonant Frequency : %.3f GHz\n', resFreq);
fprintf('Minimum S11        : %.2f dB\n', minS11);
fprintf('Resonant Dip <= -10: %s\n', mat2str(minS11 <= -10));
fprintf('===============================================\n');

%% 5. Plot S11 Spectrum
figure('Name', 'AI Pixel Antenna S11', 'Color', 'w', 'Position', [150 150 850 500]);
plot(freqGHz, s11_dB, 'b-', 'LineWidth', 2.2); hold on;
yline(-10, 'r--', 'Threshold (-10 dB)', 'LineWidth', 1.5);
plot(resFreq, minS11, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
text(resFreq, minS11 - 1.5, sprintf('%.3f GHz (%.1f dB)', resFreq, minS11), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);

grid on;
xlabel('Frequency (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('S_{11} Return Loss (dB)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Full-Wave EM S_{11} Simulation — %s', maskFile), 'FontSize', 13);
xlim([10 20]);
ylim([-35 0]);
box on;
