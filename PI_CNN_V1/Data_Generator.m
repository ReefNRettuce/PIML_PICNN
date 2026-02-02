%% 1. Physics & Geometry Setup
clear; clc; close all;

% Constants
freq = 300;             % Frequency in Hz
c_sound = 343;          % Speed of sound in m/s
omega = 2 * pi * freq;  % Angular frequency
k = omega / c_sound;    % Wavenumber

% Room Dimensions (Meters)
% Actual Room: 3.1m (W) x 3.0m (H)
x_room = [-1.55, 1.55]; 
y_room = [-1.50, 1.50];

% Source Parameters
src_pos = [0.5, 0.5];   % Source location (x, y)
src_sigma = 0.05;       % Width of Gaussian source (sharpness)
src_amp = 100;          % Source amplitude

%% 2. PDE Model Initialization
model = createpde();

% Define Geometry: Rectangle
% The geometry engine uses columns [3, 4, x1, x2, x3, x4, y1, y2, y3, y4]'
R1 = [3, 4, x_room(1), x_room(2), x_room(2), x_room(1), ...
            y_room(1), y_room(1), y_room(2), y_room(2)]';
g = decsg(R1);
geometryFromEdges(model, g);

% Boundary Conditions: Rigid Walls (Neumann BC: du/dn = 0)
% Note: Standard rectangle edges are usually numbered 1:4. 
% If geometry changes, use 'pdegplot(model, 'EdgeLabels', 'on')' to verify.
applyBoundaryCondition(model, 'neumann', 'Edge', 1:4, 'g', 0, 'q', 0);

%% 3. Equation Specification: Helmholtz
% -Delta u - k^2 u = -f
c_coeff = 1;
a_coeff = -k^2;

% Source Function: High-density Gaussian
% FIX #2: Mesh Density vs. Source Sharpness
% We define the function handle here, but we address the resolution in step 4.
f_coeff = @(location, state) -src_amp * exp(-((location.x - src_pos(1)).^2 + ...
                                              (location.y - src_pos(2)).^2) / (2 * src_sigma^2));

specifyCoefficients(model, 'm', 0, 'd', 0, 'c', c_coeff, 'a', a_coeff, 'f', f_coeff);

%% 4. Mesh Generation (Refined for Source & Physics)
% FIX #2 IMPLEMENTATION:
% We need the mesh to be fine enough for the wave (lambda/30) AND 
% fine enough to resolve the Gaussian source width (sigma/4).

lambda = c_sound / freq;
h_physics = lambda / 30;       % For accurate wave propagation
h_source  = src_sigma / 4;     % For accurate source integration (approx 1.25 cm)

% Use the stricter of the two constraints
h_max = min(h_physics, h_source); 

% Generate high-fidelity quadratic mesh
generateMesh(model, 'Hmax', h_max, 'GeometricOrder', 'quadratic');

fprintf('Mesh Generated.\n   Lambda/30: %.4f m\n   Sigma/4:   %.4f m\n   Using Hmax: %.4f m\n', ...
    h_physics, h_source, h_max);

%% 5. Solve PDE
results = solvepde(model);
u_complex = results.NodalSolution;

%% 6. Data Extraction (Isotropic Grid)
% FIX #1: The "Non-Square Pixel" Problem
% We force the sampling grid to be physically square to ensure dx == dy.
% We use the largest dimension (Width = 3.1m) to define a square ROI.

grid_res = 32; % 32x32 Output
max_dim = max(diff(x_room), diff(y_room)); % Should be 3.1
half_dim = max_dim / 2;

% Define a square sampling region centered at (0,0)
x_vec = linspace(-half_dim, half_dim, grid_res);
y_vec = linspace(-half_dim, half_dim, grid_res);

[X_grid, Y_grid] = meshgrid(x_vec, y_vec);

% Verify Isotropic Spacing
dx = x_vec(2) - x_vec(1);
dy = y_vec(2) - y_vec(1);
fprintf('Grid Spacing:\n   dx: %.4f m\n   dy: %.4f m\n', dx, dy);

% Interpolate solution
% Note: Points outside the physical room (y > 1.5 or y < -1.5) will return NaN.
u_interp = interpolateSolution(results, X_grid, Y_grid);

% Handle NaNs (Padding "Empty Space" with 0)
u_interp(isnan(u_interp)) = 0;

% Reshape to image format
sound_field_ref = reshape(u_interp, [grid_res, grid_res]);

%% 7. Simulink Integration Prep
% Serialize for streaming (Row-major raster scan)
data_stream = sound_field_ref.'; 
data_flat = data_stream(:);

ts = 0.001; 
time_vec = (0:length(data_flat)-1)' * ts;

sim_data = [time_vec, real(data_flat), imag(data_flat)];

sim_input.time = time_vec;
sim_input.signals.values = sim_data(:, 2:3);
sim_input.signals.dimensions = 2;
sim_input.signals.label = 'Ref_Pressure_Stream';

%% 8. Visualization
figure('Position', [100, 100, 1200, 500]);

% Plot 1: The Raw FEM Solution
subplot(1, 3, 1);
pdeplot(model, 'XYData', real(u_complex), 'Colormap', 'jet');
title('FEM Solution (High Fidelity)');
axis equal;

% Plot 2: The Interpolated Grid (Validating Isotropy)
subplot(1, 3, 2);
imagesc(x_vec, y_vec, real(sound_field_ref));
set(gca, 'YDir', 'normal'); % Align Y-axis with Cartesian
title(['Interpolated 32x32 Grid (dx=' num2str(dx, '%.3f') ')']);
axis equal; % This visually confirms the pixels are square
colorbar;

% Plot 3: Mesh Zoom at Source
subplot(1, 3, 3);
pdeplot(model, 'Mesh', 'on');
xlim([src_pos(1)-0.15, src_pos(1)+0.15]);
ylim([src_pos(2)-0.15, src_pos(2)+0.15]);
title('Mesh Density @ Source');
axis equal;

save('acoustic_golden_ref.mat', 'sound_field_ref', 'sim_input', 'X_grid', 'Y_grid');