%% generate_matlab_fixture.m
% Produce a .mat fixture used by the Python parity test
% (tests/test_operator_octave_parity.py).
%
% Run from the repo root in Octave:
%   octave --no-gui --eval "addpath('matlab','scripts'); generate_matlab_fixture"
% or in MATLAB:
%   addpath('matlab','scripts'); generate_matlab_fixture
%
% The fixture contains a fixed random X plus L and Q from tx_lim_operator.
% Trend extraction is not part of the parity test because the original
% tx_lim_trend.m has a known undefined-X bug; the Python port (lim.trend)
% is covered by its own tests.

function generate_matlab_fixture()
    if exist('OCTAVE_VERSION', 'builtin') ~= 0
        pkg load statistics
    end

    rand('state', 12345);
    randn('state', 12345);

    n_modes = 5;
    n_times = 2000;
    X = randn(n_modes, n_times);
    tau0 = 4;

    X0 = X(:, 1:end-tau0);
    Xtau = X(:, 1+tau0:end);
    [L, Q] = tx_lim_operator(X0, Xtau, tau0);

    out_dir = fullfile('tests', 'data', 'matlab_reference');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    out_path = fullfile(out_dir, 'lim_fixture.mat');
    save('-v7', out_path, 'X', 'tau0', 'L', 'Q');
    fprintf('Wrote %s\n', out_path);
end
