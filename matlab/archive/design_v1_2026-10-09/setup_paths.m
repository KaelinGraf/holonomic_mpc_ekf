function setup_paths(mode)
%SETUP_PATHS Add the project folders to the MATLAB path.
%   setup_paths()             use YOUR implementation in matlab/ekf and matlab/mpc
%   setup_paths('reference')  use the sealed reference in matlab/reference, e.g. to
%                             check that the tests themselves behave
if nargin < 1, mode = 'user'; end
root = fileparts(mfilename('fullpath'));

% Remove all implementation folders first so switching modes is clean
warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath(fullfile(root, 'ekf'), fullfile(root, 'mpc'), ...
       fullfile(root, 'reference', 'ekf'), fullfile(root, 'reference', 'mpc'));
warning('on', 'MATLAB:rmpath:DirNotFound');

addpath(root, ...
        fullfile(root, 'params'), ...
        fullfile(root, 'sim'), ...
        fullfile(root, 'utils'), ...
        fullfile(root, 'tests'));
switch lower(mode)
    case 'user'
        addpath(fullfile(root, 'ekf'), fullfile(root, 'mpc'));
    case 'reference'
        addpath(fullfile(root, 'reference', 'ekf'), fullfile(root, 'reference', 'mpc'));
    otherwise
        error('setup_paths:mode', 'mode must be ''user'' or ''reference''');
end
fprintf('setup_paths: %s implementation\n  ekf_predict -> %s\n  mpc_step    -> %s\n', ...
        mode, which('ekf_predict'), which('mpc_step'));
end
