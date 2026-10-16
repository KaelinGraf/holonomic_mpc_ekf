%RUN_ALL_TESTS Run the EKF test suite and print a PASS / WARN / FAIL table.
%   >> setup_paths;             run_all_tests   % against your implementation in matlab/ekf
%   >> setup_paths('reference'); run_all_tests   % against the sealed reference (checks the tests)
%
%   Order is cheap-to-expensive. A test that throws (for example a stub that is
%   not written yet) is reported as FAIL with the error message.

tests = {@test_ekf_jacobians, @test_ekf_noise_free, @test_ekf_delay, ...
         @test_ekf_target_motion, @test_ekf_dropout, @test_ekf_consistency, ...
         @test_mpc_kinematics, @test_mpc_linearisation, @test_mpc_prediction, ...
         @test_mpc_unconstrained, @test_mpc_constraints, ...
         @test_mpc_closed_loop, @test_integrated_docking};
fprintf('\n%-6s %-62s %s\n', 'result', 'test', 'detail');
fprintf('%s\n', repmat('-', 1, 110));
npass = 0;  nwarn = 0;
tic;
for k = 1:numel(tests)
    try
        r = tests{k}();
        if ~r.pass
            tag = 'FAIL';
        elseif r.warn
            tag = 'WARN';  nwarn = nwarn + 1;
        else
            tag = 'PASS';  npass = npass + 1;
        end
    catch ME
        r = struct('name', func2str(tests{k}), 'msg', ['error: ', ME.message]);
        tag = 'FAIL';
    end
    fprintf('%-6s %-62s %s\n', tag, r.name, r.msg);
end
fprintf('%s\n%d PASS, %d WARN, %d FAIL of %d tests (%.0f s)\n', repmat('-', 1, 110), ...
        npass, nwarn, numel(tests) - npass - nwarn, numel(tests), toc);
