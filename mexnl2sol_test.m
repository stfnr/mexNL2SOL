function mexnl2sol_test()
%MEXNL2SOL_TEST Test suite for mexnl2sol
%   This is a MATLAB translation of nl2sol_test1.f90 and nl2sol_test2.f90.
%   Test results are printed to console and saved to 'mexnl2sol_test_results.txt'.
%
%   Original Fortran 90 version by John Burkardt.
%   The test problems are derived from the original NL2SOL distribution by
%   John Dennis, David Gay, and Roy Welsch.
%
%   Reference:
%
%    K M Brown,
%    A Quadratically Convergent Newton-like Method Based upon
%    Gaussian Elimination,
%    SIAM Journal on Numerical Analysis,
%    Volume 6, pages 560-569, 1969.
%
%    John Dennis, David Gay and Roy Welsch,
%    Algorithm 573: An Adaptive Nonlinear Least-squares Algorithm,
%    ACM Transactions on Mathematical Software,
%    Volume 7, Number 3, pages 348-368, 1981.
%
%    Philip Gill and Walter Murray,
%    Algorithms for the Solution of the Non-linear Least-squares Problem,
%    SIAM Journal on Numerical Analysis,
%    Volume 15, Number 5, pages 977-991, 1978.
%
%    R R Meyer,
%    Theoretical and Computational Aspects of Nonlinear Regression,
%    in Nonlinear Programming,
%    edited by J B Rosen, O L Mangasarian, and K Ritter,
%    pages 465-486,
%    Academic Press, New York, 1970.


    % Setup diary to capture all output including MEX prints
    logfile = 'mexnl2sol_test_results.txt';
    if exist(logfile, 'file')
        delete(logfile);
    end
    diary(logfile);
    
    fprintf('=================================================================\n');
    fprintf('MEXNL2SOL Test Suite Results\n');
    fprintf('Date: %s\n', char(datetime('now')));
    fprintf('MATLAB Version: %s\n', version);
    fprintf('System: %s\n', computer);
    fprintf('=================================================================\n');

    fprintf('=================================================================\n');
    fprintf('NL2SOL_TEST1: MATLAB version\n');
    fprintf('Test the NL2SOL library (Madsen Example)\n');
    fprintf('=================================================================\n');
    
    test01();
    test02();

    fprintf('\n\n=================================================================\n');
    fprintf('NL2SOL_TEST2: MATLAB version\n');
    fprintf('Test the NL2SOL library (General Test Suite)\n');
    fprintf('=================================================================\n');
    
    run_test_suite();
    
    diary off;
    fprintf('Test results saved to %s\n', logfile);
end

%% TEST01: Madsen with Analytic Jacobian
function test01()
    fprintf('\nNL2SOL_PRB1:\n');
    fprintf('  MATLAB version\n');
    fprintf('  Test the NL2SOL library.\n\n');
    
    fprintf('TEST01:\n');
    fprintf('  Test the NL2SOL routine,\n');
    fprintf('  which requires a user residual and jacobian.\n\n');

    x0 = [3.0; 1.0];
    
    fprintf('      I     Initial X(i)\n\n');
    for i = 1:length(x0)
        fprintf('     %d     %e\n', i, x0(i));
    end
    fprintf('\n');

    % Options
    opts.Jacobian = 'on';
    
    % Request iteration printing from the MEX wrapper
    % PrintLevel 3 in mexnl2sol gives: fEval, fJac, SSE
    printLevel = 3;
    
    % Call solver
    [x, resnorm, ~, exitflag, output] = mexnl2sol(@madsen_problem, x0, [], [], opts, printLevel);
    
    if ismember(exitflag, [3, 4, 5, 6])
        status = 'PASS';
    else
        status = 'FAIL';
    end
    
    % MEX wrapper prints exit criterion via getStatus when printLevel > 0

    fprintf(' function     %e\n', resnorm);
    % Note: fevals not captured in this call signature, usually in 'output' or extra arg
    % We stick to basic reporting here
    
    fprintf('\n      I      Final X(I)\n\n');
    for i = 1:length(x)
        fprintf('     %d     %e\n', i, x(i));
    end
    
    fprintf('\nResult: %s\n', status);
end

%% TEST02: Madsen with Finite Difference Jacobian
function test02()
    fprintf('\nTEST02:\n');
    fprintf('  Test the NL2SNO routine (via mexnl2sol option),\n');
    fprintf('  which requires only a user residual.\n');
    fprintf('  The jacobian is approximated internally.\n\n');

    x0 = [3.0; 1.0];
    
    fprintf('      I     Initial X(i)\n\n');
    for i = 1:length(x0)
        fprintf('     %d     %e\n', i, x0(i));
    end
    fprintf('\n');
    
    % Options
    opts.Jacobian = 'off';
    printLevel = 3;
    
    % Call solver
    [x, resnorm, ~, exitflag, output] = mexnl2sol(@madsen_problem_no_jac, x0, [], [], opts, printLevel);
    
    if ismember(exitflag, [3, 4, 5, 6])
        status = 'PASS';
    else
        status = 'FAIL';
    end
    
    fprintf(' function     %e\n', resnorm);
    
    fprintf('\n      I      Final X(I)\n\n');
    for i = 1:length(x)
        fprintf('     %d     %e\n', i, x(i));
    end
    
    fprintf('\nResult: %s\n', status);
end

function [r, J] = madsen_problem(x)
    r = zeros(3,1);
    r(1) = x(1)^2 + x(2)^2 + x(1)*x(2);
    r(2) = sin(x(1));
    r(3) = cos(x(2));
    
    if nargout > 1
        J = zeros(3,2);
        J(1,1) = 2.0*x(1) + x(2);
        J(1,2) = 2.0*x(2) + x(1);
        J(2,1) = cos(x(1));
        J(2,2) = 0.0;
        J(3,1) = 0.0;
        J(3,2) = -sin(x(2));
    end
end

function r = madsen_problem_no_jac(x)
    r = zeros(3,1);
    r(1) = x(1)^2 + x(2)^2 + x(1)*x(2);
    r(2) = sin(x(1));
    r(3) = cos(x(2));
end

function c = get_iv1_char(exitflag)
    % Mapping based on nl2sol_f90 data statement for rc
    switch exitflag
        case 1, c = '.';
        case 2, c = '+';
        case 3, c = 'x';
        case 4, c = 'r';
        case 5, c = 'b';
        case 6, c = 'a';
        case 7, c = 's';
        case 8, c = 'f';
        case 9, c = 'e';
        case 10, c = 'i';
        otherwise, c = '?';
    end
end

%% NL2SOL_TEST2 Implementation
function run_test_suite()
    % Problem definitions
    % Format: {nex, n, p, title, xscal1, xscal2}
    problems = {
        1, 2, 2, 'Rosenbrock', 1, 3;
        2, 3, 3, 'Helix', 1, 3;
        3, 4, 4, 'Singular', 1, 3;
        4, 7, 4, 'Woods', 1, 3;
        5, 3, 3, 'Zangwill', 1, 1;
        6, 5, 3, 'Engvall', 1, 3;
        7, 2, 2, 'Branin', 1, 3;
        8, 3, 2, 'Beale', 1, 2;
        9, 5, 4, 'Cragg', 1, 2;
        10, 10, 3, 'Box', 1, 2;
        11, 15, 15, 'Davidon1', 1, 1;
        12, 2, 2, 'Freudenstein', 1, 3;
        13, 31, 6, 'Watson6', 1, 1;
        14, 31, 9, 'Watson9', 1, 1;
        15, 31, 12, 'Watson12', 1, 1;
        16, 31, 20, 'Watson20', 1, 3;
        17, 8, 8, 'Chebyquad', 1, 2;
        18, 20, 4, 'Brown', 1, 3;
        19, 15, 3, 'Bard', 1, 3;
        20, 10, 2, 'Jennrich', 1, 1;
        21, 11, 4, 'Kowalik', 1, 3;
        22, 33, 5, 'Osborne1', 1, 1;
        23, 65, 11, 'Osborne2', 1, 2;
        24, 3, 2, 'Madsen', 1, 3;
        25, 16, 3, 'Meyer', 1, 3;
        26, 5, 5, 'Brown5', 1, 3;
        27, 10, 10, 'Brown10', 1, 3;
        30, 15, 3, 'Bard+10', 1, 3;
        31, 11, 4, 'Kowal+10', 1, 3;
        32, 16, 3, 'Meyer+10', 1, 3;
        33, 31, 6, 'Watson6+10', 1, 3;
        34, 31, 9, 'Watson9+10', 1, 3;
        35, 31, 12, 'Watson12+10', 1, 3;
        36, 31, 20, 'Watson20+10', 1, 3;
    };

    results = [];
    
    fprintf('\nNL2SOL_PRB2:\n');
    fprintf('  MATLAB version\n');
    fprintf('  Test the NL2SOL library.\n\n');

    for k = 1:size(problems, 1)
        prob = problems(k, :);
        nex = prob{1};
        n = prob{2};
        p = prob{3};
        title = prob{4};
        xscal1 = prob{5};
        xscal2 = prob{6};
        
        for irun = xscal1:xscal2
            x0scal = 10^(irun-1);
            
            fprintf('\n ***** nl2sol on problem %s (Scale: %g) *****\n', title, x0scal);
            
            x0 = get_xinit(nex, p);
            x0 = x0 * x0scal;
            
            fprintf('\n      I     Initial X(i)\n\n');
            for i = 1:length(x0)
                fprintf('     %d     %e\n', i, x0(i));
            end
            fprintf('\n');
            
            fun = @(x) problem_wrapper(nex, x, n, p);
            
            opts.Jacobian = 'on';
            opts.MaxFunEvals = 1000;
            opts.MaxIter = 1000;
            
            if nex == 11 || nex == 16
               opts.MaxFunEvals = 20;
               opts.MaxIter = 15;
            end
            if nex == 25
                opts.MaxFunEvals = 400;
                opts.MaxIter = 300;
            end
            
            try
                % PrintLevel 3 for detailed logs
                [x_final, resnorm, ~, exitflag, iter, fevals] = mexnl2sol(fun, x0, [], [], opts, 3);
                
                iv1_char = get_iv1_char(exitflag);
                
                if ismember(exitflag, [3, 4, 5, 6])
                    result_str = 'PASS';
                else
                    result_str = 'FAIL';
                end
                
                % Compute Gradient at solution for detailed output
                [r_final, J_final] = problem_wrapper(nex, x_final, n, p);
                g_final = J_final' * r_final;
                
                fprintf('\n function     %e\n', resnorm);
                fprintf(' func. evals      %d\n', fevals);
                fprintf(' iterations       %d\n', iter);
                
                fprintf('\n      I      Final X(I)        G(I)\n\n');
                for i = 1:p
                    fprintf('     %d     %e     %e\n', i, x_final(i), g_final(i));
                end
                
            catch ME
                resnorm = NaN;
                exitflag = -999;
                iter = -1;
                fevals = -1;
                iv1_char = '?';
                result_str = 'ERR';
                fprintf('Error in %s: %s\n', title, ME.message);
                x0scal = -1; % dummy
            end
            
            % Accumulate stats
            row.title = title;
            row.n = n;
            row.p = p;
            row.iter = iter;
            row.fevals = fevals;
            row.iv1 = iv1_char;
            row.x0scal = x0scal;
            row.resnorm = resnorm;
            row.result = result_str;
            
            % Handle potential struct array init issues by ensuring array fields match
            if isempty(results)
                results = row;
            else
                results(end+1) = row;
            end
        end
    end
    
    % Print Summary
    fprintf('\n\n  Summary of test runs.\n');
    fprintf('  iv1_code: indicates exit reason.\n\n');
    fprintf(' problem        n   p  niter   nf  iv1  x0scal     final f   result\n');
    fprintf('%s\n', repmat('-', 1, 80));
    
    for i = 1:length(results)
        r = results(i);
        fprintf(' %-12s %3d %3d %6d %4d   %c   %5.1f  %10.3e   %s\n', ...
            r.title, r.n, r.p, r.iter, r.fevals, r.iv1, r.x0scal, r.resnorm, r.result);
    end
end

function [r, J] = problem_wrapper(nex, x, n, p)
    r = get_residual(nex, x, n, p);
    if nargout > 1
        J = get_jacobian(nex, x, n, p);
    end
end

function r = get_residual(nex, x, n, p)
    r = zeros(n, 1);
    
    % Data arrays (static)
    persistent ybard ykow ymeyer yosb1 yosb2
    if isempty(ybard)
        ybard = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, ...
                 0.73, 0.96, 1.34, 2.10, 4.39]';
        ykow = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, ...
                0.0323, 0.0235, 0.0246]';
        ymeyer = [34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744, ...
                  8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872]';
        yosb1 = [0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818, 0.784, ...
                 0.751, 0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558, 0.538, 0.522, ...
                 0.506, 0.490, 0.478, 0.467, 0.457, 0.448, 0.438, 0.431, 0.424, 0.420, ...
                 0.414, 0.411, 0.406]';
        yosb2 = [1.366, 1.191, 1.112, 1.013, 0.991, ...
                 0.885, 0.831, 0.847, 0.786, 0.725, ...
                 0.746, 0.679, 0.608, 0.655, 0.616, ...
                 0.606, 0.602, 0.626, 0.651, 0.724, ...
                 0.649, 0.649, 0.694, 0.644, 0.624, ...
                 0.661, 0.612, 0.558, 0.533, 0.495, ...
                 0.500, 0.423, 0.395, 0.375, 0.372, ...
                 0.391, 0.396, 0.405, 0.428, 0.429, ...
                 0.523, 0.562, 0.607, 0.653, 0.672, ...
                 0.708, 0.633, 0.668, 0.645, 0.632, ...
                 0.591, 0.559, 0.597, 0.625, 0.739, ...
                 0.710, 0.729, 0.720, 0.636, 0.581, ...
                 0.428, 0.292, 0.162, 0.098, 0.054]';
    end
    
    ukow = [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]';
    
    switch nex
        case 1 % Rosenbrock
            r(1) = 10.0 * (x(2) - x(1)^2);
            r(2) = 1.0 - x(1);
            
        case 2 % Helix
            theta = atan2(x(2), x(1)) / (2*pi);
            if (x(1) <= 0 && x(2) <= 0)
                theta = theta + 1.0;
            end
            r(1) = 10.0 * (x(3) - 10.0 * theta);
            r(2) = 10.0 * (sqrt(x(1)^2 + x(2)^2) - 1.0);
            r(3) = x(3);
            
        case 3 % Singular
            r(1) = x(1) + 10.0 * x(2);
            r(2) = sqrt(5.0) * (x(3) - x(4));
            r(3) = (x(2) - 2.0 * x(3))^2;
            r(4) = sqrt(10.0) * (x(1) - x(4))^2;
            
        case 4 % Woods
            r(1) = 10.0 * (x(2) - x(1)^2);
            r(2) = 1.0 - x(1);
            r(3) = sqrt(90.0) * (x(4) - x(3)^2);
            r(4) = 1.0 - x(3);
            r(5) = sqrt(9.9) * (x(2) + x(4) - 2.0);
            t = sqrt(0.2);
            r(6) = t * (x(2) - 1.0);
            r(7) = t * (x(4) - 1.0);
            
        case 5 % Zangwill
            r(1) = x(1) - x(2) + x(3);
            r(2) = -x(1) + x(2) + x(3);
            r(3) = x(1) + x(2) - x(3);
            
        case 6 % Engvall
            r(1) = x(1)^2 + x(2)^2 + x(3)^2 - 1.0;
            r(2) = x(1)^2 + x(2)^2 + (x(3) - 2.0)^2 - 1.0;
            r(3) = x(1) + x(2) + x(3) - 1.0;
            r(4) = x(1) + x(2) - x(3) + 1.0;
            r(5) = x(1)^3 + 3.0*x(2)^2 + (5.0*x(3) - x(1) + 1.0)^2 - 36.0;
            
        case 7 % Branin
            r(1) = 4.0 * (x(1) + x(2));
            r(2) = r(1) + (x(1) - x(2)) * ((x(1) - 2.0)^2 + x(2)^2 - 1.0);
            
        case 8 % Beale
            r(1) = 1.5 - x(1) * (1.0 - x(2));
            r(2) = 2.25 - x(1) * (1.0 - x(2)^2);
            r(3) = 2.625 - x(1) * (1.0 - x(2)^3);
            
        case 9 % Cragg and Levy
            r(1) = (exp(x(1)) - x(2))^2;
            r(2) = 10.0 * (x(2) - x(3))^3;
            r(3) = (sin(x(3) - x(4)) / cos(x(3) - x(4)))^2;
            r(4) = x(1)^4;
            r(5) = x(4) - 1.0;
            
        case 10 % Box
            expmax = 1.999 * log(realmax);
            expmin = 1.999 * log(realmin);
            if min([x(1), x(2), x(3)]) <= -expmax
               error('Box: invalid input'); 
            end
            for i = 1:10
               ti = -0.1 * i;
               t1 = ti * x(1);
               if t1 <= expmin
                   e1 = 0;
               else
                   e1 = exp(t1);
               end
               t2 = ti * x(2);
               if t2 <= expmin
                   e2 = 0;
               else
                   e2 = exp(t2);
               end
               r(i) = (e1 - e2) - x(3) * (exp(ti) - exp(10.0*ti));
            end
            
        case 11 % Davidon 1
             for i = 1:n-1
                 ti = double(i);
                 t = 1.0;
                 tmp_sum = 0.0;
                 for j = 1:p
                     tmp_sum = tmp_sum + t * x(j);
                     t = t * ti;
                 end
                 r(i) = tmp_sum;
             end
             r(n) = x(1) - 1.0;
             
        case 12 % Freudenstein and Roth
             r(1) = -13.0 + x(1) - 2.0*x(2) + 5.0*x(2)^2 - x(2)^3;
             r(2) = -29.0 + x(1) - 14.0*x(2) + x(2)^2 + x(2)^3;
             
        case {13, 14, 15, 16} % Watson
            for i = 1:29
                ti = i / 29.0;
                r1 = 0.0;
                r2 = x(1);
                t = 1.0;
                for j = 2:p
                   r1 = r1 + (j-1)*t*x(j);
                   t = t * ti;
                   r2 = r2 + t*x(j);
                end
                r(i) = r1 - r2^2 - 1.0;
            end
            r(30) = x(1);
            r(31) = x(2) - x(1)^2 - 1.0;
            if nex >= 33 % +10 variants
                r = r + 10.0;
            end
            
        case 17 % Chebyquad
            r(:) = 0.0;
            for j=1:n
               tim1 = 1.0;
               ti = 2.0*x(j) - 1.0;
               z = 2.0*ti;
               for i=1:n
                  r(i) = r(i) + ti;
                  tip1 = z * ti - tim1;
                  tim1 = ti;
                  ti = tip1;
               end
            end
            for i=1:n
               ti = 0.0;
               if mod(i,2) == 0
                  ti = -1.0 / (i*i - 1); 
               end
               r(i) = ti - r(i)/n;
            end
            
        case 18 % Brown and Dennis
            for i=1:n
               ti = 0.2 * i;
               r(i) = (x(1) + x(2)*ti - exp(ti))^2 + (x(3) + x(4)*sin(ti) - cos(ti))^2; 
            end
            
        case 19 % Bard
            for i=1:15
               u = double(i);
               v = 16.0 - u;
               w = min(u, v);
               r(i) = ybard(i) - (x(1) + u / (x(2)*v + x(3)*w));
            end
            
         case 20 % Jennrich and Sampson
             for i=1:10
                ti = double(i);
                r(i) = 2.0 + 2.0*ti - (exp(ti*x(1)) + exp(ti*x(2)));
             end
             
         case 21 % Kowalik
             for i=1:11
                r(i) = ykow(i) - x(1) * (ukow(i)^2 + x(2)*ukow(i)) / (ukow(i)^2 + x(3)*ukow(i) + x(4)); 
             end
             
         case 22 % Osborne 1
             for i=1:33
                ti = 10.0 * (1-i);
                r(i) = yosb1(i) - (x(1) + x(2)*exp(x(4)*ti) + x(3)*exp(x(5)*ti));
             end
             
         case 23 % Osborne 2
             uftolg = 1.999 * log(realmin);
             for i=1:65
                ti = 0.1 * (1-i);
                ri = x(1) * exp(x(5)*ti);
                for j=2:4
                   theta = -x(j+4) * (ti + x(j+7))^2;
                   if theta <= uftolg
                       t = 0.0;
                   else
                       t = exp(theta);
                   end
                   ri = ri + x(j)*t;
                end
                r(i) = yosb2(i) - ri;
             end
             
         case 24 % Madsen
             r(1) = x(1)^2 + x(2)^2 + x(1)*x(2);
             r(2) = sin(x(1));
             r(3) = cos(x(2));
             
         case 25 % Meyer
             for i=1:16
                ti = 5*i + 45;
                r(i) = x(1) * exp(x(2)/(ti + x(3))) - ymeyer(i);
             end
         
         case {26, 27, 28, 29} % Brown
             r(1:n-1) = x(1:n-1) + sum(x(1:n)) - (n+1);
             r(n) = prod(x(1:n)) - 1.0;
             
         case 30 % Bard + 10
             for i=1:15
               u = double(i);
               v = 16.0 - u;
               w = min(u, v);
               r(i) = ybard(i) - (x(1) + u / (x(2)*v + x(3)*w)) + 10.0;
             end
             
         case 31 % Kowalik + 10
             for i=1:11
                r(i) = ykow(i) - x(1) * (ukow(i)^2 + x(2)*ukow(i)) / (ukow(i)^2 + x(3)*ukow(i) + x(4)) + 10.0; 
             end
             
         case 32 % Meyer + 10
             for i=1:16
                ti = 5*i + 45;
                r(i) = x(1) * exp(x(2)/(ti + x(3))) - ymeyer(i) + 10.0;
             end
             
         case {33, 34, 35, 36} % Watson + 10
             % Handled in Watson case via range check
             % Recalling local function recursively? No, just copy pasta is safer or calling helper.
             % Re-implementing to avoid complexity:
             for i = 1:29
                ti = i / 29.0;
                r1 = 0.0;
                r2 = x(1);
                t = 1.0;
                for j = 2:p
                   r1 = r1 + (j-1)*t*x(j);
                   t = t * ti;
                   r2 = r2 + t*x(j);
                end
                r(i) = r1 - r2^2 - 1.0 + 10.0;
            end
            r(30) = x(1) + 10.0;
            r(31) = x(2) - x(1)^2 - 1.0 + 10.0;
             
        otherwise
            error('Unknown problem %d', nex);
    end
end

function J = get_jacobian(nex, x, n, p)
    J = zeros(n, p);
    ukow = [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]';

    switch nex
        case 1 % Rosenbrock
            J(1,1) = -20.0 * x(1);
            J(1,2) = 10.0;
            J(2,1) = -1.0;
            J(2,2) = 0.0;
            
        case 2 % Helix
            t = x(1)^2 + x(2)^2;
            ti = 100.0 / (2*pi * t);
            J(1,1) = ti * x(2);
            t = 10.0 / sqrt(t);
            J(2,1) = x(1) * t;
            J(3,1) = 0.0;
            J(1,2) = -ti * x(1);
            J(2,2) = x(2) * t;
            J(3,2) = 0.0;
            J(1,3) = 10.0;
            J(2,3) = 0.0;
            J(3,3) = 1.0;
            
        case 3 % Singular
            J(1,1) = 1.0;
            J(1,2) = 10.0;
            J(2,3) = sqrt(5.0);
            J(2,4) = -J(2,3);
            J(3,2) = 2.0 * (x(2) - 2.0*x(3));
            J(3,3) = -2.0 * J(3,2);
            J(4,1) = sqrt(40.0) * (x(1) - x(4));
            J(4,4) = -J(4,1);
            
        case 4 % Woods
            J(1,1) = -20.0 * x(1);
            J(1,2) = 10.0;
            J(2,1) = -1.0;
            J(3,4) = sqrt(90.0);
            J(3,3) = -2.0 * x(3) * J(3,4);
            J(4,3) = -1.0;
            J(5,2) = sqrt(9.9);
            J(5,4) = J(5,2);
            J(6,2) = sqrt(0.2);
            J(7,4) = J(6,2);
            
        case 5 % Zangwill
            J(:,:) = 1.0;
            J(1,2) = -1.0;
            J(2,1) = -1.0;
            J(3,3) = -1.0;
            
        case 6 % Engvall
            J(1,1) = 2.0 * x(1);
            J(1,2) = 2.0 * x(2);
            J(1,3) = 2.0 * x(3);
            J(2,1) = J(1,1);
            J(2,2) = J(1,2);
            J(2,3) = 2.0 * (x(3) - 2.0);
            J(3,:) = 1.0;
            J(4,:) = 1.0;
            J(4,3) = -1.0;
            t = 2.0 * (5.0*x(3) - x(1) + 1.0);
            J(5,1) = 3.0*x(1)^2 - t;
            J(5,2) = 6.0*x(2);
            J(5,3) = 5.0*t;
            
        case 7 % Branin
            J(1,1) = 4.0;
            J(1,2) = 4.0;
            J(2,1) = 3.0 + (x(1)-2.0)*(3.0*x(1) - 2.0*x(2) - 2.0) + x(2)^2;
            J(2,2) = 1.0 + 2.0*(2.0*x(1) - x(2)^2) - (x(1) - x(2))^2;
            
        case 8 % Beale
            J(1,1) = x(2) - 1.0;
            J(1,2) = x(1);
            J(2,1) = x(2)^2 - 1.0;
            J(2,2) = 2.0 * x(1) * x(2);
            J(3,1) = x(2)^3 - 1.0;
            J(3,2) = 3.0 * x(1) * x(2)^2;
            
        case 9 % Cragg and Levy
            t = exp(x(1));
            J(1,2) = -2.0 * (t - x(2));
            J(1,1) = -t * J(1,2);
            J(2,2) = 30.0 * (x(2) - x(3))^2;
            J(2,3) = -J(2,2);
            J(3,3) = 2.0 * sin(x(3)-x(4)) / (cos(x(3)-x(4)))^3;
            J(3,4) = -J(3,3);
            J(4,1) = 4.0 * x(1)^3;
            J(5,4) = 1.0;
            
        case 10 % Box
            expmin = 1.999 * log(realmin);
            for i=1:10
               ti = -0.1 * i;
               
               t = x(1) * ti;
               if t < expmin
                   e = 0;
               else
                   e = exp(t);
               end
               J(i,1) = ti * e;
               
               t = x(2) * ti;
               if t < expmin
                   e = 0;
               else
                   e = exp(t);
               end
               J(i,2) = -ti * e;
               J(i,3) = exp(10.0*ti) - exp(ti);
            end
            
        case 11 % Davidon 1
            for i=1:n-1
               ti = double(i);
               t = 1.0;
               for k=1:p
                  J(i,k) = t;
                  t = t * ti;
               end
            end
            J(n,1) = 1.0;
            
        case 12 % Freudenstein
            J(1,1) = 1.0;
            J(1,2) = -2.0 + x(2)*(10.0 - 3.0*x(2));
            J(2,1) = 1.0;
            J(2,2) = -14.0 + x(2)*(2.0 + 3.0*x(2));
            
        case {13, 14, 15, 16, 33, 34, 35, 36} % Watson
            for i=1:29
               ti = i / 29.0;
               r2 = x(1);
               t = 1.0;
               for k=2:p
                  t = t * ti;
                  r2 = r2 + t*x(k);
               end
               r2 = -2.0 * r2;
               J(i,1) = r2;
               t = 1.0;
               r2 = ti * r2;
               for k=2:p
                  J(i,k) = t * (double(k-1) + r2);
                  t = t * ti;
               end
            end
            J(30,1) = 1.0;
            J(31,1) = -2.0 * x(1);
            J(31,2) = 1.0;
            
        case 17 % Chebyquad
            for k=1:n
               tim1 = -1.0/n;
               z = 2.0*x(k) - 1.0;
               ti = z * tim1;
               tpim1 = 0.0;
               tpi = 2.0*tim1;
               z = z + z;
               for i=1:n
                  J(i,k) = tpi;
                  tpip1 = 4.0*ti + z*tpi - tpim1;
                  tpim1 = tpi;
                  tpi = tpip1;
                  tip1 = z*ti - tim1;
                  tim1 = ti;
                  ti = tip1;
               end
            end
            
        case 18 % Brown and Dennis
            for i=1:n
               ti = 0.2*i;
               J(i,1) = 2.0 * (x(1) + x(2)*ti - exp(ti));
               J(i,2) = ti * J(i,1);
               t = sin(ti);
               J(i,3) = 2.0 * (x(3) + x(4)*t - cos(ti));
               J(i,4) = t * J(i,3);
            end
            
        case {19, 30} % Bard
            for i=1:15
               J(i,1) = -1.0;
               u = double(i);
               v = 16.0 - u;
               w = min(u, v);
               t = u / (x(2)*v + x(3)*w)^2;
               J(i,2) = v * t;
               J(i,3) = w * t;
            end
            
        case 20 % Jennrich
            for i=1:10
               ti = double(i);
               J(i,1) = -ti * exp(ti * x(1));
               J(i,2) = -ti * exp(ti * x(2));
            end
            
        case {21, 31} % Kowalik
            for i=1:11
               t = -1.0 / (ukow(i)^2 + x(3)*ukow(i) + x(4));
               J(i,1) = t * (ukow(i)^2 + x(2)*ukow(i));
               J(i,2) = x(1) * ukow(i) * t;
               t = t * J(i,1) * x(1);
               J(i,3) = ukow(i) * t;
               J(i,4) = t;
            end
            
        case 22 % Osborne 1
            for i=1:33
               ti = 10.0 * (1-i);
               J(i,1) = -1.0;
               J(i,2) = -exp(x(4) * ti);
               J(i,3) = -exp(x(5) * ti);
               J(i,4) = ti * x(2) * J(i,2);
               J(i,5) = ti * x(3) * J(i,3);
            end
            
        case 23 % Osborne 2
            uftolg = 1.999 * log(realmin);
            for i=1:65
               ti = 0.1 * (1-i);
               J(i,1) = -exp(x(5) * ti);
               J(i,5) = x(1) * ti * J(i,1);
               for k=2:4
                  t = x(k+7) + ti;
                  theta = -x(k+4)*t*t;
                  if theta <= uftolg
                      r2 = 0.0;
                  else
                      r2 = -exp(theta);
                  end
                  J(i,k) = r2;
                  r2 = -t * r2 * x(k);
                  J(i,k+4) = r2 * t;
                  J(i,k+7) = 2.0 * x(k+4) * r2;
               end
            end
        
        case 24 % Madsen
            J(1,1) = 2.0*x(1) + x(2);
            J(1,2) = 2.0*x(2) + x(1);
            J(2,1) = cos(x(1));
            J(2,2) = 0.0;
            J(3,1) = 0.0;
            J(3,2) = -sin(x(2));
            
        case {25, 32} % Meyer
            for i=1:16
               ti = 5*i + 45;
               u = ti + x(3);
               t = exp(x(2)/u);
               J(i,1) = t;
               J(i,2) = x(1) * t / u;
               J(i,3) = -x(1) * x(2) * t / (u*u);
            end
            
        case {26, 27, 28, 29} % Brown
            for k=1:n
               for i=1:n-1
                   if i == k
                       J(i,k) = 2.0;
                   else
                       J(i,k) = 1.0;
                   end
               end
            end
            for k=1:n
               t = 1.0;
               for i=1:n
                   if i ~= k
                       t = t * x(i);
                   end
               end
               J(n,k) = t;
            end
            
        otherwise
            error('Unknown problem %d (Jacobian)', nex);
    end
end


function x = get_xinit(nex, p)
    x = zeros(p, 1);
    switch nex
        case 1 % Rosenbrock
            x(1) = -1.2; x(2) = 1.0;
        case 2 % Helix
            x(1) = -1.0; x(2) = 0.0; x(3) = 0.0;
        case 3 % Singular
            x(1) = 3.0; x(2) = -1.0; x(3) = 0.0; x(4) = 1.0;
        case 4 % Woods
            x(1) = -3.0; x(2) = -1.0; x(3) = -3.0; x(4) = -1.0;
        case 5 % Zangwill
            x(1) = 100.0; x(2) = -1.0; x(3) = 2.5;
        case 6 % Engvall
            x(1) = 1.0; x(2) = 2.0; x(3) = 0.0;
        case 7 % Branin
            x(1) = 2.0; x(2) = 0.0;
        case 8 % Beale
            x(1) = 0.1; x(2) = 0.1;
        case 9 % Cragg
            x(1) = 1.0; x(2) = 2.0; x(3) = 2.0; x(4) = 2.0;
        case 10 % Box
            x(1) = 0.0; x(2) = 10.0; x(3) = 20.0;
        case 11 % Davidon 1
            x(:) = 0.0;
        case 12 % Freudenstein
            x(1) = 15.0; x(2) = -2.0;
        case {13, 14, 15, 16} % Watson
            x(:) = 0.0;
        case 17 % Chebyquad
            for i=1:p
               x(i) = double(i) / (p+1);
            end
        case 18 % Brown and Dennis
            x(1) = 25.0; x(2) = 5.0; x(3) = -5.0; x(4) = -1.0;
        case 19 % Bard
            x(1) = 1.0; x(2) = 1.0; x(3) = 1.0;
        case 20 % Jennrich
            x(1) = 0.3; x(2) = 0.4;
        case 21 % Kowalik
            x(1) = 0.25; x(2) = 0.39; x(3) = 0.415; x(4) = 0.39;
        case 22 % Osborne 1
            x(1) = 0.5; x(2) = 1.5; x(3) = -1.0; x(4) = 0.01; x(5) = 0.02;
        case 23 % Osborne 2
            vals = [1.3, 0.65, 0.65, 0.7, 0.60, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5];
            x = vals(:);
        case 24 % Madsen
            x(1) = 3.0; x(2) = 1.0;
        case 25 % Meyer
            x(1) = 0.02; x(2) = 4000.0; x(3) = 250.0;
        case {26, 27} % Brown
            x(:) = 0.5;
        case 30 % Bard+10
            x(1) = 1.0; x(2) = 1.0; x(3) = 1.0;
        case 31 % Kowal+10
            x(1) = 0.25; x(2) = 0.39; x(3) = 0.415; x(4) = 0.39;
        case 32 % Meyer+10
            x(1) = 0.02; x(2) = 4000.0; x(3) = 250.0;
        case {33, 34, 35, 36} % Watson+10
            x(:) = 0.0;
            
        otherwise
            error('Unknown problem %d (xinit)', nex);
    end
end
