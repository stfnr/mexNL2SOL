function setup_gfortran()
%SETUP_GFORTRAN Summary of this function goes here
%   This function was implemented base on the information given here:
%   https://de.mathworks.com/matlabcentral/answers/338303-how-to-set-up-mex-with-gfortran-on-mac#answer_285758

if ismac
    if strcmpi(computer, 'MACI64')

        setenv('PATH', [getenv('PATH') ':/usr/local/bin'])
        [status, cmdout] = system('realpath $(which gfortran)');

        if ~status
            gfortran_dir = fileparts(cmdout);
        else
            error('Could not determine gfortran directory. Check if gfortran is installed using the cmd line: which gfortran')
        end

        % get macOS version
        [status, cmdout] = system('sw_vers');
        expression = '(?<major>\d+).(?<minor>\d+).(?<micro>\d+)';
        macos_version = regexp(cmdout,expression,'names');

        copyfile(['.' filesep 'mexconfig' filesep 'macos' filesep 'gfortran.xml'], [matlabroot filesep 'bin/maci64/mexopts/'])
        copyfile(['.' filesep 'mexconfig' filesep 'macos' filesep 'gfortrani8'], gfortran_dir)
        system(['chmod +x ' gfortran_dir '/gfortrani8'])
        setenv('DYLD_LIBRARY_PATH', [matlabroot(),'/bin/maci64:', matlabroot(), '/sys/os/maci64:', getenv('DYLD_LIBRARY_PATH')])

        warning('The following commands need to be manually performed using sudo from the command line:')
        disp(['sudo mkdir -p /Library/Developer/CommandLineTools/Platforms/MacOSX.platform/Developer/SDKs/MacOSX' macos_version(1).major '.' macos_version(1).minor '.sdk'])
        disp(['sudo cp -r /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk /Library/Developer/CommandLineTools/Platforms/MacOSX.platform/Developer/SDKs/MacOSX' macos_version(1).major '.' macos_version(1).minor '.sdk'])

        disp('Afterwards run the following commands in Matlab: ')
        disp('mex -setup -v C')
        disp('mex -setup -v FORTRAN')

    elseif strcmpi(computer, 'MACA64')

        setenv('PATH', [getenv('PATH') ':/opt/homebrew/bin'])
        [status, cmdout] = system('realpath $(which gfortran-12)');

        if ~status
            gfortran_dir = fileparts(cmdout);
        else
            error('Could not determine gfortran directory. Check if gfortran is installed using the cmd line: which gfortran')
        end

        % get macOS version
        [status, cmdout] = system('sw_vers');
        expression = '(?<major>\d+).(?<minor>\d+).(?<micro>\d+)';
        macos_version = regexp(cmdout,expression,'names');

        copyfile(['.' filesep 'mexconfig' filesep 'maca64' filesep 'gfortran.xml'], [matlabroot filesep 'bin/maca64/mexopts/'])
        copyfile(['.' filesep 'mexconfig' filesep 'maca64' filesep 'gfortrani8'], gfortran_dir)
        system(['chmod +x ' gfortran_dir '/gfortrani8'])
        setenv('DYLD_LIBRARY_PATH', [matlabroot(),'/bin/maca64:', matlabroot(), '/sys/os/maca64:', getenv('DYLD_LIBRARY_PATH')])

        %warning('The following commands need to be manually performed using sudo from the command line:')
        %disp(['sudo mkdir -p /Library/Developer/CommandLineTools/Platforms/MacOSX.platform/Developer/SDKs/MacOSX' macos_version(1).major '.' macos_version(1).minor '.sdk'])
        %disp(['sudo cp -r /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk /Library/Developer/CommandLineTools/Platforms/MacOSX.platform/Developer/SDKs/MacOSX' macos_version(1).major '.' macos_version(1).minor '.sdk'])

        disp('Now run the following commands in Matlab: ')
        disp('mex -setup -v C')
        disp('mex -setup -v FORTRAN')

    else
        error('Unkown computer architecture.')
    end

elseif isunix

    setenv('PATH', [getenv('PATH') ':/usr/local/bin'])

    [status, cmdout] = system('which gcc');
    if status
        error('cannot find gcc. install gcc first.')
    end

    [status, cmdout] = system('which gfortran');
    if status
        error('cannot find gfortran. install gfortran first.')
    else
        gfortran_dir = fileparts(cmdout);
    end

    disp('Copy gfortrani8 script ... (requires sudo)')
    [status, cmdout] = system(['sudo cp .' filesep 'mexconfig' filesep 'linux' filesep 'gfortrani8 ' gfortran_dir filesep 'gfortrani8 '], '-echo');
    disp(cmdout)
    if status
        error('could not copy gfortrani8 script ...')
    end
    disp('Chmod gfortrani8 script ... (requires sudo)')
    [status, cmdout] = system(['sudo chmod 777 ' gfortran_dir filesep 'gfortrani8 '], '-echo');

    disp('Setup mexopts script ... (requires sudo)')
    [status, cmdout] = system(['sudo sed -i ''s+FC="$GFORTRAN_INSTALLDIR/gfortran"+FC="$GFORTRAN_INSTALLDIR/gfortrani8"+g'' ' matlabroot filesep 'bin/glnxa64/mexopts/gfortran6.xml'], '-echo');
    disp(cmdout)
    if status
        error('could not setup mexopts script...')
    end


    mex -setup -v C
    mex -setup -v FORTRAN

elseif ispc
    error('setup_gfortran currently only supports linux and macOS')
else
    error('unkown platform')
end

end