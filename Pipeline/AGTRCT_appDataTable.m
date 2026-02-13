function appDataTable   = AGTRCT_appDataTable(folderName, sampleRate)

dirContents             = dir(folderName);
dirFiles                = dirContents([dirContents.isdir] ~= 1, :);

numFiles                = length(dirFiles);
appData                 = cell(numFiles, 5 + 4);

for i                   = 1:numFiles
    fileName            = dirFiles(i).name;

    if(strcmp(fileName, '.DS_Store'))
        continue
    end


    disp(fileName)

    if(contains(fileName, 'FP'))
        fprintf('Skipping force plate file: %s\n', fileName);
        continue
    end

    fileTokens          = regexp(fileName, '([0-9]+)_([ABC])_?W[kK]([0-9]+)_([A-Za-z]+)_([A-Za-z]+)\s?.csv', 'tokens');

    if length(fileTokens{1}) ~= 5
        disp(fileTokens{1});
        error 'Invalid file name';
    end

    fileData            = importGnBExportedFile(fullfile(folderName, fileName));

    fileData.Timestamp  = seconds(fileData.Timestamp - fileData.Timestamp(1)) + 1/sampleRate;
    timeVect            = fileData.Timestamp;

    accelData           = [fileData.AccelX fileData.AccelY fileData.AccelZ];
    rotData             = [fileData.QuatW fileData.QuatX fileData.QuatY fileData.QuatZ];
    gyroData            = [fileData.GyroX fileData.GyroY fileData.GyroZ];

    % Pack file data
    personName          = fileTokens{1}{1};
    groupTag            = fileTokens{1}{2};
    weekNo              = fileTokens{1}{3};
    task                = fileTokens{1}{4};
    qual                = fileTokens{1}{5};
    appData(i, 1:5)     = {personName, groupTag, weekNo, task, qual};
    appData{i, 6}       = timeVect;
    appData{i, 7}       = accelData;
    appData{i, 8}       = rotData;
    appData{i, 9}       = gyroData;
end

appData                 = appData(~all(cellfun(@isempty, appData), 2), :);

appDataTable            = cell2table(appData);
appDataTable.Properties.VariableNames ...
    = {'Part', 'Group', 'Week', 'Task', 'Qual', 'TimeVect',...
    'Accel', 'Quat', 'Gyro'};
end