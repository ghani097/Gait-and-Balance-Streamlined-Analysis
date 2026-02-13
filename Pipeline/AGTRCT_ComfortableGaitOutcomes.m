function comfortableGaitOutcomes = AGTRCT_ComfortableGaitOutcomes(appDataTable, partCharacteristics)

% Constants
FS                                  = 100;
IS_ANDROID                          = 0;    % Must check
DEBUG_FLAG                          = 0;
DEBUG_PART                          = '';

numFiles                            = height(appDataTable);
comfortableGaitOutcomes             = cell(numFiles, 5 + 16);
rowNum                              = 1;
for i                               = 1:numFiles
    
    if (~strcmp(DEBUG_PART, '') && ~strcmp(appDataTable.Part(i), DEBUG_PART))
        continue;
    end
    
    if(~strcmp(appDataTable.Task(i), 'Walk'))
        continue;
    end
    
    % Load data
    accelData                       = cell2mat(table2array(appDataTable(i, 'Accel')));
    gyroData                        = cell2mat(table2array(appDataTable(i, 'Gyro')));
    rotData                         = cell2mat(table2array(appDataTable(i, 'Quat')));
    timeVect                        = cell2mat(table2array(appDataTable(i, 'TimeVect')));
    
    % Person Height
    personHeight                    = partCharacteristics.Height_cm(partCharacteristics.ID == strcat(appDataTable.Part(i), appDataTable.Group(i)));
    personHeight                    = personHeight / 100; % Meters
    
    % Compute and print outcomes
    [outcomes, outcomeString]       = estimateGnBGaitOutcomes(timeVect, accelData, rotData, gyroData, FS, personHeight, IS_ANDROID, DEBUG_FLAG);
    disp(outcomeString)
    
    comfortableGaitOutcomes(rowNum, 1:5)   = table2cell(appDataTable(i, 1:5));
    comfortableGaitOutcomes(rowNum, 6:21)  = num2cell(outcomes);
    rowNum                                  = rowNum + 1;
    
    if DEBUG_FLAG
        questAns                        = questdlg('Continute?');
        if(~strcmp(questAns, 'Yes'))
            break;
        end
        close all;
    end
end

comfortableGaitOutcomes             = array2table(comfortableGaitOutcomes(1:(rowNum-1), :));

names                               = appDataTable.Properties.VariableNames;
names                               = names(1:5);
names(6:21)                         = {'Gait symmetry', 'Step length', 'Step length left', 'Step length right', ...
                                        'Step time', 'Step time left', 'Step time right', ...
                                        'Step length var', 'Step time var', 'Step length asym', 'Step time asym', 'Step velocity',...
                                        'Step count lap 1', 'Step count lap 2', 'Step count lap 3', 'Step count lap 4'};
comfortableGaitOutcomes.Properties.VariableNames = names;
end