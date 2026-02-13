function sBalanceOutcomes = AGTRCT_StaticBalanceOutcomes(appDataTable)

% Constants
FS                                  = 100;
IS_ANDROID                          = 0;    % Must check

numFiles                            = height(appDataTable);
sBalanceOutcomes                    = cell(numFiles, 5 + 3);
rowNum                              = 1;
for i                               = 1:numFiles
    
    if(strcmp(appDataTable.Task(i), 'Walk'))
        continue;
    end
    
    % Load data
    accelData                       = cell2mat(table2array(appDataTable(i, 'Accel')));
    rotData                         = cell2mat(table2array(appDataTable(i, 'Quat')));
    timeVect                        = cell2mat(table2array(appDataTable(i, 'TimeVect')));
    
    % Plot raw data
    %plot(timeVect(1:length(accelData)), detrend(accelData))
    %ylabel('Acceleration (m/sec/sec)')
    %title('Detrended raw data')
    % Compute and print outcomes
    [outcomes, outcomeString]       = estimateGnBStaticOutcomes(accelData, rotData, FS, IS_ANDROID);
    disp(outcomeString)
    
    sBalanceOutcomes(rowNum, 1:5)   = table2cell(appDataTable(i, 1:5));
    sBalanceOutcomes(rowNum, 6:8)  = num2cell(outcomes);
    rowNum                          = rowNum + 1;
    
    %questAns                        = questdlg('Continute?');
    %if(~strcmp(questAns, 'Yes'))
    %    break;
    %end
end

sBalanceOutcomes                    = array2table(sBalanceOutcomes(1:(rowNum-1), :));

names                               = appDataTable.Properties.VariableNames;
names                               = names(1:5);
names(6:8)                         = {'PS R', 'PS ML', 'PS AP'};
sBalanceOutcomes.Properties.VariableNames = names;
end

