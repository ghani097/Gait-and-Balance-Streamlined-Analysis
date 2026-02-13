function run_gb_pipeline(rawFolder, outcomeFolder, options)
%RUN_GB_PIPELINE Streamlined Gait & Balance pipeline for GB raw data.
%
% Usage:
%   run_gb_pipeline()
%   run_gb_pipeline(rawFolder, outcomeFolder)
%   run_gb_pipeline(rawFolder, outcomeFolder, options)
%
% Options fields:
%   fs               (default: 100)
%   isAndroid        (default: 0)  % iOS=0, Android=1
%   defaultHeightM   (default: 1.68)
%   heightsFile      (default: '')
%       CSV/XLSX with Name + Height_cm/Height_m or Height (cm)
%   preferBaseFile   (default: true) % Prefer base file if -1/-2 exists
%   fillMissingHeightWithMean (default: true)
%   verbose          (default: true)
%
% Output:
%   Writes "Outcomes for {Name}.csv" into outcomeFolder.

%% Resolve paths and defaults
thisDir                     = fileparts(mfilename('fullpath'));
repoRoot                    = fullfile(thisDir, '..');

if nargin < 1 || isempty(rawFolder)
    rawFolder               = fullfile(repoRoot, 'Raw_Sensor_Data');
end
if nargin < 2 || isempty(outcomeFolder)
    outcomeFolder           = fullfile(repoRoot, 'Outcome_Data_Replicated');
end
if nargin < 3
    options                 = struct();
end

options                     = applyDefaults(options);

if ~exist(outcomeFolder, 'dir')
    mkdir(outcomeFolder);
end

% Add SincMotion MATLAB library
addpath(genpath(fullfile(repoRoot, 'sincmotion-matlab')));

%% Load participant heights if provided
heightLookup                = buildHeightLookup(options);

%% Load file list
files                       = dir(fullfile(rawFolder, '*.csv'));
if isempty(files)
    error('No CSV files found in %s', rawFolder);
end

% Build base-name map to handle -1/-2 duplicates
baseNameMap                 = containers.Map('KeyType', 'char', 'ValueType', 'logical');
for i = 1:numel(files)
    baseName                = stripSuffix(files(i).name);
    if strcmp(baseName, files(i).name)
        baseNameMap(baseName) = true;
    elseif ~isKey(baseNameMap, baseName)
        baseNameMap(baseName) = false;
    end
end

%% Parse and compute outcomes
participantRows             = containers.Map('KeyType', 'char', 'ValueType', 'any');
participantNames            = containers.Map('KeyType', 'char', 'ValueType', 'char');

for i = 1:numel(files)
    fileName                = files(i).name;

    % Skip non-raw outcome files
    if startsWith(fileName, 'Outcomes for ', 'IgnoreCase', true)
        continue;
    end
    if contains(fileName, 'Balance Study_', 'IgnoreCase', true)
        continue;
    end

    baseName                = stripSuffix(fileName);
    if options.preferBaseFile && ~strcmp(baseName, fileName) ...
            && isKey(baseNameMap, baseName) && baseNameMap(baseName)
        % Skip duplicate -1/-2 if base exists
        continue;
    end

    meta                    = parseFileName(baseName);
    if isempty(meta)
        if options.verbose
            fprintf('Skipping unrecognized file name: %s\n', fileName);
        end
        continue;
    end

    filePath                = fullfile(rawFolder, fileName);

    % Load data
    [accelData, rotData, timeVect, gyroData] ...
                            = loadGnBExportedFile(options.fs, filePath);

    % Compute outcomes
    if startsWith(meta.test, 'Walk', 'IgnoreCase', true)
        personHeight        = lookupHeight(heightLookup, meta.name, options.defaultHeightM);
        outcomes            = estimateGnBGaitOutcomes(timeVect, accelData, ...
                                rotData, gyroData, options.fs, ...
                                personHeight, options.isAndroid, 0);
        row                 = makeGaitRow(meta, outcomes);
    else
        outcomes            = estimateGnBStaticOutcomes(accelData, rotData, ...
                                options.fs, options.isAndroid);
        row                 = makeStaticRow(meta, outcomes);
    end

    key                     = lower(strtrim(meta.name));
    if ~isKey(participantRows, key)
        participantRows(key) = {row};
        participantNames(key) = meta.name;
    else
        rows                = participantRows(key);
        rows{end+1}         = row;
        participantRows(key) = rows;
    end
end

%% Write outcomes per participant
participantKeys              = keys(participantRows);
for i = 1:numel(participantKeys)
    key                      = participantKeys{i};
    name                     = participantNames(key);
    rows                     = participantRows(key);
    rows                     = sortRows(rows);

    outFile                  = fullfile(outcomeFolder, ...
                                sprintf('Outcomes for %s.csv', name));
    writeOutcomesCsv(outFile, rows);
    if options.verbose
        fprintf('Wrote %s\n', outFile);
    end
end
end

%% Helper: options defaults
function options = applyDefaults(options)
    if ~isfield(options, 'fs');             options.fs = 100; end
    if ~isfield(options, 'isAndroid');      options.isAndroid = 0; end
    if ~isfield(options, 'defaultHeightM'); options.defaultHeightM = 1.68; end
    if ~isfield(options, 'heightsFile')
        defaultHeights = fullfile(fileparts(mfilename('fullpath')), '..', ...
            'Participant HeightWeight.xlsx');
        if exist(defaultHeights, 'file')
            options.heightsFile = defaultHeights;
        else
            options.heightsFile = '';
        end
    end
    if ~isfield(options, 'preferBaseFile'); options.preferBaseFile = true; end
    if ~isfield(options, 'fillMissingHeightWithMean')
        options.fillMissingHeightWithMean = true;
    end
    if ~isfield(options, 'verbose');        options.verbose = true; end
end

%% Helper: strip -1/-2 suffix before .csv
function baseName = stripSuffix(fileName)
    baseName = regexprep(fileName, '-\d+(?=\.csv$)', '');
end

%% Helper: parse filename into metadata
function meta = parseFileName(fileName)
    meta = struct();
    pattern = '^(?<name>.+?)\s+Test set\s+(?<testset>\d+)\s+on\s+(?<date>\d{1,2}-\d{1,2}-\d{4})\s+(?<test>.+?)\.csv$';
    tokens = regexp(fileName, pattern, 'names');
    if isempty(tokens)
        meta = [];
        return;
    end
    meta.name       = strtrim(tokens.name);
    meta.testset    = str2double(tokens.testset);
    meta.dateRaw    = tokens.date;
    meta.date       = datetime(tokens.date, 'InputFormat', 'd-M-yyyy');
    meta.test       = strtrim(tokens.test);
end

%% Helper: build height lookup
function lookup = buildHeightLookup(options)
    lookup = containers.Map('KeyType', 'char', 'ValueType', 'double');
    if isempty(options.heightsFile) || ~exist(options.heightsFile, 'file')
        return;
    end
    t = readtable(options.heightsFile, 'VariableNamingRule', 'preserve');
    % normalize columns: lower, trim, replace non-alnum with underscore
    rawNames = string(t.Properties.VariableNames);
    normNames = lower(rawNames);
    normNames = regexprep(normNames, '[^a-z0-9]+', '_');
    nameIdx = find(strcmp(normNames, 'name'), 1);
    hcmIdx  = find(strcmp(normNames, 'height_cm') | strcmp(normNames, 'height_cm_'), 1);
    hmIdx   = find(strcmp(normNames, 'height_m') | strcmp(normNames, 'height_m_'), 1);
    if isempty(nameIdx) || (isempty(hcmIdx) && isempty(hmIdx))
        warning('Heights file missing Name and Height column. Ignoring.');
        return;
    end
    % compute mean height (meters) for missing entries
    if ~isempty(hmIdx)
        hcol = t{:, hmIdx};
    else
        hcol = t{:, hcmIdx} ./ 100;
    end
    meanHeight = mean(hcol(~isnan(hcol)));
    for i = 1:height(t)
        nm = strtrim(string(t{i, nameIdx}));
        if strlength(nm) == 0
            continue;
        end
        if ~isempty(hmIdx)
            h = t{i, hmIdx};
        else
            h = t{i, hcmIdx} / 100;
        end
        if isnan(h) && options.fillMissingHeightWithMean
            h = meanHeight;
        end
        if ~isnan(h)
            lookup(lower(nm)) = h;
        end
    end
end

function h = lookupHeight(lookup, name, defaultHeight)
    key = lower(strtrim(name));
    if isKey(lookup, key)
        h = lookup(key);
    else
        h = defaultHeight;
    end
end

%% Helper: build row for gait outcome
function row = makeGaitRow(meta, outcomes)
    row = cell(1, 14);
    row{1}  = datestr(meta.date, 'yyyy-mm-dd');
    row{2}  = meta.testset;
    row{3}  = meta.test;
    row{4}  = [];
    row{5}  = [];
    row{6}  = [];
    row{7}  = outcomes(1);  % Walking balance (%)
    row{8}  = outcomes(2);  % Step length
    row{9}  = outcomes(5);  % Step time
    row{10} = outcomes(8);  % Step length variability
    row{11} = outcomes(9);  % Step time variability
    row{12} = outcomes(10); % Step length asymmetry
    row{13} = outcomes(11); % Step time asymmetry
    row{14} = outcomes(12); % Walking speed
end

%% Helper: build row for static outcome
function row = makeStaticRow(meta, outcomes)
    row = cell(1, 14);
    row{1}  = datestr(meta.date, 'yyyy-mm-dd');
    row{2}  = meta.testset;
    row{3}  = meta.test;
    row{4}  = outcomes(1); % Stability
    row{5}  = outcomes(2); % Stability ML
    row{6}  = outcomes(3); % Stability AP
    row{7}  = [];
    row{8}  = [];
    row{9}  = [];
    row{10} = [];
    row{11} = [];
    row{12} = [];
    row{13} = [];
    row{14} = [];
end

%% Helper: sort rows by date then test order
function sortedRows = sortRows(rows)
    order = {'Firm EO','Firm EC','Compliant EO','Compliant EC','Walk HF','Walk HT'};
    n = numel(rows);
    dateNum = NaN(n, 1);
    ords = zeros(n, 1);
    for i = 1:n
        try
            dt = datetime(rows{i}{1}, 'InputFormat', 'yyyy-MM-dd');
            if ~isnat(dt)
                dateNum(i) = datenum(dt);
            end
        catch
            dateNum(i) = NaN;
        end
        test = rows{i}{3};
        idx = find(strcmp(order, test), 1);
        if isempty(idx)
            ords(i) = 99;
        else
            ords(i) = idx;
        end
    end
    [~, ix] = sortrows([dateNum, ords]);
    sortedRows = rows(ix);
end

%% Helper: write CSV with blank fields for missing values
function writeOutcomesCsv(filePath, rows)
    headers = {'Date','Test set','Test','Stability(-ln[m/s²])', ...
        'Stability ML(-ln[m/s²])','Stability AP(-ln[m/s²])', ...
        'Walking balance(%)','Step length(m)','Step time(s)', ...
        'Step length variability(%)','Step time variability(%)', ...
        'Step length asymmetry(%)','Step time asymmetry(%)', ...
        'Walking speed(m/s)'};

    fid = fopen(filePath, 'w');
    fprintf(fid, '%s\n', strjoin(headers, ','));
    for i = 1:numel(rows)
        row = rows{i};
        fields = cell(1, numel(headers));
        for j = 1:numel(headers)
            v = row{j};
            if isempty(v)
                fields{j} = '';
            elseif isnumeric(v)
                fields{j} = sprintf('%.17g', v);
            else
                fields{j} = char(v);
            end
        end
        fprintf(fid, '%s\n', strjoin(fields, ','));
    end
    fclose(fid);
end
