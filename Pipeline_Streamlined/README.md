# Streamlined GB Pipeline

This pipeline uses the SincMotion MATLAB reference implementation to process
`Raw_Sensor_Data` and produce per-participant outcomes matching the existing
`Outcome_Data` format.

## Usage (MATLAB)

```matlab
run_gb_pipeline()

% or with explicit paths
run_gb_pipeline('E:\GIT_HUB_MAIN\GB_ASRF\Raw_Sensor_Data', ...
                'E:\GIT_HUB_MAIN\GB_ASRF\Outcome_Data_Replicated');

% with options (iOS defaults shown)
opts.fs = 100;
opts.isAndroid = 0;
opts.defaultHeightM = 1.68;
opts.heightsFile = ''; % CSV/XLSX: Name, Height_cm/Height_m or Height (cm)
opts.preferBaseFile = true;
opts.fillMissingHeightWithMean = true;
opts.verbose = true;
run_gb_pipeline([], [], opts);
```

## Dependencies

1. **MATLAB** installed and on PATH.
2. **SincMotion MATLAB** repo available at:
   `E:\GIT_HUB_MAIN\GB_ASRF\sincmotion-matlab`.
3. **wavelib** submodule initialized (currently empty in this workspace):
   - Initialize/update submodule to populate `sincmotion-matlab\wavelib`.
4. **diff_cwtft** MEX built against wavelib.

### Build notes (Windows)

The upstream README includes macOS instructions. On Windows you will need:
1. `cmake` and a C/C++ toolchain (e.g., Visual Studio Build Tools).
2. Build wavelib and produce a library in `wavelib\Bin`.
3. From MATLAB, run:
```matlab
cd('E:\GIT_HUB_MAIN\GB_ASRF\sincmotion-matlab\diff_cwtft')
mex -v -I..\wavelib\header -L..\wavelib\Bin -lwavelib diff_cwtft.c
```

## Notes

- The pipeline prefers base filenames when `-1/-2` duplicates exist.
- It expects filenames like:
  `Name Test set 1 on 16-12-2024 Walk HF.csv`
- If a file does not match the naming convention, it is skipped.
- If `Participant HeightWeight.xlsx` exists in the repo root, it will be used
  by default for heights. Missing heights are filled with the mean height.
