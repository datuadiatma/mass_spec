# mass_spec
This repository contains poorly written but useful scripts that I often use to process raw mass spectrometer data.

## Repository Structure
```bash
├── Element2 (Thermo Element 2 ICP MS)
    ├── load_asc_files.py    (updated version of load_asc that can handle multiple files)
    
    ├── Calibration_Table    (template to enter concentration data and produce calibration table and curve)
        ├── i_ca_conc_input.csv    (Calibration Table for I/Ca)
        ├── tdigest_conc_input.csv    (Calibration Table for Total Digest Sample)

├── Triton    (Thermo Triton TIMS, soon to be publised)
    ├── spike_reduction.py    (spike reduction for Isotope Dilution)
```

## Change Log
```bash
Element2
========
2024/11/17
enable multi-file processing

2024/11/18 
add calibration table template to calculate concentrations

2024/12/30
add blank correction
```