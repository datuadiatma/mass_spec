# mass_spec
This repository contains poorly written but useful scripts that I often use to process raw mass spectrometer data.

## Repository Structure
```bash
├── Element2 (Thermo Element 2 ICP MS)
    ├── load_asc.py    (!OLD VERSION! Python script to clean up E2 raw data (.asc files) into .csv)
    ├── load_asc_files.py    (updated version of load_asc that can handle multiple files)
    
    ├── Calibration_Table    (template to enter concentration data and produce calibration table and curve)
        ├── i_ca_conc_input.csv    (Calibration Table for I/Ca)
        ├── tdigest_conc_input.csv    (Calibration Table for Total Digest Sample)

├── Triton    (Thermo Triton TIMS)
    ├── spike_reduction.py    (spike reduction for Isotope Dilution, or smth)
```
