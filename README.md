# mass_spec
This repository contains poorly written but useful scripts that I often use to process raw mass spectrometer data.

## Repository Structure
```bash
├── Element2 (Thermo Element 2 ICP MS)
    ├── load_asc.py    (Python script to clean up E2 raw data (.asc files) into .csv)
    ├── load_asc_files.py    (updated version of load_asc that can handle multiple files)

├── Triton    (Thermo Triton TIMS)
    ├── spike_reduction.py    (spike reduction for Isotope Dilution, or smth)
```