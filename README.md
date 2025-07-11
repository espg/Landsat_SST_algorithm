# Landsat SST Algorithm

This repository provides a modular and cloud-ready pipeline for generating atmospherically corrected sea surface temperatures (SST) from Landsat thermal infrared imagery using an approach tailored to high-latitude applications, currently only validated in West Antarctica at the moment. The pipeline includes tools for atmospheric correction, cross-calibration against MODIS SST, time series generation, and validation with in situ observations.

## 🔁 Workflow Overview

1. **Build MODTRAN Atmospheric Correction Model**  
   - **`ERADownload.ipynb`**  
     Downloads ERA5 atmospheric profiles and SSTs to train the radiative transfer model.  
   - **`MODTRAN4_prep.ipynb`**  
     Prepares ERA5 profiles for MODTRAN ingestion. Output is used to fit the SST retrieval model.
     
2. **Generate Uncalibrated SSTs**  
   - **`SSTpipeline.ipynb`**  
     Main pipeline to produce uncalibrated SSTs from Landsat imagery. Uses monthly retrieval coefficients based on MODTRAN simulations.  
   - **External Inputs:** MODTRAN outputs in `Data/AtmCorrection/` (e.g. `modtran_atmprofiles_01.txt`, `modtran_atmprofiles_01.bts+tau+dbtdsst.txt`)  
   - **Relies on:** functions in `SSTutils.py`
   - **Generates:** Monthly `Data/AtmCorrection/TCWV_01.csv` files and Uncalibrated SST Cloud-Optimized Geotiff files for each Landsat scene at each
     calibration sampling location. The outputs created here that are used in subsequent steps in the manuscript are available at [doi: 10.21227/4ttz-p423](https://dx.doi.org/10.21227/4ttz-p423).

3. **Cross-Calibrate Against MODIS SST**  
   - **`LandsatCalibration.ipynb`**  
     Builds matchups between Landsat SST or Landsat Surface Temperature (ST) and MODIS SST to build calibration relationships using Orthogonal Distance Regression (ODR).
   - **External Inputs:** Uncalibrated SST Cloud-Optimized Geotiffs produced by `SSTpipeline.ipynb` or downloaded from [doi: 10.21227/4ttz-p423](https://dx.doi.org/10.21227/4ttz-p423) as well as      LST Geotiffs only available at that DOI.
   - **Generates:** SST-MODIS and LST-MODIS matchups for each sampling region (e.g., `Data/MODISvLandsat_LST_Burke.csv` and `Data/MODISvLandsat_SST_Burke.csv`), and regression models used to        correct both SST and ST products.

4. **Validate Against In Situ Buoy Data**  
   - **`SSTvalidation.ipynb`**  
     Compares Landsat SST to Argo buoy measurements (e.g., from iQuam) for independent validation.
   - **External Inputs:** iQuam files available for download at [doi: 10.21227/4ttz-p423](https://dx.doi.org/10.21227/4ttz-p423)
   - **Generates:** Uncalibrated SST Cloud-Optimized Geotiff files for each Landsat scene used, validation matchups recorded in `Data/Landsat_validation_201309_201403_1.0.csv`

5. **Build a Time Series**  
   - **`SSTtimeseries.ipynb`**  
     Aggregates SST, MODIS, and Landsat ST products over time and evaluates comparative performance.

## 📂 Repository Structure

```text
├── Data/
│   ├── AtmCorrection/                # MODTRAN-based correction profiles
│   ├── Landsat_validation_*.csv      # Matchups for SST validation
│   ├── MODISvLandsat_*.csv           # Matchups for MODIS-to-Landsat comparisons
│
├── nlsst/                            # (if used for organized pipeline components)
│
├── ERADownload.ipynb                # ERA5 atmospheric profile downloader
├── MODTRAN4_prep.ipynb              # ERA5-to-MODTRAN conversion
├── SSTpipeline.ipynb                # Main SST generation notebook
├── SSTutils.ipynb                   # Notebook used for generation of SST functions
├── SSTutils.py                      # Core Python module of reusable functions
├── LandsatCalibration.ipynb         # MODIS vs. Landsat SST/ST cross-calibration
├── SSTvalidation.ipynb              # Buoy-based validation of Landsat SST
├── landsat_timeseries.ipynb         # Builds and evaluates MODIS vs. SST/ST time series
├── environment.yml                  # Environment dependencies
└── README.md                        # This file
```

## 📦 Dependencies

This project requires a number of Python libraries. Install all dependencies with the provided Conda environment:

```bash
mamba env create -f conda/environment.yml
conda activate sst
```

## ☁️ Cloud-Ready Design

This repository was developed with cloud computing in mind, specifically optimized for the [CryoCloud](https://cryointhecloud.com) virtual research environment. It supports modular, reproducible workflows ideal for high-latitude cryosphere research.

**Key features:**

- Works seamlessly in JupyterHub environments
- Scalable and modular pipeline design
- Compatible with distributed compute platforms
- Built using community-driven open science tools

> **CryoCloud JupyterHub DOI:**  
> [https://doi.org/10.5281/zenodo.7576601](https://doi.org/10.5281/zenodo.7576601)


## 📄 License

This project is distributed under the **MIT License**.  
You are free to use, modify, and share this code with attribution.

View the full license in the [LICENSE](./LICENSE) file.


## 📫 Contact

**Tasha Snow, PhD**  
Earth System Science Interdisciplinary Center (ESSIC), University of Maryland \
NASA Goddard Space Flight Center  \
📧 [tsnow03@umd.edu](mailto:tsnow03@umd.edu)
