<p float="center">
  <img src="Figures/samples_dataset.png" width="1000" />
</p>

# Exposing the Unseen: Exposure Time Emulation for Offline Benchmarking of Vision Algorithms

[![DOI](https://zenodo.org/badge/DOI/10.48550/arxiv.2309.110718.svg)](https://doi.org/10.48550/arXiv.2309.13139)

This repository contains the code used in our paper *Exposing the Unseen: Exposure Time Emulation for Offline Benchmarking of Vision Algorithms* submitted at ICRA2024. 

[![Exposing the Unseen](https://img.youtube.com/vi/btkO12L6AYs/0.jpg)](https://youtu.be/btkO12L6AYs)


## Menu

  - [**Emulator**](#emulator)

  - [**BorealHDR Dataset**](#borealhdr-dataset)

  - [**Citing**](#citing)


## Emulator

**The BorealHDR dataset and the code are in preparation. They will be added soon!**

## BorealHDR Dataset

<img align="right" src="Figures/bracketing.gif" width="500" height="" />

**The BorealHDR dataset and the code are in preparation. They will be added soon!** 

The BorealHDR Dataset was acquired at the Montmorency Forest in Québec City, Canada.
In winter, this environment creates several HDR scenes coming from snow and trees.
It was developed mainly to be used with our emulation technique.

The images were collected using the bracketing technique with six exposure times: **1, 2, 4, 8, 16, and 32 ms**.
A ground truth is provided using the 3D lidar data and the *Iterative Closest Point (ICP)* algorithm 

BorealHDR contains:

    - 50 trajectories
    - 8.4 km
    - 4 hours of data
    - 333 813 images
    - Ground truth 3D maps
    - 3D lidar point clouds
    - IMU measurements
    - GPS data


## Citing

If you use our emulator or our BorealHDR dataset in your work, please cite [our preprint](https://doi.org/10.48550/arXiv.2309.13139):

```bibtex
@misc{gamache2023exposing,
      title={{Exposing the Unseen: Exposure Time Emulation for Offline Benchmarking of Vision Algorithms}}, 
      author={Olivier Gamache and Jean-Michel Fortin and Matěj Boxan and François Pomerleau and Philippe Giguère},
      year={2023},
      eprint={2309.13139},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
