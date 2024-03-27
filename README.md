<p float="center">
  <img src="figures_readme/samples_dataset.jpg" width="1000" />
</p>

# Exposing the Unseen: Exposure Time Emulation for Offline Benchmarking of Vision Algorithms

[![DOI](https://zenodo.org/badge/DOI/10.48550/arxiv.2309.110718.svg)](https://doi.org/10.48550/arXiv.2309.13139)

This repository contains the code used in our paper *Exposing the Unseen: Exposure Time Emulation for Offline Benchmarking of Vision Algorithms* submitted at IROS2024. 

[![Exposing the Unseen](figures_readme/thumbnail_video.jpg)](https://youtu.be/JN9faAvCRkU)


## Menu

  - [**Emulator**](#emulator)

  - [**BorealHDR Dataset**](#borealhdr-dataset)

  - [**Citing**](#citing)


## Emulator

We created a Dockerfile to easily run our code using a docker-compose.yaml.

### Dependencies

If you want to use the docker container, you have to install Docker using this website: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

### Running the emulator

First, start by cloning this repository on your computer. 
```bash
git clone git@github.com:norlab-ulaval/BorealHDR.git
```

#### Docker

If you have downloaded the dataset in another directory, the first step is to modify the last line of `.devcontainer/docker-compose.yaml` to mount the location of your data into the container at `/home/user/code/dataset_mount_point/`. If you did not download the dataset, we added a small part of a trajectory direclty into this repository to enable testing our pipeline.

Then, you can open the devcontainer in `vscode`, or build the image with `docker compose up --build`.
```bash
docker compose up --build
```

When your inside the docker container, you can direclty emulate images from the dataset by running

```bash
cd /home/user/code/scripts/
python3 emulator_threads.py
```

####  Python virtual environment

Instead of using the docker container, you can also direclty run the code locally using a virtual environment. First you have to create the python virtual environment and install the dependencies from the `requirements.txt` file:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, you can adapt the `parameters.yaml` file to define the position of the dataset. The default path to the dataset is to `../data_sample/`. To emulate images, you need to run the script `emulator_threads.py`

```bash
cd scripts/
python emulator_threads.py
```

### Parameters

The `emulator_threads.py` loads `parameters.yaml` file. You can adapt some parameters to choose which automatic-exposure technique to use and also some debugs parameters. Note that `emulator_threads.py` uses multiple threads to accelerate the processes. You can then emulate multiple automatic-exposure algorithms by un-commenting methods in `automatic_exposure_techniques` from `parameters.yaml`. The following table describes the available parameters related to the emulation. 

| Parameter                        |                    Description                  | Values (default first) |
| :---                             |                      :---                         |                   ---: |
| `exposure_time_init`             | Exposure time of the first emulated image       | `4.0`                       |
| `dataset_folder`                 | Parent path to the dataset       | `"../data_sample/"`                      |
| `location_acquisition`           | To select the good folder, choose the location from where the sequence you want to emulate was acquired       | `"ulaval_campus"`<br />`"belair"`<br />`"forest_20"`<br />`"forest_21"`                       |
| `experiment`                     | Sequence name       | `"backpack_2023-09-25-15-05-03"`                       |
| `depth_emulated_imgs`            | Emulate images in 8bits or 12bits       | `8`<br />`12`                       |
| `emulated_in_color`              | Boolean: Emulate images in color (only for 8bits)       | `True`<br />`False`                       |
| `automatic_exposure_techniques`  | Select one or more automatic-exposure methods to emulate       | `"classical-50"`<br />`"classical-30"`<br />`"classical-70"`<br />`"manual-0"`<br />`"gradient-0"`<br />`"ewg-0"`<br />`"softperc-0"`                       |
| `save_or_show_emulated_imgs`     | Show or save the emulated images       | `"save"`<br />`"show"`                       |
| `save_path`                      | Path to which the results of the emulation will be saved       | `"/home/user/code/output/emulated_images/"`                       |

## BorealHDR Dataset

<img align="right" src="figures_readme/dataset_acquisition.gif" width="500" height="" />

**Due to its size, the entire BorealHDR dataset is in preparation. It will be added soon!**

The BorealHDR Dataset was acquired at the Montmorency Forest in Québec City, Canada.
In winter, this environment creates several HDR scenes coming from snow and trees.
It was developed mainly to be used with our emulation technique.

The images were collected using the bracketing technique with six exposure times: **1, 2, 4, 8, 16, and 32 ms**.
A ground truth is provided using the 3D lidar data and a lidar-inertial-SLAM algorithm based on [Libpointmatcher](https://github.com/norlab-ulaval/libpointmatcher).

BorealHDR contains:

    - 55 trajectories
    - 10 km
    - 5 hours of data
    - 393 238 images
    - Ground truth 3D maps
    - 3D lidar point clouds
    - IMU measurements
    - GPS data


## Citing

If you use our emulator or our BorealHDR dataset in your work, please cite [our preprint](https://doi.org/10.48550/arXiv.2309.13139):

```bibtex
@misc{gamache2023exposing,
      title={{Exposing the Unseen: Exposure Time Emulation for Offline Benchmarking of Vision Algorithms}}, 
      author={Olivier Gamache and Jean-Michel Fortin and Matěj Boxan and Maxime Vaidis and François Pomerleau and Philippe Giguère},
      year={2023},
      eprint={2309.13139},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
