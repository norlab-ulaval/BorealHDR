<p float="center">
  <img src="figures_readme/samples_dataset.jpg" width="1000" />
</p>

# Exposing the Unseen: Exposure Time Emulation for Offline Benchmarking of Vision Algorithms

[![DOI](https://zenodo.org/badge/DOI/10.48550/arxiv.2309.110718.svg)](https://doi.org/10.48550/arXiv.2309.13139)

This repository contains the code used in our paper *Exposing the Unseen: Exposure Time Emulation for Offline Benchmarking of Vision Algorithms* accepted at IROS2024. 

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

#### Download the datset

You can download each trajectory independently in [BorealHDR Dataset](#borealhdr-dataset)'s section. Or, we also added a small part of a trajecotry direclty in this repository to allow quick testing of our setup.

#### Docker

If you have downloaded the [BorealHDR Dataset](#borealhdr-dataset), the first step is to modify the last line of `.devcontainer/docker-compose.yaml` to mount the location of your data into the container at `/home/user/code/dataset_mount_point/`.

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

### Download

We provide a compressed version of the dataset in the following tables. In this version, the images still have a depth of 12-bits, but they have been compressed to reduce the overall size of the dataset.

Once you downloaded a trajectory, you will have to decompress it before using it with the emulator. You may have to copy-paste the link if directly clicking on 'Download' did not work.

-------------------------------------------------------------

<div align="center">

| Mont-Bélair  | Size (GB) | Download Link |
|:----:|:----:|:----:|
| backpack_2023-09-27-12-46-32 | 6.7        | [Download](http://norlab2.s3.valeria.science/BorealHDR/belair/backpack_2023-09-27-12-46-32.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286308&Signature=PHxQOfj865jByh2SKLsdAOPw6oQ%3D) |
| backpack_2023-09-27-12-51-03 | 4.6        | [Download](http://norlab2.s3.valeria.science/BorealHDR/belair/backpack_2023-09-27-12-51-03.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286308&Signature=T6FtMd4XPV4CWaJzfcQngsJ9nos%3D) |
| backpack_2023-09-27-13-20-03 | 6.3        | [Download](http://norlab2.s3.valeria.science/BorealHDR/belair/backpack_2023-09-27-13-20-03.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286308&Signature=7wWiPOUwKHVTxFktZdtGOOFp5p4%3D) |
| backpack_2023-09-27-13-25-44 | 6.1        | [Download](http://norlab2.s3.valeria.science/BorealHDR/belair/backpack_2023-09-27-13-25-44.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286309&Signature=du3nJUl%2BEBEV6DmTBMkv5MOmaJA%3D) |
| backpack_2023-09-27-13-29-22 | 8.4        | [Download](http://norlab2.s3.valeria.science/BorealHDR/belair/backpack_2023-09-27-13-29-22.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286309&Signature=nql%2B78xw5jf5GX%2FuRbQ%2BUnEcLtM%3D) |
| backpack_2023-09-27-13-34-17 | 7.9        | [Download](http://norlab2.s3.valeria.science/BorealHDR/belair/backpack_2023-09-27-13-34-17.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286309&Signature=591hIctyQBvdvWBhytBjgrPDrHs%3D) |

</div>

-------------------------------------------------------------

<div align="center">

| Campus  | Size (GB) | Download Link |
|:----:|:----:|:----:|
| backpack_2023-09-25-15-00-05 | 6.5        | [Download](http://norlab2.s3.valeria.science/BorealHDR/campus/backpack_2023-09-25-15-00-05.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286542&Signature=bONBUGsCF4UotiCpD7QGd9swCEE%3D) |
| backpack_2023-09-25-15-05-03 | 7.6        | [Download](http://norlab2.s3.valeria.science/BorealHDR/campus/backpack_2023-09-25-15-05-03.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286542&Signature=XYHdhynolIt93SwtxzYG%2BLwPt6c%3D) |
| backpack_2023-09-25-15-22-43 | 14.5        | [Download](http://norlab2.s3.valeria.science/BorealHDR/campus/backpack_2023-09-25-15-22-43.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286543&Signature=bqVqrGzJxNc4iYXBErCMOxlCZXM%3D) |

</div>

-------------------------------------------------------------

<div align="center">

| Forest-20  | Size (GB) | Download Link |
|:----:|:----:|:----:|
| backpack_2023-04-20-09-29-14 | 4.6        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-09-29-14.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286652&Signature=HgFAGetlz9W3m61pTkNl0Z4u4K8%3D) |
| backpack_2023-04-20-09-51-13 | 18.3        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-09-51-13.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286652&Signature=qMSwGknx%2FgSSkG6rqoW2ycjkhcA%3D) |
| backpack_2023-04-20-10-04-23 | 11.2        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-10-04-23.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286652&Signature=20gvAuXWUCAIjILxV6seZcxu9GI%3D) |
| backpack_2023-04-20-10-12-17 | 6.8        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-10-12-17.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286652&Signature=BhiXFgUFAWZyqcAZNGzL8HVlazI%3D) |
| backpack_2023-04-20-10-17-24 | 10.5        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-10-17-24.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286653&Signature=sM6z0%2F4RZQVSp1naJizQToGNYXM%3D) |
| backpack_2023-04-20-10-26-28 | 6.8        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-10-26-28.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286653&Signature=5aYeFWRUTDVj3kWWBAFf%2Bsilmyo%3D) |
| backpack_2023-04-20-10-51-41 | 10.1        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-10-51-41.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286653&Signature=Efe%2Bol2%2BdJkKX%2BKRbF1gvudcfm8%3D) |
| backpack_2023-04-20-10-59-06 | 9.1        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-10-59-06.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286653&Signature=WGd6K%2FycAgegpPiSw5XLeZ%2FEr14%3D) |
| backpack_2023-04-20-11-05-33 | 3.7        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-11-05-33.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286653&Signature=huhslAyiB59xhaVZxkPKNL9MI4s%3D) |
| backpack_2023-04-20-11-08-58 | 13.8        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-11-08-58.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286653&Signature=J3e%2Fx28nvmzbFYyaXjw9PVSqjCo%3D) |
| backpack_2023-04-20-11-17-07 | 8.3        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-11-17-07.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286654&Signature=7ihJu5%2FQ8KNGQEHAukJ7LvU3bpQ%3D) |
| backpack_2023-04-20-11-23-00 | 12.3        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-11-23-00.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286654&Signature=%2BU166Q%2F2r1M4xLVp8EZFD%2FcvPtU%3D) |
| backpack_2023-04-20-11-33-10 | 8.8        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-11-33-10.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286654&Signature=huXzA3Y5EGatyvmu6sScEzxd7wE%3D) |
| backpack_2023-04-20-11-43-53 | 12.5        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-11-43-53.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286654&Signature=SkpCTNqntwQ5Ts6kJLEhooApvwo%3D) |
| backpack_2023-04-20-14-02-06 | 7.8        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-14-02-06.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286654&Signature=MyUlzB3oYL5rADdNqwdjKqaGbCI%3D) |
| backpack_2023-04-20-14-07-00 | 3.0        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-14-07-00.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286654&Signature=g3yDdiGOLRGGmS4VDX1jQQa5KZI%3D) |
| backpack_2023-04-20-14-09-25 | 2.9        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-14-09-25.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286655&Signature=zgUIfBxuRg8SnBmNmjLbEcKp5eE%3D) |
| backpack_2023-04-20-14-14-15 | 8.4        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-14-14-15.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286655&Signature=vUSbjU5mvJwMdnNP2RGDfOmAtZo%3D) |
| backpack_2023-04-20-14-21-21 | 9.8        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-14-21-21.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286655&Signature=Za5GhhYTzx1ivp3jn0JYUISSeCk%3D) |
| backpack_2023-04-20-14-32-48 | 7.4        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-14-32-48.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286655&Signature=8i5dy0aXFWIZTAQ4PQIa32oL%2FS4%3D) |
| backpack_2023-04-20-14-39-36 | 0.7        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-14-39-36.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286655&Signature=ski3e33I%2B5JkwwMmJbChTieyQe0%3D) |
| backpack_2023-04-20-14-55-15 | 5.6        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-14-55-15.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286655&Signature=6LhI6fB6j%2FkeNPy7gwkmUtE04JQ%3D) |
| backpack_2023-04-20-14-59-46 | 6.4        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-14-59-46.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286656&Signature=B2BQpv1ENjU768mmKA4BEk1zRX8%3D) |
| backpack_2023-04-20-15-03-58 | 2.8        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-15-03-58.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286656&Signature=JgXhMgfgcR5TtTlXv6TgdhVotXA%3D) |
| backpack_2023-04-20-15-05-25 | 5.6        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-15-05-25.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286656&Signature=xr63X3m7Q8JDnsHMNZQKx53UlQU%3D) |
| backpack_2023-04-20-15-13-55 | 8.2        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-15-13-55.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286656&Signature=Bz8Q7V0EqkssuC%2BYCNGEhM2EnZ4%3D) |
| backpack_2023-04-20-15-38-49 | 19.1        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-20/backpack_2023-04-20-15-38-49.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286656&Signature=is1bCSaxlWOKq5d%2FqVWeim2rIpw%3D) |

</div>

-------------------------------------------------------------

<div align="center">

| Forest-21  | Size (GB) | Download Link |
|:----:|:----:|:----:|
| backpack_2023-04-21-08-51-27 | 6.0        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-08-51-27.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286828&Signature=4U4ysW6mgClRLBKXQzZOABWHKPE%3D) |
| backpack_2023-04-21-09-15-59 | 9.3        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-09-15-59.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286829&Signature=kqvYzRNs9cJ%2BzT5nxGYa2163Ra4%3D) |
| backpack_2023-04-21-09-22-05 | 9.2        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-09-22-05.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286829&Signature=wQpCMBakfk2X7d2ekymWoYioWb8%3D) |
| backpack_2023-04-21-09-31-09 | 16.1        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-09-31-09.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286829&Signature=E7cWt6touAnVScNfIXgoM3zhZzs%3D) |
| backpack_2023-04-21-09-41-22 | 8.8        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-09-41-22.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286829&Signature=szW6U%2Fsd%2By%2Bc%2BWDH1cjDv2syLjE%3D) |
| backpack_2023-04-21-09-49-58 | 0.3        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-09-49-58.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286829&Signature=8FNTNI9bnohmfEI%2FwMtnBp3Yl0U%3D) |
| backpack_2023-04-21-09-52-00 | 5.1        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-09-52-00.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286830&Signature=FKkQVsRZ%2BQPNz%2FMGdj2VzL4xHCc%3D) |
| backpack_2023-04-21-09-54-38 | 4.1        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-09-54-38.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286830&Signature=i9%2BxEi2pH1E37rcgC6ukdv1vTHA%3D) |
| backpack_2023-04-21-10-23-23 | 14.6        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-10-23-23.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286830&Signature=ex%2FGzEphuJbkuX1AZAdFCF4KQMI%3D) |
| backpack_2023-04-21-10-32-34 | 13.6        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-10-32-34.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286830&Signature=0poK4akNE92uQptyMOMzIEHMcE8%3D) |
| backpack_2023-04-21-10-46-54 | 6.8        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-10-46-54.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286830&Signature=5OeKxSe2wN9MC8R6QSYjBHWza3E%3D) |
| backpack_2023-04-21-10-57-59 | 7.9        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-10-57-59.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286831&Signature=J6etZrsu3xJTcsN5At05iFl1T%2BA%3D) |
| backpack_2023-04-21-11-03-44 | 5.7        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-11-03-44.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286831&Signature=6zccnBxU%2FA1hAbRBYkOCle5aflI%3D) |
| backpack_2023-04-21-11-08-10 | 3.0        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-11-08-10.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286831&Signature=hy4%2FeMtfX4VnqMDymZ51VBP%2BCYA%3D) |
| backpack_2023-04-21-11-27-52 | 3.0        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-11-27-52.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286831&Signature=%2Fq8Ok4gnWebpT%2BZe53U625PDUW4%3D) |
| backpack_2023-04-21-11-30-58 | 9.5        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-11-30-58.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286831&Signature=p3jwU4FHbBiQ6aJfpSXVk3vk%2BZo%3D) |
| backpack_2023-04-21-11-38-04 | 3.9        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-11-38-04.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286831&Signature=REzniR%2BUrca%2FDsE8KvSXtxhV1Gw%3D) |
| backpack_2023-04-21-11-43-19 | 3.6        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-11-43-19.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286832&Signature=ycve5YnskjB%2BTHm2p35LmVdi610%3D) |
| backpack_2023-04-21-11-47-04 | 12.7        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-11-47-04.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286832&Signature=8NMuz2SUNymd1yTr2XGKkghK76Q%3D) |
| backpack_2023-04-21-12-07-33 | 6.9        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-12-07-33.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286832&Signature=ARtgH0YLJglVkzEFlA1tjMKvu%2F4%3D) |
| backpack_2023-04-21-13-44-26 | 25.2        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-13-44-26.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286832&Signature=Y9k%2BIUh4tIsaV%2FpBSunzZa9yIbk%3D) |
| backpack_2023-04-21-14-01-39 | 6.4        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-14-01-39.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286832&Signature=jRn8N%2BcNAWr9BN%2F65YPyHkUonDo%3D) |
| backpack_2023-04-21-14-10-00 | 6.2        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-14-10-00.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286833&Signature=8WbgUqu9yCBed5PF5YZv1ZwGg0g%3D) |
| backpack_2023-04-21-14-20-06 | 0.4        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-14-20-06.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286833&Signature=LF4o7BYC9%2FuhsdeSoQr4hH7ETCs%3D) |
| backpack_2023-04-21-14-59-17 | 9.1        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-14-59-17.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286833&Signature=vUvuKgYGcTiHkY5ij8tTvaTbKrs%3D) |
| backpack_2023-04-21-15-07-12 | 4.6        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-15-07-12.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286833&Signature=TwC0LmsuxYCZRvJ%2FxD8qXyVe6XQ%3D) |
| backpack_2023-04-21-15-10-54 | 4.9        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-15-10-54.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286833&Signature=eE0heHarveQAyX9thHyy4R97fCA%3D) |
| backpack_2023-04-21-15-16-29 | 7.9        | [Download](http://norlab2.s3.valeria.science/BorealHDR/forest-21/backpack_2023-04-21-15-16-29.zip?AWSAccessKeyId=0H3T890M5GYEV6TJW6FP&Expires=2333286833&Signature=C1Tbc2Ho7vrEXi%2Bly10hwCzMrB4%3D) |

</div>

-------------------------------------------------------------

## Citing

If you use our emulator or our BorealHDR dataset in your work, please cite [our preprint](https://doi.org/10.48550/arXiv.2309.13139):

```bibtex
@misc{gamache2023exposing,
      title={{Exposing the Unseen: Exposure Time Emulation for Offline Benchmarking of Vision Algorithms}}, 
      author={Olivier Gamache and Jean-Michel Fortin and Matěj Boxan and Maxime Vaidis and François Pomerleau and Philippe Giguère},
      year={2024},
      eprint={2309.13139},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
