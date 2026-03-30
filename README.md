# Geese Camera Trap Object Detection

---

This repo contains some YOLO models and weights for the camera trap data only.

## Training

To reproduce the training performance, first create the data folder. The original data is consisted of two folder `./First training` and `./Second training`. Put those two folders under this project than run

```cmd
python merge_1&2_data.py
```

This will generate the `./cameratrap_data` folder along with ./cameratrap_data/images` ./cameratrap_data/labels` and `./cameratrap_data/dataset.yaml`

The content in `./cameratrap_data/dataset.yaml` looks like:

```yaml
path: ./cameratrap_data
train: images/train
val:   images/val
test:  images/test

nc: 4
names:
  0: Brent_Up
  1: Brent_Down
  2: Barnacle_Up
  3: Barnacle_Down
```

Then you can delete `./First training` and `./Second training` and start training/testing.

## References

- Jocher, G., Qiu, J., & Chaurasia, A. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

