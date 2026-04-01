# Geese Camera Trap Object Detection (OD)

---

This repo contains some yolo OD models and weights for the camera trap data only.

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

Then you can delete `./First training` and `./Second training` and start training/testing with.

```python
python train_yolo26m_baseline.py
```



## Testing

For testing the yolo OD model, now only the yolo26m weight is saved in this repo.

To run plain yolo26m for testing, run

```python
python test_yolo26m_baseline.py eval
```

To run yolo26m for testing with top 25% area excluded, run

```python
python test_yolo26m_baseline_region.py
```

To run yolo26m for testing with smallest 20% samples excluded, run

```python
python test_yolo26m_size_filter.py
```

## References

- Jocher, G., Qiu, J., & Chaurasia, A. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

