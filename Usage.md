## User Manual

### Software Information

- **Version**: 2.0
- **Author**: lh9171338
- **Date**: 2022-02-20

### Default Hotkeys

- **Ctrl + O**: Open image folder
- **Ctrl + S**: Save annotation results
- **Ctrl + V**: Go to the next image
- **Ctrl + B**: Go back to the previous image
- **Ctrl + C**: Enter line segment annotation mode
- **Ctrl + D**: Delete the selected line segment annotation
- **Ctrl + U**: View the user manual

**Note**: You can edit the `default.yaml` file to modify the hotkeys.

### Mouse Controls

- **Left Click**: Annotate a line segment endpoint
- **Right Click**: Enter or exit line segment annotation mode

### Workflow

1. Select the folder containing the images to be annotated.
2. Right-click (or press `Ctrl + C`) to enter line segment annotation mode.
3. Use the left mouse button to annotate the endpoints of a line segment. After marking two endpoints, the line segment annotation is complete. You can exit the annotation mode at any time by right-clicking (or pressing `Ctrl + C`).
4. Repeat steps 2-3. It is recommended to save annotations frequently to avoid data loss in case of program crashes.

### Dataset Directory Structure

```
|-- dataset
    |-- <image folder>
        |-- 000001.png
        |-- 000002.png
        |-- ...
    |-- <label folder>
        |-- 000001.mat
        |-- 000002.mat
        |-- ...
    |-- <coeff folder>
        |-- 000001.yaml
        |-- 000002.yaml
        |-- ...
```

**Note**: The paths for `<image folder>`, `<label folder>`, and `<coeff folder>` are configured in the `default.yaml` file. These paths can overlap. The `<coeff folder>` is optional.

### Running the Program

```shell
python Labelline.py --type <image type> [--coeff_file <coeff image>]
```

**Options**:

- **type**: Specifies the type of image:
  - `0`: Planar image
  - `1`: Fisheye image
  - `2`: Spherical image
- **--coeff_file**: Specifies the camera parameters for fisheye images. If not explicitly provided using the `--coeff_file` option, the program will attempt to fetch the corresponding camera parameters from the `coeff_folder` path.

**Note**: Camera parameters are not required for planar and spherical images.
