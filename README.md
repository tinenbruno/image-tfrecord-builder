# Image TFRecord Builder
A python tfrecord builder tool focused on images

## Usage
### Dataset Setup

This tool expects your image dataset to be structured in the following way:

```
> base_dir
  > classname 1
    > image 1
    > image 2
    > ...
  > classname 2
    > image 1
    > image 2
    > ...
```

### Settings

You will also need to adjust the `settings.py`.

| Setting | Description | Example |
| ------- | ----------- | ------- |
| IMAGES_INPUT_FOLDER | Absolute path for your image dataset folder | /home/user/dataset |
| OUTPUT_FILENAME | Absolute base filename for output (can be used with a remote setting as well) | /home/user/output |
| NUMBER_OF_SHARDS | Number of splits for both training and test tfrecord files | 2 |
| TRAINING_EXAMPLES_SPLIT | Percentage of examples that will be used as training | 0.8 |
| SEED | Seed to allow example shuffling with repeatability | 123 |

### Requirements

Using pip: `pip install -r requirements.txt`

### Running

To run simply use `python image-tfrecord-builder.py`
