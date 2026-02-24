import fiftyone as fo
import kagglehub

# Download dice dataset
path = kagglehub.dataset_download("nellbyler/d6-dice")
print("Path to dataset files:", path)
images_path = path + "/d6-dice/Images"
ann_path = path + "/d6-dice/Annotations"
name = "Dice Detection"

# Create the FiftyOne dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=images_path,
    dataset_type=fo.types.ImageDirectory,
    name=name,
    overwrite=True,
)

print(dataset)
print(dataset.head())

# Loop through for each sample in our dataset
for sample in dataset:
    sample_root = sample.filepath.split("/")[-1].split(".")[0]
    sample_ann_path = ann_path + "/" + sample_root + ".txt"
    with open(sample_ann_path, 'r') as file:
        list_of_anns = [line.strip().split() for line in file]

    detections = []
    for ann in list_of_anns:
        label = str(int(ann[0]) + 1)
        bbox = [float(x) for x in ann[1:]]
        bbox_adjusted = [bbox[0]-bbox[3]/2, bbox[1]-bbox[2]/2, bbox[3], bbox[2]]
        det = fo.Detection(label=label, bounding_box=bbox_adjusted)
        detections.append(det)

    sample["ground_truth"] = fo.Detections(detections=detections)
    sample.save()

# Launch AFTER annotations are loaded
session = fo.launch_app(dataset)
session.wait()
