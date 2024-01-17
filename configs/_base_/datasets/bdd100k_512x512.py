bdd_type = "CityscapesDataset"
bdd_root = "data/bdd100k/"
bdd_crop_size = (512, 512)
bdd_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1280, 720)),
    dict(type="RandomCrop", crop_size=bdd_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
bdd_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_bdd = dict(
    type=bdd_type,
    data_root=bdd_root,
    data_prefix=dict(
        img_path="images/10k/train",
        seg_map_path="labels/sem_seg/masks/train",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=bdd_train_pipeline,
)
val_bdd = dict(
    type=bdd_type,
    data_root=bdd_root,
    data_prefix=dict(
        img_path="images/10k/val",
        seg_map_path="labels/sem_seg/masks/val",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=bdd_test_pipeline,
)
