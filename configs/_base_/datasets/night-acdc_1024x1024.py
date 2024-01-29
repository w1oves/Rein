night_acdc_type = "CityscapesDataset"
night_acdc_root = "data/acdc/"
night_acdc_crop_size = (1024, 1024)
night_acdc_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(1080 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size=night_acdc_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
night_acdc_val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
night_acdc_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="PackSegInputs"),
]
train_night_acdc = dict(
    type=night_acdc_type,
    data_root=night_acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/night/train",
        seg_map_path="gt/night/train",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=night_acdc_train_pipeline,
)
val_night_acdc = dict(
    type=night_acdc_type,
    data_root=night_acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/night/val",
        seg_map_path="gt/night/val",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=night_acdc_val_pipeline,
)
test_night_acdc = dict(
    type=night_acdc_type,
    data_root=night_acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/night/test",
        seg_map_path="gt/night/test",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=night_acdc_test_pipeline,
)
