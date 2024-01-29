fog_acdc_type = "CityscapesDataset"
fog_acdc_root = "data/acdc/"
fog_acdc_crop_size = (1024, 1024)
fog_acdc_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(1080 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size=fog_acdc_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
fog_acdc_val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
fog_acdc_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="PackSegInputs"),
]
train_fog_acdc = dict(
    type=fog_acdc_type,
    data_root=fog_acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/fog/train",
        seg_map_path="gt/fog/train",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=fog_acdc_train_pipeline,
)
val_fog_acdc = dict(
    type=fog_acdc_type,
    data_root=fog_acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/fog/val",
        seg_map_path="gt/fog/val",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=fog_acdc_val_pipeline,
)

test_fog_acdc = dict(
    type=fog_acdc_type,
    data_root=fog_acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/fog/test",
        seg_map_path="gt/fog/test",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=fog_acdc_test_pipeline,
)
