Patches for model development, one folder per split, patch type and label (`1` = real face, `0` / `2` = attacks).
`patch_info` is `<scale>_<height>x<width>` (`org_1_80x60` = whole image resized, no crop).
Keep the same person in only one split, or validation and test scores will be optimistic.

```
datasets/rgb_image/
├── train/
│   ├── 2.7_80x80/
│   │   ├── 0/   aaa.png ...
│   │   ├── 1/   ddd.png ...
│   │   └── 2/   ggg.png ...
│   └── 4_80x80/ ...
├── val/      (same layout)   -> 02_validate.ipynb
└── test/     (same layout)   -> 03_test.ipynb
```
