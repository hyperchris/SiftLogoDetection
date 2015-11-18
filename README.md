# SiftLogoDetection
Detect a logo in an image with SIFT features

## Version Update
- Version: 1.0
- Update: all files uploaded

## How It Works
1. Scale the logo into differnet sizes
2. For each size, extract SIFT feature and locate it in the test image with RANSAC
3. Exclude unmatched localization results (too small, not rectangle, etc.).
4. Store the result image in 'res/'

## How to Use
- Compile the cpp file with 'cmake .' and 'make'
- Run: "$ ./sift [logo_path] [test_image_path_1] .. [test_image_path_n]"