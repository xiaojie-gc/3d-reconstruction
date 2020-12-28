## 3D Reconstruction with [openMVG](https://github.com/openMVG/openMVG) + [openMVS](https://github.com/cdcseacave/openMVS) + Background subtraction

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/xiaojie-gc/3d-reconstruction.git
   ```
2. Install packages

   ```sh
   pip3 install -r requirements.txt
   ```
3. Modify reconstructor (i.e., MvgMvsPipeline.py)

   ```sh 
   CAMERA_SENSOR_DB_DIRECTORY = "your path"
   ```
   
<!-- USAGE EXAMPLES -->
## Usage

1. Run 3D reconstruction using a single machine

    ```sh
    python3 main.py --data_dir "path to dataset" --fg_dir "path to foreground" --bg_dir "path to background" --output_dir "path to final output"
    ```
   The running time of each step will be stored in ***"time.json"***, for example:
   
   Use Google Compute Engine with 8 vCPUs and NVIDIA Tesla V100 (5120 CUDA cores).
   
   a. Those results are from 3D reconstruction using original images. The "Densify point cloud" takes 70 seconds.
   ```sh
   {
        "timeList": [
            {
                "model": "path to /full",
                "time": {
                    "Intrinsics analysis": 0.02153801918029785,
                    "Compute features": 2.1476070880889893,
                    "Compute matches": 0.5026075839996338,
                    "Incremental reconstruction": 0.7004132270812988,
                    "Export to openMVS": 1.9742863178253174,
                    "Densify point cloud": 70.24964547157288,
                    "Reconstruct the mesh": 11.7359459400177,
                    "Refine the mesh": 11.944321632385254,
                    "Texture the mesh": 10.571678638458252
                }
             }
        ]
    } 
   ```
  
   b. Those results are from 3D reconstruction using background subtraction (we will add merger time later). The total "Densify point cloud" takes
   12 + 47 = 59 seconds.
   ```sh
    {
        "timeList": [
            {
                "model": "path to /fg/00001",
                "time": {
                    "Intrinsics analysis": 0.012930154800415039,
                    "Compute features": 1.7421512603759766,
                    "Compute matches": 0.25075769424438477,
                    "Incremental reconstruction": 0.2518298625946045,
                    "Export to openMVS": 0.6381652355194092,
                    "Densify point cloud": 12.0526442527771,
                    "Reconstruct the mesh": 3.208322048187256
                }
            },
            {
                "model": "path to /bg/00001",
                "time": {
                    "Intrinsics analysis": 0.02222466468811035,
                    "Compute features": 1.956984043121338,
                    "Compute matches": 0.38178491592407227,
                    "Incremental reconstruction": 0.5192770957946777,
                    "Export to openMVS": 1.5061004161834717,
                    "Densify point cloud": 47.973461627960205,
                    "Reconstruct the mesh": 8.4803626537323,
                    "Refine the mesh": 8.971415042877197,
                    "Texture the mesh": 8.95643663406372
                }
            }
        ]
    }
    ```


<!-- problems -->
## Current Problems (unsolved)
1. The creation of foreground and background needs improvement.
2. The reconstruction of foreground may stops after "Reconstruct the mesh".
3. Finish merge.

## Authors

1. Andrew Hilton
2. Mingjun Li
3. Xiaojie Zhang
