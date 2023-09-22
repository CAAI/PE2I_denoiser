# PE2I_denoiser

The tool denoises Vision and Quadra PET data acquired with PE2I tracer at low-doses.

The model was trained with ~100 PE2I patients imaged on a Siemens Vision mCT system for 10 minutes with >180 MBq injected dose. Low-dose was simulated at a 5% level.

The model is transfer learned from the model proposed by Raphael S. Daveau, *et al.* *Deep learning based low-activity PET reconstruction of [18F]FE-PE2I and [11C]PiB in Neurodegenerative Disorders*, NeuroImage (2022).

## Usage

### Use with original data
You can use the model on your own data without any preprocessing:
`PE2I_denoiser --pet <PATH_TO_PET.nii.gz> --ct <PATH_TO_CT.nii.gz> --out <OUTPUT_DENOISED_PET.nii.gz>`

You can also supply a path to a directory with DICOM files as input (both PET and CT). OBS: Output continues to be in NII.GZ for now.

If you wish to scale the input data (e.g. for decay correction) use the `--scale` flag that scaled the low-dose input.

You can save the intermediate files by giving the flags `-si -io <OUTPUT_DIR>`

The expected runtime for the model including pre/post-processing is ~10 minutes.

### Use with preprocessed data
If you do not supply the `--ct` flag, no pre/post-processing will be performed, i.e. the model assumes that your data is:
 - skullstripped, 
 - spatially normalized to MNI space (256x256x256 matrix / 1mm isotropic voxel size), 
 - with images cropped to 176x176x200, 
 - decay correction and 95% percentile normalization applied

If so, only inference will be performed, which takes ~20 seconds.