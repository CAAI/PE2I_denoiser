import os
import numpy as np
import onnxruntime
import torch
from torch.utils.data import DataLoader
from torchio.data import GridSampler, GridAggregator
import torchio as tio
import shutil
import warnings
import tempfile
import time
from pathlib import Path
import subprocess as subp
from multiprocessing import Process
from PE2I_denoiser.utils import (
    convert_to_nii,
    register,
    resample,
    concat_XFMs,
    apply_mask,
    skullstrip_HDCTBET,
    get_params_fname,
    get_template_fname
)

#suppress warnings
warnings.filterwarnings('ignore')

class Denoiser():
    def __init__(self, pet, ct, verbose=False):
        self.verbose = verbose
        self.tmp_dir_object = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_object.name)
        self.pet = pet
        self.ct = ct
        self.template = get_template_fname()
        self.is_preprocessed = False

        if verbose:
             print("Created tmp folder", self.tmp_dir)


    def clean(self):
        self.tmp_dir_object.cleanup()


    def copy_intermediate_files(self, to_dir):
        files = [
            self.pet, # if converted from dicom..
        ]
        for f in files:
            shutil.copyfile(f, to_dir/f.name)

    def preprocess(self, scale=1.0):

        if self.ct is None:
             if self.verbose:
                  print("CT was not passed - assuming PET image is already preprocessed")
             self.preprocessed = tio.ScalarImage(self.pet) # Assumes the image is already preprocessed
             return

        self.is_preprocessed = True
        
        crop_config = {
            "x_lim": [43, 37],
            "y_lim": [33, 47],
            "z_lim": [16, 40]
        }
        self.crop = tio.Crop((*crop_config['x_lim'], *crop_config['y_lim'], *crop_config['z_lim']))
        self.pad = tio.Pad((43, 37, 33, 47, 16, 40))
        self.scale = tio.Lambda(lambda x: x * scale)

        # Convert to nii if DICOM input
        if self.pet.endswith('.nii.gz'):
            shutil.copyfile(self.pet, self.tmp_dir / 'PET.nii.gz')
            shutil.copyfile(self.ct, self.tmp_dir / 'ACCT.nii.gz')
            self.pet = self.tmp_dir / 'PET.nii.gz'
            self.ct = self.tmp_dir / 'ACCT.nii.gz'
        else:
            if self.verbose:
                print('Converting DICOM files to NII')
            convert_to_nii(self.tmp_dir, 'PET', self.pet)
            convert_to_nii(self.tmp_dir, 'ACCT', self.ct)
            self.pet_dicom = self.pet
            self.pet = self.tmp_dir / 'PET.nii.gz'

        # Skullstrip
        if self.verbose:
            print("Skullstripping CT")
        p_BET = Process(target=skullstrip_HDCTBET, args=(self.ct,))
        p_BET.start()

        # Register PET to CT
        if self.verbose:
            print("Registering PET to CT")
        PET_to_CT = self.tmp_dir / 'PET_to_CT.nii.gz'
        p_RegtoCT = Process(target=register, args=(self.pet,self.ct, PET_to_CT, 6, False))
        p_RegtoCT.start()

        # Wait for CT_BET to finish
        p_BET.join()
        CT_BET = str(self.ct).replace('.nii.gz', '_BET.nii.gz')
        BETmask = str(self.ct).replace('.nii.gz', '_BET_mask.nii.gz')

        # Align CT_BET to avg
        if self.verbose:
            print("Registering CT_BET to MNI")
        CT_to_avg = str(CT_BET).replace('.nii.gz', '_to_MNI.nii.gz')
        p_RegtoMNI = Process(target=register, args=(CT_BET,self.template, CT_to_avg, 12, False))
        p_RegtoMNI.start()

        # Wait for FLIRT processes to finish
        p_RegtoCT.join()
        mat_to_ct = str(PET_to_CT).replace(".nii.gz", ".mat")
        p_RegtoMNI.join()
        mat_to_avg = str(CT_to_avg).replace(".nii.gz", ".mat")

        # Concat to_ct and to_avg
        self.concatted_xfms = os.path.join(self.tmp_dir, 'reg_to_ct_to_avg.mat')
        concat_XFMs(mat_to_ct, mat_to_avg, self.concatted_xfms)

        # Resample PET to avg
        PET_to_avg = self.tmp_dir / 'PET_to_avg.nii.gz'
        resample(self.pet, self.template, self.concatted_xfms, PET_to_avg)

        # Resample BETmask to avg
        self.bet_mni = str(BETmask).replace('.nii.gz', '_to_avg.nii.gz')
        resample(BETmask, self.template, mat_to_avg, self.bet_mni, interp='nearestneighbour')

        # Apply BETmask to PET in avg space
        self.pet_mni = str(PET_to_avg).replace('.nii.gz', '_BET.nii.gz')
        apply_mask(PET_to_avg, self.bet_mni, self.pet_mni)

        # Load image
        self.PET_LD = tio.ScalarImage(self.pet)
        self.PET_LD_MNI = tio.ScalarImage(self.pet_mni)

        # Crop
        img = self.crop(self.PET_LD_MNI)

        # Scale
        img = self.scale(img)

        # Normalize
        self.percentile_value = np.percentile(img.numpy(), q=99.5)
        norm = tio.Lambda(lambda x: x / self.percentile_value)
        self.preprocessed = norm(img)
        self.de_norm = tio.Lambda(lambda x: x * self.percentile_value)


    def inference(self, model_name, ps=[176,16,200], po=4, bs=1, GPU=True):
         # Load model
        if self.verbose:
            print('Loading model')
        model_path = get_params_fname(model_name)
        model = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        
        input_name = model.get_inputs()[0].name

        subject = tio.Subject(img=self.preprocessed)
        grid_sampler = GridSampler(subject, ps, po)
        patch_loader = DataLoader(grid_sampler, batch_size=bs)
        aggregator = GridAggregator(grid_sampler, overlap_mode='average')
        if self.verbose:
            print('Starting inference')
            start_time = time.time()
        for patches_batch in patch_loader:
            patch_x = patches_batch['img'][tio.DATA].float().numpy()
            locations = patches_batch[tio.LOCATION]
            ort_outs = model.run(None, {input_name: patch_x})
            patch_y = torch.from_numpy(ort_outs[0])
            aggregator.add_batch(patch_y, locations)

        self.denoised = tio.ScalarImage(tensor=aggregator.get_output_tensor(), affine=self.preprocessed.affine)
        if self.verbose:
            print(f'Inference done in {time.time()-start_time:.01f} seconds')

        
    def postprocess(self, add_blurring=False):

        if not self.is_preprocessed:
            return # Cant postprocess if the data was not preprocessed

        # Revert normalization
        if self.verbose:
            print("De-normalization")
        denoised = self.de_norm(self.denoised)

        # Revert crop by zero padding
        if self.verbose:
            print("De-cropping")
        denoised = self.pad(denoised)
        
        # Invert XFM
        if self.verbose:
            print("Inverting mat")
        inverted = self.tmp_dir / 'inverted.mat'
        cmd = ['convert_xfm', '-omat', inverted, '-inverse', self.concatted_xfms]
        subp.run(cmd)

        # Resample to patient space
        if self.verbose:
            print("Resampling PET")
        denoised_file = self.tmp_dir / 'denoised_mni_space.nii.gz'
        denoised.save(denoised_file)
        patient_space = self.tmp_dir / 'patient_space.nii.gz'
        cmd = ['flirt', '-in', denoised_file, '-ref', self.pet, '-init', inverted, '-applyxfm', '-out', patient_space]
        subp.run(cmd)

        # Also resample BET
        if self.verbose:
            print("Resampling BET")
        BET_patient_space = self.tmp_dir / 'BET_patient_space.nii.gz'
        cmd = ['flirt', '-in', self.bet_mni, '-ref', self.pet, '-init', inverted, '-applyxfm', '-out', BET_patient_space, '-interp', 'nearestneighbour']
        subp.run(cmd)

        # Blur LD PET
        LD = self.scale(self.PET_LD)
        if add_blurring:
            if self.verbose:
                print("Blurring lowdose") 
            LD_blurred = self.tmp_dir / 'LD_blurred.nii.gz'
            cmd = ['fslmaths', self.pet, '-s', str(3.9 / 2.3548), LD_blurred]
            subp.run(cmd)
            LD = tio.ScalarImage(LD_blurred)
            LD = self.scale(LD)

        # Merge lowdose and denoised PET
        if self.verbose:
            print("Merging images and saving")
        denoised = tio.ScalarImage(patient_space)
        arr = LD.numpy()
        BET = tio.LabelMap(BET_patient_space).numpy()
        arr[BET>0] = denoised.numpy()[BET>0]
        # Reapply LD mask of background
        LD_lower_mask = self.PET_LD.numpy().copy()<5 # Hardcoded to 5, based on 5% LD
        arr[LD_lower_mask] = 0
        # Set strictly possitive
        arr[arr<0] = 0

        self.postprocessed = tio.ScalarImage(tensor=arr, affine=LD.affine)


    def save_intermediate_files(self, out_dir, overwrite=False):
        if Path(out_dir).exists():
            # Out directory already exists - copy over individual files
            for f in self.tmp_dir.iterdir():
                if (to_f := Path(out_dir).joinpath(f.name)).exists() and overwrite:
                    to_f.unlink()
                if not to_f.exists():
                    shutil.copyfile(f, to_f)
        else:
            # Cope entire directory
            shutil.copytree(self.tmp_dir, out_dir)

    
    def save(self, out):
        if self.is_preprocessed:
            try:
                self.postprocessed.save(out)
            except:
                raise ValueError('Cannot save denoised image as it has not yet been created')
            
        else:
            try:
                self.denoised.save(out)
            except:
                raise ValueError('Cannot save denoised image from MNI space as it has not yet been created')
