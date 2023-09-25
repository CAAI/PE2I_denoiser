from urllib.request import urlopen
import os
from PE2I_denoiser.paths import folder_with_parameter_files
from pathlib import Path
import subprocess as subp
from nipype.interfaces.fsl import ConvertXFM
import nibabel as nib

# Actual models in use
models = {
    'Vision_TLmCT_1-5pct': 'Vision_TLmCT_1-5pct_v3_220923'
}

def get_params_fname(model, is_key=True):
    if not is_key:
        return os.path.join(folder_with_parameter_files, f"{model}.onnx")
    else:
        return os.path.join(folder_with_parameter_files, f"{models[model]}.onnx")

def get_template_fname():
    return os.path.join(folder_with_parameter_files, 'avg_template.nii.gz')

def maybe_download_parameters_and_template(force_overwrite=False):
    """
    Downloads the parameters if it is not present yet.
    :param force_overwrite: if True the old parameter file will be deleted (if present) prior to download
    :return:
    """

    maybe_mkdir_p(folder_with_parameter_files)

    # Remove old files when version is updated
    for f in os.listdir(folder_with_parameter_files):
        if f.endswith('.onnx') and f not in models.values():
            os.remove(get_params_fname(f, is_key=False))

    for model in models.values():
        out_filename = get_params_fname(model, is_key=False)
        if force_overwrite and os.path.isfile(out_filename):
            os.remove(out_filename)

        if not os.path.isfile(out_filename):
            url = f"https://zenodo.org/record/8376789/files/{model}.onnx?download=1"
            print("Downloading", url, "...")
            data = urlopen(url).read()
            with open(out_filename, 'wb') as f:
                f.write(data)

    out_templatename = get_template_fname()
    if not os.path.isfile(out_templatename):
        url = f"https://zenodo.org/record/8376789/files/avg_template.nii.gz?download=1"
        print("Downloading", url, "...")
        data = urlopen(url).read()
        with open(out_templatename, 'wb') as f:
            f.write(data)

def maybe_mkdir_p(directory):
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            os.mkdir(os.path.join("/", *splits[:i+1]))

def convert_to_nii(out_dir, out_name, dcm_folder):
        cmd = ['dcm2niix',
            '-o', str(out_dir),
            '-f', out_name,
            '-z', 'y',
            str(dcm_folder)]
        out = subp.check_output(cmd)
        given_name = str(out).split('DICOM as ')[1].split(' ')[0]
        # Fix name suffix
        if not (out_dir / (out_name+'.nii.gz')).exists():
            Path(f'{given_name}.nii.gz').rename(out_dir / (out_name+'.nii.gz'))
            Path(f'{given_name}.json').rename(out_dir / (out_name+'.json'))

def register(file_, reference_file, out_file, dof=6, overwrite=False):
        mat_file = str(out_file).replace(".nii.gz", ".mat")
        if not os.path.exists(out_file) or overwrite:
            cmd = [
                "flirt",
                "-in",
                file_,
                "-ref",
                reference_file,
                "-out",
                out_file,
                "-dof",
                str(dof),
                "-omat",
                mat_file,
            ]
            _ = subp.check_output(cmd)

def skullstrip_HDCTBET(CT):
        if not os.path.exists(out_BET := str(CT).replace('.nii.gz', '_BET.nii.gz')):
            cmd = ['hd-ctbet', '-i', CT, '-o', out_BET]
            subp.check_output(cmd, text=True,env=os.environ.copy())
        BET_mask = str(out_BET).replace('_BET.nii.gz', '_BET_mask.nii.gz')

def resample(file_, reference_file, mat_file, out_file, interp=None):
        cmd = [
                "flirt",
                "-in",
                file_,
                "-out",
                out_file,
                "-ref",
                reference_file,
                "-applyxfm",
                "-init",
                mat_file
            ]
        if interp is not None:
            cmd += ['-interp', interp]
        output = subp.check_output(cmd)
        return output

def concat_XFMs(xfm1, xfm2, xfm_out):
    concat = ConvertXFM()
    concat.inputs.concat_xfm = True
    concat.inputs.in_file = xfm1
    concat.inputs.in_file2 = xfm2
    concat.inputs.out_file = xfm_out
    concat.run()

def apply_mask(file_, mask, out_file):
    img = nib.load(file_)
    BET = nib.load(mask)
    arr = img.get_fdata() * BET.get_fdata()
    img = nib.Nifti1Image(arr, img.affine, img.header)
    img.to_filename(out_file) 