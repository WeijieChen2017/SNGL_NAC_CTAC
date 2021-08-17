import os
import glob

os.system("set AFNI_NIFTI_TYPE_WARN = NO")
case_list = glob.glob("./CT_*.nii.gz")
case_list.sort()
for case_path in case_list:
    name = os.path.basename(case_path)[3:6]
    # idx_str = "{0:0>3}".format(idx+1)
    print(name)
    # cmd_1 = "3dresample -dxyz 1.172 1.172 1.065 -prefix "+tag+idx_str+" -inset "+tag+idx_str+"_PET.nii.gz -rmode Cu"
    cmd_1 = "3dresample -prefix NPR_"+name+".nii.gz -master CT_"+name+".nii.gz -rmode Cu -input NP_"+name+".nii.gz -bound_type SLAB"
    # cmd_2 = "3dZeropad -I 17 -S 17 p"+idx_str+"+orig"
    # cmd_3 = "3dAFNItoNIFTI -prefix "+tag+idx_str+" "+tag+idx_str+"+orig"
    # cmd_4 = "rm -f zeropad+orig.BRIK"
    # cmd_5 = "rm -f zeropad+orig.HEAD"
    # cmd_6 = "rm -f "+tag+idx_str+"+orig.BRIK"
    # cmd_7 = "rm -f "+tag+idx_str+"+orig.HEAD"
    # cmd_8 = "mv "+tag+idx_str+".nii ./PET_2x/"
    # cmd_6 = "mv y"+idx_str+".nii ../inv_RSZP"
    for cmd in [cmd_1]:
        print(cmd)
        os.system(cmd)
# 3dresample -dxyz 1.172 1.172 2.78 -prefix test -inset BraTS20_Training_001_t1_inv.nii
# 3dZeropad -I 16 -S 17 -A 25 -P 26 -L 25 -R 26 Z001+orig -prefix 123
# 3dAFNItoNIFTI -prefix test test+orig