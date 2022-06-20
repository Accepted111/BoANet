import time
import numpy as np
from torch import multiprocessing
import nibabel as nib

def make_affine(spacing):
    affine = np.array(((0, 0, -1, 0),
                       (0, -1, 0, 0),
                       (-1, 0, 0, 0),
                       (0, 0, 0, 1)))
    spacing = np.diag(list(spacing) + [1])
    return np.matmul(affine, spacing)


def read_nii(path, method='nib'):
    """
    Read ".nii.gz" data
    :param path: path to image
    :param method: method to read data, only support ('nib', 'sitk')
    :returns:
      data    : numpy data [channel, x, y]
      spacing : (x_spacing, y_spacing, z_spacing)

    """
    import SimpleITK as sitk
    path = str(path)
    method = method.lower()
    if method == 'nib':
        from nibabel.filebasedimages import ImageFileError
        try:
            img = nib.load(path)
            data = img.get_data()
            spacing = img.header.get_zooms()[:3]
            affine = img.affine
            return data, spacing, affine
        except ImageFileError as e:
            method = 'sitk'

    if method == 'sitk':
        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img)
        # channel first
        spacing = img.GetSpacing()[::-1]
        affine = make_affine(spacing)
        return data, spacing, affine
    else:
        raise Exception("method only supports nib(nibabel) or sitk(SimpleITK)")


def read_img(img_path):
    img_path = str(img_path)

    import skimage.io as skio
    if img_path.endswith('jpg') or img_path.endswith('png') or img_path.endswith('bmp'):
        return skio.imread(img_path)
    elif img_path.endswith('nii.gz') or img_path.endswith('.dcm'):
        return read_nii(img_path)[0]
    elif img_path.endswith('npy'):
        return np.load(img_path)
    elif img_path.endswith('.mhd'):
        return read_nii(img_path, method='sitk')[0]
    else:
        raise Exception("Error file format for {}, only support ['bmp', 'jpg', 'png', 'nii.gz', 'npy', 'mhd']".format(img_path))


"""
To process quickly, use multi-cpu to process functions
"""
def multiprocess_task(func, dynamic_args, static_args=(), split_func=np.array_split, ret=False, cpu_num=None):
    """
    Process task with multi cpus.
    :param func: task to be processed, func must be the top level function
    :param dynamic_args: args to be split to assign to cpus,
              it is a list by default
    :param static_args:  args doesn't need to be split
    :param split_func:   function to split args, use the function
              to split a list args by default
    :return:
    """
    start = time.time()
    if cpu_num is None:
        cpu_num = multiprocessing.cpu_count() // 2

    if cpu_num <= 1:
        ret = func(dynamic_args, *static_args)
    else:
        # split dynamic args with cpu num
        dynamic_args_splits = split_func(dynamic_args, cpu_num)
        workers = multiprocessing.Pool(processes=cpu_num)
        processes = []
        for proc_id, dynamic_args in enumerate(dynamic_args_splits):
            # do processing, concat dynamic args and static args
            dynamic_args = list(dynamic_args)
            p = workers.apply_async(func, (dynamic_args, *static_args))
            processes.append(p)
        workers.close()
        workers.join()

    duration = time.time() - start
    print('total time : {} min'.format(duration / 60.))

    if ret:
        # collect results
        if cpu_num > 1:
            res = []
            for p in processes:
                p = p.get()
                res.extend(p)
        else:
            res = ret
        return res


