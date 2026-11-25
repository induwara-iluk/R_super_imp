"""
rsuper_msd.py - MSD Task07 Pancreas -> R-Super training format + synthetic radiology reports.

Course-project substitution (documented in the notebook):
  * Paper: UCSF CT-report pairs (private) + AbdomenAtlas masks.  Here: MSD Task07 Pancreas only.
  * "Reports" are SYNTHETIC: tumour count, location (head/body/tail) and two axial diameters are
    measured on the MSD tumour mask and perturbed with log-normal noise. They stand in for the
    LLM-extracted report fields the paper uses (R-Super report_extraction/ -> per-tumour CSV).

What is copied verbatim from the repo (see 'repo code' below):
  * dataset_conversion/utils.py      : reorient_image, ResampleXYZAxis, ResampleLabelToRef
  * dataset_conversion/nii2npz.py    : pad()  (pad every axis to >= 128)
  and the same steps as abdomenatlas_3d.py + nii2npz.py:
      reorient RAI -> 1x1x1 mm (BSpline in-plane, nearest along z; nearest for labels)
      -> clip HU [-991, 500] -> z-score -> pad -> np.savez_compressed (key 'arr_0')
      image: float32 (D,H,W);  labels: int8 (C,D,H,W), one channel per class, classes sorted by name.

Deviations (all disclosed):
  1. Before normalisation each CT is cropped to the body bounding box (HU > -500); no label is used.
  2. Training cases are then cropped to the pancreas bounding box + margin to save disk.  The
     repo's loader samples 96^3/128^3 patches around organs/tumours anyway.  Eval cases keep the
     whole body crop (no label is used to crop them).
  3. MSD has no pancreas head/body/tail masks.  We split the pancreas mask in thirds along its
     principal axis (head = patient-right end).  The paper gets these from an nnU-Net sub-segment model.
  4. Organ masks for report-only cases are the MSD ground-truth pancreas (label >= 1), not predicted.
"""
import os
import json
import math
import zipfile
import tempfile

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

MASK_CLASSES = ['pancreas', 'pancreas_body', 'pancreas_head', 'pancreas_tail', 'pancreatic_lesion']
REPORT_CLASSES = ['pancreas', 'pancreas_body', 'pancreas_head', 'pancreas_tail']   # NO lesion channel
SEGMENTS = ['head', 'body', 'tail']
SPLIT_DIRS = {'mask_train': 'mask_npz', 'report_only': 'report_npz', 'val': 'eval_npz', 'test': 'eval_npz'}

# Columns of the repo's per-tumour report table (rsuper_train/Merlin_per_tumor_metadata_hf_clean.csv),
# which training/dataset/dim3/dataset_abdomenatlas_UFO.py reads.
REPORT_COLUMNS = ['Encrypted Accession Number', 'Tumor ID', 'Organ', 'Tumor Type', 'Tumor Location',
                  'Tumor Size (mm)', 'Type Certainty', 'BDMAP ID', 'no lesion', 'Standardized Organ',
                  'Standardized Location', 'Unknow Tumor Size']


# ----------------------------------------------------------------------------- repo code
# Copied verbatim from R-Super (no clone needed). U = dataset_conversion/utils.py, N = nii2npz.py
# --- copied verbatim: R-Super@0cac36f rsuper_train/dataset_conversion/utils.py lines 10-24 (ResampleXYZAxis)
def ResampleXYZAxis(imImage, space=(1., 1., 1.), interp=sitk.sitkLinear):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    sp1 = imImage.GetSpacing()
    sz1 = imImage.GetSize()

    sz2 = (int(round(sz1[0]*sp1[0]*1.0/space[0])), int(round(sz1[1]*sp1[1]*1.0/space[1])), int(round(sz1[2]*sp1[2]*1.0/space[2])))

    imRefImage = sitk.Image(sz2, imImage.GetPixelIDValue())
    imRefImage.SetSpacing(space)
    imRefImage.SetOrigin(imImage.GetOrigin())
    imRefImage.SetDirection(imImage.GetDirection())

    imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

    return imOutImage


# --- copied verbatim: R-Super@0cac36f rsuper_train/dataset_conversion/utils.py lines 26-36 (ResampleLabelToRef)
def ResampleLabelToRef(imLabel, imRef, interp=sitk.sitkNearestNeighbor):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)

    imRefImage = sitk.Image(imRef.GetSize(), imLabel.GetPixelIDValue())
    imRefImage.SetSpacing(imRef.GetSpacing())
    imRefImage.SetOrigin(imRef.GetOrigin())
    imRefImage.SetDirection(imRef.GetDirection())
        
    ResampledLabel = sitk.Resample(imLabel, imRefImage, identity1, interp)
    
    return ResampledLabel


# --- copied verbatim: R-Super@0cac36f rsuper_train/dataset_conversion/utils.py lines 38-47 (reorient_image)
def reorient_image(image, desired_orientation='RAI'):
    
    current_orientation = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(image.GetDirection())

    if current_orientation != desired_orientation:
        reorient_filter = sitk.DICOMOrientImageFilter()
        reorient_filter.SetDesiredCoordinateOrientation(desired_orientation)
        image = reorient_filter.Execute(image)

    return image


# --- copied verbatim: R-Super@0cac36f rsuper_train/dataset_conversion/nii2npz.py lines 13-29 (pad)
def pad(img, lab):
    z, y, x = img.shape
    # pad if the image size is smaller than training size
    if z < 128:
        diff = int(math.ceil((128. - z) / 2)) 
        img = np.pad(img, ((diff, diff), (0, 0), (0, 0)))
        lab = np.pad(lab, ((0, 0), (diff, diff), (0, 0), (0, 0)))
    if y < 128:
        diff = int(math.ceil((128. - y) / 2)) 
        img = np.pad(img, ((0, 0), (diff, diff), (0, 0)))
        lab = np.pad(lab, ((0, 0), (0, 0), (diff, diff), (0, 0)))
    if x < 128:
        diff = int(math.ceil((128. - x) / 2)) 
        img = np.pad(img, ((0, 0), (0, 0), (diff, diff)))
        lab = np.pad(lab, ((0, 0), (0, 0), (0, 0), (diff, diff)))

    return img, lab


import types
_U = types.SimpleNamespace(ResampleXYZAxis=ResampleXYZAxis, ResampleLabelToRef=ResampleLabelToRef,
                           reorient_image=reorient_image)
_N = types.SimpleNamespace(pad=pad)


def repo_fns(repo_train_dir=None):
    """The repo's conversion helpers (copied above)."""
    return _U, _N


# ----------------------------------------------------------------------------- MSD source
def bdmap_id(case):
    """pancreas_001 -> BDMAP_00000001 (the repo slices IDs as f[:len('BDMAP_00000000')])."""
    return 'BDMAP_%08d' % int(case.split('_')[-1])


class MSDSource:
    """Reads MSD Task07 cases from the nnU-Net style zip (Dataset007_Pancreas.zip) or an extracted folder."""

    def __init__(self, src):
        self.src = src
        self.is_zip = src.endswith('.zip')
        if not self.is_zip:
            inner = os.path.join(src, 'Dataset007_Pancreas')
            self.root = inner if os.path.isdir(inner) else src

    def _rel(self, case, kind):
        if kind == 'image':
            return 'Dataset007_Pancreas/imagesTr/%s_0000.nii.gz' % case
        return 'Dataset007_Pancreas/labelsTr/%s.nii.gz' % case

    def list_cases(self):
        if self.is_zip:
            with zipfile.ZipFile(self.src) as z:
                names = z.namelist()
        else:
            names = ['Dataset007_Pancreas/imagesTr/' + f for f in os.listdir(os.path.join(self.root, 'imagesTr'))]
        cases = [os.path.basename(n)[:-len('_0000.nii.gz')] for n in names
                 if '/imagesTr/' in n and n.endswith('_0000.nii.gz')]
        return sorted(cases)

    def read(self, case, kind):
        rel = self._rel(case, kind)
        if not self.is_zip:
            return _read_sitk(os.path.join(self.root, rel.split('/', 1)[1]))
        with zipfile.ZipFile(self.src) as z:
            data = z.read(rel)
        fd, tmp = tempfile.mkstemp(suffix='.nii.gz')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            return _read_sitk(tmp)
        finally:
            os.remove(tmp)


def _read_sitk(path):
    """sitk.ReadImage with the repo's nibabel fallback for non-orthonormal direction cosines
    (same idea as read_sitk_with_nib_fallback in dataset_conversion/abdomenatlas_3d.py)."""
    try:
        return sitk.ReadImage(path)
    except RuntimeError as e:
        if 'orthonormal' not in str(e).lower():
            raise
        import nibabel as nib
        img = nib.load(path)
        aff = img.affine.copy()
        u, _, vt = np.linalg.svd(aff[:3, :3])
        aff[:3, :3] = u @ vt
        fd, tmp = tempfile.mkstemp(suffix='.nii.gz')
        os.close(fd)
        try:
            nib.Nifti1Image(np.asanyarray(img.dataobj), aff, img.header).to_filename(tmp)
            return sitk.ReadImage(tmp)
        finally:
            os.remove(tmp)


def scan_case(args):
    """Label-only pass: voxel counts, used to decide which cases contain tumour before splitting."""
    src, case = args
    lab = sitk.GetArrayFromImage(MSDSource(src).read(case, 'label'))
    return case, int((lab >= 1).sum()), int((lab == 2).sum())


def make_splits(tumor_cases, sizes, seed=42):
    """Deterministic split of tumour cases (sorted, then permuted with numpy default_rng(seed))."""
    cases = sorted(tumor_cases)
    perm = np.random.default_rng(seed).permutation(len(cases))
    cases = [cases[i] for i in perm]
    out, i = {}, 0
    for name, n in sizes:
        out[name] = sorted(cases[i:i + n])
        i += n
    out['unused'] = sorted(cases[i:])
    return out


# ----------------------------------------------------------------------------- geometry
def resample_1mm(img, lab, U):
    """Exactly the resampling of ResampleImage() in dataset_conversion/abdomenatlas_3d.py."""
    img = U.reorient_image(img, 'RAI')
    lab = U.reorient_image(lab, 'RAI')
    if lab.GetSize() != img.GetSize():
        raise ValueError('image/label size mismatch %s vs %s' % (img.GetSize(), lab.GetSize()))
    if (np.allclose(lab.GetOrigin(), img.GetOrigin(), atol=1e-2)
            and np.allclose(lab.GetDirection(), img.GetDirection(), atol=1e-3)
            and np.allclose(lab.GetSpacing(), img.GetSpacing(), atol=1e-3)):
        lab.CopyInformation(img)  # remove float noise in headers before nearest-neighbour resampling
    sp = img.GetSpacing()
    img_xy = U.ResampleXYZAxis(img, space=(1.0, 1.0, sp[2]), interp=sitk.sitkBSpline)
    lab_xy = U.ResampleLabelToRef(lab, img_xy, interp=sitk.sitkNearestNeighbor)
    img_xyz = U.ResampleXYZAxis(img_xy, space=(1.0, 1.0, 1.0), interp=sitk.sitkNearestNeighbor)
    lab_xyz = U.ResampleLabelToRef(lab_xy, img_xyz, interp=sitk.sitkNearestNeighbor)
    return img_xyz, lab_xyz


def body_bbox(ct, thr=-500.0, margin=5, step=4):
    """Bounding box (z,y,x) of the largest connected component with HU > thr. Uses no label."""
    small = ct[::step, ::step, ::step] > thr
    cc, n = ndi.label(small)
    if n == 0:
        return [(0, s) for s in ct.shape]
    k = int(np.argmax(ndi.sum(small, cc, range(1, n + 1)))) + 1
    sl = ndi.find_objects((cc == k).astype(np.int32))[0]
    return [(max(0, s.start * step - margin), min(dim, s.stop * step + margin)) for s, dim in zip(sl, ct.shape)]


def mask_bbox(mask, margin, shape):
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    return [(max(0, int(a) - margin), min(int(d), int(b) + margin)) for a, b, d in zip(lo, hi, shape)]


def crop(arr, bb):
    return arr[..., bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]]


def split_pancreas(panc, origin, direction):
    """Approximate head/body/tail: thirds along the pancreas principal axis.
    ITK physical space is LPS (+x = patient left), so the axis is oriented towards +x:
    the patient-right end is the head, the left end is the tail."""
    zyx = np.argwhere(panc)
    masks = {s: np.zeros(panc.shape, bool) for s in SEGMENTS}
    if len(zyx) < 30:
        masks['body'][panc] = True
        return masks, None
    xyz = zyx[:, ::-1].astype(np.float64)                       # numpy (z,y,x) -> ITK (x,y,z)
    D = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    phys = xyz @ D.T + np.asarray(origin, dtype=np.float64)     # spacing is 1 mm after resampling
    X = phys - phys.mean(0)
    rng = np.random.default_rng(0)
    sub = X[rng.choice(len(X), size=min(len(X), 20000), replace=False)]
    v = np.linalg.svd(sub, full_matrices=False)[2][0]
    if v[0] < 0:
        v = -v
    t = X @ v
    lo, hi = np.percentile(t, [1, 99])
    b1, b2 = lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0
    seg = np.where(t < b1, 0, np.where(t < b2, 1, 2))
    for k, s in enumerate(SEGMENTS):
        sel = zyx[seg == k]
        masks[s][sel[:, 0], sel[:, 1], sel[:, 2]] = True
    return masks, v.tolist()


def _extents(pts):
    """Extent (mm) of a point cloud along its principal axes, largest first (+1 voxel)."""
    if len(pts) < 2:
        return [1.0] * pts.shape[1]
    X = pts - pts.mean(0)
    vt = np.linalg.svd(X, full_matrices=False)[2]
    proj = X @ vt.T
    ext = proj.max(0) - proj.min(0) + 1.0
    return sorted(ext.tolist(), reverse=True)


def tumor_attributes(lesion, seg_masks, min_vox):
    """The simulated 'radiologist': reads the tumour mask and returns what a report would say
    (count, location, two axial diameters) plus true values kept only for analysis."""
    cc, _ = ndi.label(lesion, structure=np.ones((3, 3, 3), bool))
    tumors = []
    for i, sl in enumerate(ndi.find_objects(cc), 1):
        if sl is None:
            continue
        comp = cc[sl] == i
        vol = int(comp.sum())
        areas = comp.sum(axis=(1, 2))
        zk = int(np.argmax(areas))                            # axial slice with the largest cross-section
        d1, d2 = _extents(np.argwhere(comp[zk]).astype(np.float64))[:2] if areas[zk] > 1 else (1.0, 1.0)
        ext3 = _extents(np.argwhere(comp).astype(np.float64))
        frac = {s: float((comp & seg_masks[s][sl]).sum()) / vol for s in SEGMENTS}
        loc = [s for s in SEGMENTS if frac[s] >= 0.15] or [max(frac, key=frac.get)]
        cz, cy, cx = (np.argwhere(comp).mean(0) + [s.start for s in sl]).tolist()
        tumors.append(dict(
            true_volume_mm3=vol,
            d1_axial_mm=float(d1), d2_axial_mm=float(d2),
            pca3d_extents_mm=[float(e) for e in ext3],
            craniocaudal_mm=int(comp.any(axis=(1, 2)).sum()),
            location=' / '.join(loc), segment_fraction=frac,
            centroid_zyx=[cz, cy, cx]))
    tumors.sort(key=lambda t: -t['true_volume_mm3'])
    for t in tumors:
        t['reported'] = t['true_volume_mm3'] >= min_vox
    if tumors and not any(t['reported'] for t in tumors):
        tumors[0]['reported'] = True                          # every tumour case reports >= 1 tumour
    return tumors


# ----------------------------------------------------------------------------- conversion
def convert_case(job):
    """One MSD case -> <out>/<mask_npz|report_npz|eval_npz>/BDMAP_xxxxxxxx{.npz,_gt.npz} + attributes."""
    case, split, cfg = job['case'], job['split'], job['cfg']
    bd = bdmap_id(case)
    dst = os.path.join(cfg['out'], SPLIT_DIRS[split])
    cache = os.path.join(cfg['out'], 'meta', 'case_cache', bd + '.json')
    if os.path.exists(cache) and os.path.exists(os.path.join(dst, bd + '_gt.npz')):
        with open(cache) as f:
            return json.load(f)

    U, N = repo_fns(cfg['repo_train_dir'])
    src = MSDSource(cfg['src'])
    img0, lab0 = src.read(case, 'image'), src.read(case, 'label')
    img, lab = resample_1mm(img0, lab0, U)
    ct = sitk.GetArrayFromImage(img).astype(np.float32)        # (z,y,x), HU
    L = sitk.GetArrayFromImage(lab).astype(np.uint8)

    # 1) body crop (no label used) and index -> physical origin update
    bb = body_bbox(ct)
    D = np.asarray(img.GetDirection()).reshape(3, 3)
    origin = (np.asarray(img.GetOrigin()) + D @ np.array([bb[2][0], bb[1][0], bb[0][0]], float)).tolist()
    ct, L = crop(ct, bb), crop(L, bb)
    panc_lost = int((sitk.GetArrayFromImage(lab) >= 1).sum() - (L >= 1).sum())

    # 2) normalisation exactly as nii2npz.py (statistics over the body crop)
    ct = np.clip(ct, -991, 500)
    ct = (ct - ct.mean()) / ct.std()

    panc, lesion = L >= 1, L == 2          # pancreas mask includes the tumour (tumour lies inside the organ)
    segs, axis = split_pancreas(panc, origin, img.GetDirection())

    # 3) training cases: crop to pancreas bbox + margin (disk); eval cases keep the body crop
    if split in ('mask_train', 'report_only'):
        bb2 = mask_bbox(panc, cfg['train_margin'], panc.shape)
        ct, panc, lesion = crop(ct, bb2), crop(panc, bb2), crop(lesion, bb2)
        segs = {k: crop(v, bb2) for k, v in segs.items()}

    # 4) simulated radiologist reads the tumour mask -> report attributes.
    #    For report_only cases THIS IS THE ONLY USE OF THE TUMOUR MASK; it is not written to disk below.
    tumors = tumor_attributes(lesion, segs, cfg['min_tumor_vox'])

    channels = {'pancreas': panc, 'pancreas_head': segs['head'], 'pancreas_body': segs['body'],
                'pancreas_tail': segs['tail']}
    if split != 'report_only':
        channels['pancreatic_lesion'] = lesion
    names = sorted(channels)
    assert names == (REPORT_CLASSES if split == 'report_only' else MASK_CLASSES)
    lab_arr = np.stack([channels[n] for n in names]).astype(np.int8)
    ct, lab_arr = N.pad(ct, lab_arr)

    os.makedirs(dst, exist_ok=True)
    np.savez_compressed(os.path.join(dst, bd + '.npz'), ct.astype(np.float32))
    np.savez_compressed(os.path.join(dst, bd + '_gt.npz'), lab_arr.astype(np.int8))

    rec = dict(case=case, bdmap=bd, split=split, folder=SPLIT_DIRS[split],
               orig_spacing=list(img0.GetSpacing()), orig_size=list(img0.GetSize()),
               npz_shape=list(ct.shape), channels=names, pancreas_axis_lps=axis,
               pancreas_vox=int(panc.sum()), pancreas_vox_lost_by_body_crop=panc_lost,
               lesion_vox=int(lesion.sum()), tumors=tumors)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, 'w') as f:
        json.dump(rec, f)
    return rec


def write_lists(out, splits, records):
    """list/dataset.yaml and list/label_names.yaml, as written by abdomenatlas_3d.py / nii2npz.py."""
    import yaml
    for folder, classes in [('mask_npz', MASK_CLASSES), ('report_npz', REPORT_CLASSES), ('eval_npz', MASK_CLASSES)]:
        ids = sorted(r['bdmap'] for r in records if r['folder'] == folder)
        os.makedirs(os.path.join(out, folder, 'list'), exist_ok=True)
        with open(os.path.join(out, folder, 'list', 'dataset.yaml'), 'w') as f:
            yaml.dump(ids, f)
        with open(os.path.join(out, folder, 'list', 'label_names.yaml'), 'w') as f:
            yaml.dump(list(classes), f)


# ----------------------------------------------------------------------------- reports
def add_noise_draws(records, seed=42):
    """Fix one pair of standard-normal draws per reported tumour, so that reports at different
    sigma differ only in noise magnitude: d_report = round(d_true * exp(sigma * z))."""
    rng = np.random.default_rng(seed)
    for r in sorted(records, key=lambda r: r['bdmap']):
        for t in r['tumors']:
            t['noise_z'] = rng.standard_normal(2).tolist()
    return records


def report_rows(records, sigma, ids=None):
    """Rows in the repo's per-tumour report format. Size string 'd1 x d2' (mm): the repo sets
    d3 = (d1+d2)/2 and volume = 4/3*pi*(d1/2)(d2/2)(d3/2) (estimate_tumor_volume)."""
    rows, analysis = [], []
    for r in sorted(records, key=lambda r: r['bdmap']):
        if ids is not None and r['bdmap'] not in ids:
            continue
        reported = [t for t in r['tumors'] if t['reported']]
        if not reported:
            rows.append({'Encrypted Accession Number': 'SYN_' + r['bdmap'], 'Tumor ID': 'no lesion',
                         'Organ': 'no lesion', 'Tumor Type': 'no lesion', 'Tumor Location': 'no lesion',
                         'Tumor Size (mm)': 'U', 'Type Certainty': 'no lesion', 'BDMAP ID': r['bdmap'],
                         'no lesion': True, 'Standardized Organ': 'u', 'Standardized Location': '',
                         'Unknow Tumor Size': 'no'})
            continue
        for k, t in enumerate(reported, 1):
            z = t['noise_z']
            d1 = max(int(round(t['d1_axial_mm'] * math.exp(sigma * z[0]))), 3)
            d2 = max(int(round(t['d2_axial_mm'] * math.exp(sigma * z[1]))), 3)
            d1, d2 = max(d1, d2), min(d1, d2)
            size = '%.1f x %.1f' % (d1, d2)
            rows.append({'Encrypted Accession Number': 'SYN_' + r['bdmap'], 'Tumor ID': 'tumor %d' % k,
                         'Organ': 'pancreas', 'Tumor Type': 'pancreatic mass (MSD label 2)',
                         'Tumor Location': t['location'], 'Tumor Size (mm)': size, 'Type Certainty': 'synthetic',
                         'BDMAP ID': r['bdmap'], 'no lesion': False, 'Standardized Organ': 'pancreas',
                         'Standardized Location': t['location'], 'Unknow Tumor Size': 'no'})
            d3 = (d1 + d2) / 2.0
            analysis.append(dict(bdmap=r['bdmap'], split=r['split'], tumor=k, sigma=sigma, size=size,
                                 location=t['location'], d1_true=t['d1_axial_mm'], d2_true=t['d2_axial_mm'],
                                 true_volume_mm3=t['true_volume_mm3'],
                                 report_volume_mm3=4.0 / 3.0 * math.pi * (d1 / 2) * (d2 / 2) * (d3 / 2)))
    return rows, analysis


def per_ct_metadata(records):
    """Per-CT table in the format the repo's evaluation scripts expect
    ('number of pancreatic lesion instances', see calculate_sensitivity_specificity.py)."""
    out = []
    for r in sorted(records, key=lambda r: r['bdmap']):
        rep = [t for t in r['tumors'] if t['reported']]
        out.append({'BDMAP ID': r['bdmap'], 'split': r['split'], 'msd_case': r['case'],
                    'number of pancreatic lesion instances': len(rep),
                    'largest pancreatic lesion diameter (cm)': (max(t['d1_axial_mm'] for t in rep) / 10.0) if rep else 0.0,
                    'largest pancreatic lesion location (head, body, tail)': rep[0]['location'] if rep else '',
                    'no lesion': int(len(rep) == 0)})
    return out
