"""Analyse one Kaggle run of R_super_1.ipynb: re-score the saved predictions and draw the figures.

usage:  python analysis/analyze_run.py results/run1 [--data <rsuper_data dir>] [--workers 3] [--rescore]

Expects the folder layout of the Kaggle output (downloaded with `kaggle kernels output ... --file-pattern`):
  exp/<run>/history.jsonl       per-epoch losses
  logs/<run>.log, logs/<run>.json   training logs and configs
  results/run_status.json, results/time_plan.json
  results/test/<model>/<id>.npz  thresholded predictions (key 'pred', 5 channels)
  rsuper_data/eval_npz/<id>_gt.npz, rsuper_data/meta/ids_test.csv, rsuper_data/meta/report_analysis.csv

Writes <run>/analysis/*.csv and <run>/figures/*.png.

Two scoring regions:
  whole_ct     : the body-cropped test CT, as the notebook's evaluation did.
  pancreas_roi : the ground-truth pancreas bounding box + 48 mm, i.e. the same field of view the training crops
                 come from (rsuper_msd.py crops training CTs to the pancreas bbox + TRAIN_MARGIN = 48 mm).
                 The ROI uses the ground-truth pancreas, which is the same assumption as the training data
                 (MSD ground-truth pancreas instead of the paper's nnU-Net organ masks).
"""
import json, os, re, sys, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats

CLASSES = ['pancreas', 'pancreas_body', 'pancreas_head', 'pancreas_tail', 'pancreatic_lesion']
MODELS = ['baseline', 'control', 'volume', 'volume_ball']     # narrowed in main() to the models with predictions
MIN_VOX = 50
ROI_MARGIN = 48
STRUCT = np.ones((3, 3, 3), bool)

# ------------------------------------------------------------------------------------------- scoring
def components(mask):
    cc, n = ndi.label(mask, structure=STRUCT)
    if not n:
        return cc, [], []
    sizes = ndi.sum(mask, cc, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= MIN_VOX]
    return cc, keep, [int(sizes[i - 1]) for i in keep]


def roi_slices(panc, margin, shape):
    idx = np.argwhere(panc)
    lo, hi = idx.min(0), idx.max(0) + 1
    return tuple(slice(max(0, int(a) - margin), min(int(d), int(b) + margin)) for a, b, d in zip(lo, hi, shape))


def score_case(args):
    run, data, bd = args
    gt = np.load(f'{data}/eval_npz/{bd}_gt.npz')['arr_0']
    les_gt_full, panc_gt_full = gt[4] > 0, gt[0] > 0
    roi = roi_slices(panc_gt_full, ROI_MARGIN, panc_gt_full.shape)
    # distance (mm) from the GT pancreas on a 2 mm grid (memory), read back at full resolution (error <= ~1.7 mm)
    dist_half = ndi.distance_transform_edt(~panc_gt_full[::2, ::2, ::2], sampling=2).astype(np.float32)
    rows, lesions = [], []
    for m in MODELS:
        pr = np.load(f'{run}/results/test/{m}/{bd}.npz')['pred']
        les_p_full, panc_p_full = pr[4] > 0, pr[0] > 0
        # where the predicted lesion voxels are (distance from the GT pancreas, mm)
        zz, yy, xx = np.nonzero(les_p_full)
        d = dist_half[zz // 2, yy // 2, xx // 2]
        loc = dict(pred_les_within_10mm=float(np.mean(d <= 10)) if d.size else np.nan,
                   pred_les_beyond_48mm=float(np.mean(d > 48)) if d.size else np.nan)
        for region, sl in [('whole_ct', (slice(None),) * 3), ('pancreas_roi', roi)]:
            les_gt, panc_gt, les_p, panc_p = les_gt_full[sl], panc_gt_full[sl], les_p_full[sl], panc_p_full[sl]
            organ = ndi.binary_dilation(panc_p, structure=STRUCT)
            for variant, lp in [('raw', les_p), ('organ_masked', les_p & organ)]:
                s = lp.sum() + les_gt.sum()
                ccg, gid, gsz = components(les_gt)
                ccp, pid, _ = components(lp)
                det = [bool(lp[ccg == i].any()) for i in gid]
                fp = sum(not les_gt[ccp == j].any() for j in pid)
                rows.append(dict(model=m, case=bd, region=region, variant=variant,
                                 dice_lesion=2 * (lp & les_gt).sum() / s if s else 1.0,
                                 dice_pancreas=2 * (panc_p & panc_gt).sum() / max(1, panc_p.sum() + panc_gt.sum()),
                                 gt_lesions=len(gid), tp=sum(det), fp=fp, pred_vol=int(lp.sum()), gt_vol=int(les_gt.sum()),
                                 patient_detected=int(lp.sum() >= MIN_VOX), **loc))
                if variant == 'raw':
                    lesions += [dict(model=m, case=bd, region=region, lesion_vox=v, detected=t) for v, t in zip(gsz, det)]
    return rows, lesions


def score_run(run, data, workers=3):
    ids = pd.read_csv(f'{data}/meta/ids_test.csv')['BDMAP ID'].tolist()
    ids = [b for b in ids if all(os.path.exists(f'{run}/results/test/{m}/{b}.npz') for m in MODELS)]
    rows, lesions = [], []
    with ProcessPoolExecutor(workers) as ex:
        for k, (r, l) in enumerate(ex.map(score_case, [(run, data, b) for b in ids])):
            rows += r; lesions += l
            print(f'\rscored {k + 1}/{len(ids)}', end='', flush=True)
    print()
    return pd.DataFrame(rows), pd.DataFrame(lesions)


def summarize(df):
    g = df.groupby(['region', 'variant', 'model'], sort=False)
    s = g.agg(dice_lesion_mean=('dice_lesion', 'mean'), dice_lesion_sd=('dice_lesion', 'std'),
              dice_pancreas=('dice_pancreas', 'mean'), tp=('tp', 'sum'), fp=('fp', 'sum'),
              fp_per_ct=('fp', 'mean'), n_gt_lesions=('gt_lesions', 'sum'), patient_sens=('patient_detected', 'mean'),
              pred_vol_median=('pred_vol', 'median'), gt_vol_median=('gt_vol', 'median'), n=('case', 'nunique'))
    s['lesion_recall'] = s['tp'] / s['n_gt_lesions'].clip(lower=1)
    s['lesion_precision'] = s['tp'] / (s['tp'] + s['fp']).clip(lower=1)
    s['lesion_F1'] = 2 * s['lesion_precision'] * s['lesion_recall'] / (s['lesion_precision'] + s['lesion_recall']).clip(lower=1e-9)
    return s.round(4).reset_index()


def paired(df, region='pancreas_roi', variant='raw', metric='dice_lesion', refs=('baseline', 'control'), n_boot=5000, seed=0):
    """Per-case paired differences (model - reference), bootstrap 95 % CI and Wilcoxon signed-rank p."""
    d = df[(df.region == region) & (df.variant == variant)].pivot(index='case', columns='model', values=metric)
    rng = np.random.default_rng(seed)
    out = []
    for ref in [r for r in refs if r in d]:
        for m in MODELS:
            if m == ref:
                continue
            diff = (d[m] - d[ref]).to_numpy()
            boots = np.array([rng.choice(diff, diff.size).mean() for _ in range(n_boot)])
            p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            out.append(dict(metric=metric, region=region, variant=variant, model=m, reference=ref,
                            mean_diff=diff.mean(), ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                            wins=int((diff > 0).sum()), losses=int((diff < 0).sum()), wilcoxon_p=p))
    return pd.DataFrame(out)

# ------------------------------------------------------------------------------------------- logs
def histories(run):
    h = {}
    for m in MODELS:
        p = f'{run}/exp/{m}/history.jsonl'
        if os.path.exists(p):
            h[m] = pd.DataFrame([json.loads(l) for l in open(p)])
    return h


def lr_from_log(run, m, h=None):
    if h is not None and 'lr' in h:
        return h['lr'].tolist()
    lrs = [float(x) for x in re.findall(r'Starting epoch \d+/\d+ lr ([0-9.e+-]+)', open(f'{run}/logs/{m}.log', errors='replace').read())]
    return lrs

# ------------------------------------------------------------------------------------------- figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# reference palette (dataviz skill, light mode); baseline = neutral ink, fine-tunes = categorical slots 1-3
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
COL = {'baseline': '#7a7974', 'control': '#2a78d6', 'volume': '#eb6834', 'volume_ball': '#1baf7a'}
LABEL = {'baseline': 'baseline (50 masks)', 'control': 'control (+reports, no report loss)',
         'volume': 'volume (+Volume Loss)', 'volume_ball': 'volume_ball (+Ball & Volume Loss)'}
SHORT = {'baseline': 'baseline', 'control': 'control', 'volume': 'volume', 'volume_ball': 'volume_ball'}
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
                     'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'legend.frameon': False,
                     'lines.linewidth': 2})


def save(fig, out, name):
    fig.savefig(f'{out}/{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('figure', name)


def fig_training(h, run, out):
    base = h['baseline']; nb = len(base)
    fts = [m for m in ['control', 'volume', 'volume_ball'] if m in h]
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[3, 1.4])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(base.epoch, base.segmentation, color=COL['baseline'], label=LABEL['baseline'])
    for m in fts:
        x = np.r_[nb, nb + h[m].epoch]
        ax.plot(x, np.r_[base.segmentation.iloc[-1], h[m].segmentation], color=COL[m], label=LABEL[m])
    ax.axvline(nb + 0.5, color=MUTED, lw=1, ls='--')
    ax.annotate('fine-tunes start\nfrom the baseline', (nb + 0.5, base.segmentation.max() * 0.75), xytext=(-8, 0),
                textcoords='offset points', ha='right', fontsize=9, color=INK2)
    ax.set_ylabel('segmentation loss (BCE + Dice)')
    ax.set_title('Segmentation loss per epoch (200 iterations each)')
    ax.legend(loc='upper right', fontsize=9)
    # zoom: the fine-tunes nearly overlap, so show each minus control
    axz = fig.add_subplot(gs[0, 1])
    if 'control' in h:
        for m in fts:
            n = min(len(h[m]), len(h['control']))
            axz.plot(h[m].epoch[:n], h[m].segmentation[:n].to_numpy() - h['control'].segmentation[:n].to_numpy(), color=COL[m],
                     marker='o', ms=3, label=SHORT[m])
        axz.axhline(0, color=INK, lw=1)
        axz.set_title('Fine-tunes: minus control', fontsize=10); axz.set_xlabel('fine-tune epoch')
        axz.set_ylabel('segmentation loss difference'); axz.legend(fontsize=8.5, loc='best')
        axz.xaxis.get_major_locator().set_params(integer=True)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(base.epoch, lr_from_log(run, 'baseline', base), color=COL['baseline'])
    for m in fts:
        ax.plot(nb + h[m].epoch, lr_from_log(run, m, h[m]), color=COL[m])
    ax.set_yscale('log'); ax.set_ylabel('learning rate'); ax.set_xlabel('epoch (baseline 1-%d, then fine-tune)' % nb)
    ax.set_title('Learning rate (exponential warm-up, then poly decay)')
    # crop statistics (written to history.jsonl from run 2 on)
    axc = fig.add_subplot(gs[1, 1])
    rows = [(m, h[m]['crops'].iloc[-1]) for m in fts if 'crops' in h[m] and isinstance(h[m]['crops'].iloc[-1], dict)]
    if rows:
        kinds = ['mask_case', 'on_segment', 'failed', 'random']
        shades = ['#c9c8c2', COL['volume_ball'], COL['volume'], '#a3a29c']
        for i, (m, c) in enumerate(rows):
            left = 0
            for k, col in zip(kinds, shades):
                v = c.get(k, 0)
                axc.barh(i, v, left=left, color=col, edgecolor=SURF, lw=1.5, label=k if i == 0 else None)
                left += v
        axc.set_yticks(range(len(rows)), [SHORT[m] for m, _ in rows]); axc.invert_yaxis()
        axc.set_xlabel('samples in the last epoch'); axc.legend(fontsize=8, loc='lower right', ncol=2)
        axc.set_title('What the fine-tune batches contained', fontsize=10)
    else:
        axc.axis('off')
        axc.text(0.0, 0.5, 'Crop statistics: not recorded in this run\n(run 1 counted them inside the data-loader\n'
                 'worker processes, so the log showed zeros)', fontsize=9, color=INK2, va='center')
    fig.tight_layout()
    save(fig, out, '01_training_loss_and_lr')


def fig_report_losses(h, out):
    keys = [('volume', 'dice_volume_loss', 'Volume Loss (both outputs)'),
            ('volume_ball', 'dice_volume_loss', 'Volume Loss (deep-supervision output)'),
            ('volume_ball', 'ball_loss_dice', 'Ball Loss, Dice term'),
            ('volume_ball', 'ball_loss_bce', 'Ball Loss, BCE term')]
    keys = [k for k in keys if k[0] in h and k[1] in h[k[0]]]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 3.2))
    for ax, (m, k, t) in zip(np.atleast_1d(axes), keys):
        ax.plot(h[m].epoch, h[m][k], color=COL[m], marker='o', ms=4)
        ax.set_title(t, fontsize=10); ax.set_xlabel('fine-tune epoch'); ax.set_ylim(bottom=0)
        ax.xaxis.get_major_locator().set_params(integer=True)
    np.atleast_1d(axes)[0].set_ylabel('weighted loss (x 0.1)')
    fig.suptitle('Report losses during fine-tuning (values already multiplied by the report weight 0.1)', fontsize=11, fontweight='bold')
    fig.tight_layout()
    save(fig, out, '02_report_losses')


def fig_time(run, out):
    st = json.load(open(f'{run}/results/run_status.json')); plan = json.load(open(f'{run}/results/time_plan.json'))
    proj = plan.get('projected_h', {})
    stages = [m for m in MODELS if st.get(m, {}).get('hours') is not None]
    actual = [st[s]['hours'] for s in stages]
    planned = [proj.get(s, proj.get('each_finetune', np.nan)) for s in stages]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    y = np.arange(len(stages))
    ax[0].barh(y + 0.2, planned, 0.36, color='#c9c8c2', label='planned (from the probe)')
    ax[0].barh(y - 0.2, actual, 0.36, color=[COL[s] for s in stages], label='actual')
    for i, (p, a) in enumerate(zip(planned, actual)):
        p = 0 if np.isnan(p) else p
        ax[0].text(a + 0.03, i - 0.2, f'{a:.2f} h', va='center', fontsize=9, color=INK2)
        ax[0].text(p + 0.03, i + 0.2, f'{p:.2f} h', va='center', fontsize=9, color=MUTED)
    ax[0].set_yticks(y, stages); ax[0].invert_yaxis(); ax[0].set_xlabel('hours')
    ax[0].set_xlim(0, np.nanmax(planned + actual) * 1.25)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9)
    ax[0].set_title('Planned vs actual training time')
    # whole-run budget
    prep = st.get('prep', {}).get('minutes', 0) / 60
    probes = [v for k, v in st.items() if k.startswith('probe_')]
    probe = sum(v.get('hours', 0) for v in probes)
    evals = [v['updated_h'] for k, v in st.items() if k.startswith('eval_') and 'updated_h' in v]
    end = max(evals) if evals else max(v.get('updated_h', 0) for v in st.values())
    evalh = end - max(st[s]['updated_h'] for s in stages)
    parts = [('data prep', prep, '#c9c8c2'), ('probe', probe, '#a3a29c'), ('training', sum(actual), COL['control']),
             ('evaluation', evalh, COL['volume']), ('unused', max(0, 11.3 - end), '#ffffff')]
    left = 0
    for n, v, c in parts:
        ax[1].barh(0, v, left=left, color=c, edgecolor=MUTED if n == 'unused' else SURF, lw=1.5, hatch='//' if n == 'unused' else None)
        if v >= 0.8:                                  # label wide segments on the bar, small ones in the legend line below
            ax[1].text(left + v / 2, 0.45, f'{n}\n{v:.1f} h', ha='center', va='bottom', fontsize=8.5, color=INK2)
        left += v
    small = ', '.join(f'{n} {v:.2f} h' for n, v, _ in parts if 0 < v < 0.8)
    if small:
        ax[1].text(0, -0.55, small, fontsize=8.5, color=INK2, va='top')
    ax[1].set_xlim(0, 11.5); ax[1].set_ylim(-0.9, 1.3); ax[1].set_yticks([]); ax[1].grid(False, axis='y')
    ax[1].set_xlabel('hours since kernel start (limit 11.3 h)'); ax[1].set_title('How the 11.3 h budget was used')
    fig.tight_layout()
    save(fig, out, '03_time_budget')


def fig_metrics(s, out):
    mets = [('dice_lesion_mean', 'lesion Dice'), ('dice_pancreas', 'pancreas Dice'), ('lesion_recall', 'lesion recall'),
            ('lesion_precision', 'lesion precision'), ('lesion_F1', 'lesion F1')]
    fig, axes = plt.subplots(2, len(mets), figsize=(15, 6), sharey='col')
    for r, region in enumerate(['whole_ct', 'pancreas_roi']):
        d = s[(s.region == region) & (s.variant == 'raw')].set_index('model').loc[MODELS]
        for c, (k, t) in enumerate(mets):
            ax = axes[r, c]
            v = d[k].to_numpy()
            ax.bar(range(len(MODELS)), v, 0.72, color=[COL[m] for m in MODELS], edgecolor=SURF, lw=2)
            for i, x in enumerate(v):
                ax.text(i, x, f'{x:.2f}', ha='center', va='bottom', fontsize=8.5, color=INK2)
            ax.set_xticks(range(len(MODELS)), [SHORT[m] for m in MODELS], rotation=35, ha='right', fontsize=8.5)
            ax.set_ylim(0, 1.0 if k != 'dice_lesion_mean' else max(0.3, v.max() * 1.3))
            if r == 0:
                ax.set_title(t)
        axes[r, 0].set_ylabel({'whole_ct': 'WHOLE CT\n(as evaluated on Kaggle)', 'pancreas_roi': 'PANCREAS ROI\n(pancreas bbox + 48 mm)'}[region])
    fig.suptitle('Test set (50 MSD CTs): accuracy per method', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, out, '04_metrics_per_method')


def fig_fp(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, region, t in [(axes[0], 'whole_ct', 'Whole CT'), (axes[1], 'pancreas_roi', 'Pancreas ROI')]:
        d = df[(df.region == region) & (df.variant == 'raw')]
        rng = np.random.default_rng(0)
        for i, m in enumerate(MODELS):
            v = d[d.model == m].fp.to_numpy()
            ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=14, color=COL[m], alpha=0.7, edgecolor=SURF, lw=0.5)
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=2)
            ax.text(i + 0.33, np.median(v), f'median {np.median(v):.0f}', va='center', fontsize=8.5, color=INK2)
        ax.set_xticks(range(len(MODELS)), [SHORT[m] for m in MODELS]); ax.set_ylabel('false-positive blobs per CT'); ax.set_title(t)
    fig.suptitle('False-positive tumour components (>= 50 mm^3) per test CT', fontsize=11, fontweight='bold')
    fig.tight_layout()
    save(fig, out, '05_false_positives_per_ct')


def fig_fp_location(df, out):
    d = df[(df.region == 'whole_ct') & (df.variant == 'raw')]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.38
    for i, m in enumerate(MODELS):
        dm = d[d.model == m]
        a, b = dm.pred_les_within_10mm.mean(), dm.pred_les_beyond_48mm.mean()
        ax.bar(i - w / 2, a, w, color=COL[m], edgecolor=SURF, lw=2)
        ax.bar(i + w / 2, b, w, color=COL[m], alpha=0.45, edgecolor=SURF, lw=2, hatch='//')
        ax.text(i - w / 2, a, f'{a:.0%}', ha='center', va='bottom', fontsize=8.5, color=INK2)
        ax.text(i + w / 2, b, f'{b:.0%}', ha='center', va='bottom', fontsize=8.5, color=INK2)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=MUTED, label='within 10 mm of the pancreas'),
               Patch(facecolor=MUTED, alpha=0.45, hatch='//', label='more than 48 mm away (outside the training field of view)')]
    ax.set_xticks(range(len(MODELS)), [SHORT[m] for m in MODELS]); ax.set_ylim(0, 1.25); ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel('share of predicted tumour voxels')
    ax.set_title('Where the predicted tumour voxels are (whole CT)'); ax.legend(handles=handles, fontsize=8.5, loc='upper left', ncol=1)
    save(fig, out, '06_where_predictions_are')


def fig_paired(pr, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.0), sharex=True)
    lo, hi = pr.ci_low.min(), pr.ci_high.max()
    for ax, ref, others in zip(axes, ['baseline', 'control'], [['control', 'volume', 'volume_ball'], ['volume', 'volume_ball']]):
        d = pr[(pr.reference == ref) & pr.model.isin(others)].reset_index(drop=True)
        for i, r in d.iterrows():
            ax.plot([r.ci_low, r.ci_high], [i, i], color=COL[r.model], lw=2)
            ax.plot(r.mean_diff, i, 'o', color=COL[r.model], ms=8, mec=SURF, mew=2)
            p = 'p < 0.01' if r.wilcoxon_p < 0.01 else f'p = {r.wilcoxon_p:.2f}'
            ax.text(r.ci_high + 0.06 * (hi - lo), i, f'{r.mean_diff:+.3f}   better in {r.wins}, worse in {r.losses} CTs, {p}',
                    va='center', fontsize=8.5, color=INK2)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(range(len(d)), [SHORT[m] for m in d.model]); ax.invert_yaxis(); ax.set_ylim(len(d) - 0.5, -0.5)
        ax.set_title(f'minus {ref}'); ax.set_xlabel('change in lesion Dice (mean, 95 % bootstrap CI)')
    axes[0].set_xlim(lo - 0.1 * (hi - lo), hi + 1.6 * (hi - lo))
    fig.suptitle('Paired per-case differences in lesion Dice, pancreas ROI (same 50 test CTs; Wilcoxon signed-rank p)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    save(fig, out, '07_paired_differences')


def fig_size(les, out):
    d = les[les.region == 'pancreas_roi'].copy()
    edges = np.quantile(d[d.model == 'baseline'].lesion_vox, [0, 1 / 3, 2 / 3, 1])
    names = [f'small\n< {edges[1]/1000:.1f} cm^3', f'medium\n{edges[1]/1000:.1f}-{edges[2]/1000:.1f} cm^3', f'large\n> {edges[2]/1000:.1f} cm^3']
    d['bin'] = pd.cut(d.lesion_vox, [0, edges[1], edges[2], np.inf], labels=[0, 1, 2], right=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.2
    for i, m in enumerate(MODELS):
        g = d[d.model == m].groupby('bin', observed=False).detected.agg(['mean', 'sum', 'count'])
        x = np.arange(3) + (i - (len(MODELS) - 1) / 2) * w
        ax.bar(x, g['mean'], w * 0.92, color=COL[m], label=LABEL[m], edgecolor=SURF, lw=1)
        for xx, (_, r) in zip(x, g.iterrows()):
            ax.text(xx, r['mean'], f"{int(r['sum'])}/{int(r['count'])}", ha='center', va='bottom', fontsize=7.5, color=INK2)
    ax.set_xticks(range(3), names); ax.set_ylim(0, 1.12); ax.set_ylabel('lesion recall (detected / all)')
    ax.set_title('Detection by tumour size (pancreas ROI)'); ax.legend(fontsize=8, loc='lower right')
    save(fig, out, '08_detection_by_size')


def fig_reports(data, out):
    ra = pd.read_csv(f'{data}/meta/report_analysis.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    for sg, c in zip([0.0, 0.1, 0.2, 0.3], ['#2a78d6', '#1baf7a', '#eda100', '#e34948']):
        r = (ra[ra.sigma == sg].report_volume_mm3 / ra[ra.sigma == sg].true_volume_mm3).clip(0, 4)
        ax.hist(r, bins=np.linspace(0, 3, 41), histtype='step', lw=2, color=c, label=f'sigma = {sg:.2f}')
    ax.axvspan(0.9, 1.1, color=GRID, zorder=0, label='+-10 % (Volume Loss tolerance)')
    ax.set_xlabel('report volume / true volume'); ax.set_ylabel('tumours'); ax.legend(fontsize=8.5)
    ax.set_title('Synthetic reports: implied volume vs true volume')
    ax = axes[1]
    r0 = ra[ra.sigma == 0.0]
    ax.scatter(r0.true_volume_mm3 / 1000, r0.report_volume_mm3 / 1000, s=10, color='#2a78d6', alpha=0.6)
    lim = max(r0.true_volume_mm3.max(), r0.report_volume_mm3.max()) / 1000
    ax.plot([0, lim], [0, lim], color=INK, lw=1); ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('true tumour volume (cm^3)'); ax.set_ylabel('ellipsoid from report (cm^3)')
    ax.set_title('sigma = 0: ellipsoid estimate alone')
    fig.tight_layout()
    save(fig, out, '09_report_volume_accuracy')


def fig_example(run, data, df, out):
    """One test CT: GT vs baseline / volume_ball prediction, axial slice through the tumour centre, with the ROI box."""
    cands = [os.path.basename(p)[:-4] for p in glob.glob(f'{data}/eval_npz/BDMAP_*[0-9].npz')
             if os.path.exists(f'{run}/results/test/baseline/{os.path.basename(p)}')]
    if not cands:
        return
    bd = sorted(cands)[0]
    img = np.load(f'{data}/eval_npz/{bd}.npz')['arr_0']
    gt = np.load(f'{data}/eval_npz/{bd}_gt.npz')['arr_0']
    z = int(np.argmax(gt[4].sum((1, 2)))) if gt[4].any() else img.shape[0] // 2
    roi = roi_slices(gt[0] > 0, ROI_MARGIN, gt[0].shape)
    models = [m for m in ['baseline', 'volume_ball'] if os.path.exists(f'{run}/results/test/{m}/{bd}.npz')]
    fig, axes = plt.subplots(1, 1 + len(models), figsize=(4.2 * (1 + len(models)), 4.4))
    def show(ax, les, panc, t):
        ax.imshow(img[z], cmap='gray', vmin=-1.2, vmax=1.8)
        ax.contour(panc[z], levels=[0.5], colors=['#1baf7a'], linewidths=1.2)
        ax.contour(les[z], levels=[0.5], colors=['#e34948'], linewidths=1.2)
        ax.add_patch(plt.Rectangle((roi[2].start, roi[1].start), roi[2].stop - roi[2].start, roi[1].stop - roi[1].start,
                                   fill=False, ec='#eda100', lw=1.2, ls='--'))
        ax.set_title(t, fontsize=10); ax.axis('off')
    show(axes[0], gt[4] > 0, gt[0] > 0, f'{bd}: ground truth')
    for ax, m in zip(axes[1:], models):
        pr = np.load(f'{run}/results/test/{m}/{bd}.npz')['pred']
        dsc = df[(df.model == m) & (df.case == bd) & (df.region == 'whole_ct') & (df.variant == 'raw')].dice_lesion.iloc[0]
        show(ax, pr[4] > 0, pr[0] > 0, f'{m} prediction (lesion Dice {dsc:.2f})')
    fig.text(0.5, 0.02, 'green = pancreas, red = tumour, dashed = pancreas ROI (training field of view)', ha='center', fontsize=9, color=INK2)
    save(fig, out, '10_example_slice')

# ------------------------------------------------------------------------------------------- main
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--data'); ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--rescore', action='store_true')
    a = ap.parse_args()
    run = a.run.rstrip('/\\')
    data = (a.data or f'{run}/rsuper_data').rstrip('/\\')
    MODELS[:] = [m for m in MODELS if os.path.isdir(f'{run}/results/test/{m}')]
    out_a, out_f = f'{run}/analysis', f'{run}/figures'
    os.makedirs(out_a, exist_ok=True); os.makedirs(out_f, exist_ok=True)
    cache = f'{out_a}/per_case_metrics.csv'
    if os.path.exists(cache) and not a.rescore:
        df, les = pd.read_csv(cache), pd.read_csv(f'{out_a}/per_lesion.csv')
    else:
        df, les = score_run(run, data, a.workers)
        df.to_csv(cache, index=False); les.to_csv(f'{out_a}/per_lesion.csv', index=False)
    s = summarize(df); s.to_csv(f'{out_a}/summary.csv', index=False)
    pr = pd.concat([paired(df, metric=m) for m in ['dice_lesion', 'dice_pancreas']])
    pr.to_csv(f'{out_a}/paired_differences.csv', index=False)
    pd.set_option('display.width', 250)
    print(s[s.variant == 'raw'][['region', 'model', 'dice_lesion_mean', 'dice_pancreas', 'lesion_recall', 'lesion_precision',
                                 'lesion_F1', 'fp_per_ct', 'pred_vol_median', 'gt_vol_median']].to_string(index=False))
    print(pr.round(4).to_string(index=False))
    h = histories(run)
    fig_training(h, run, out_f); fig_report_losses(h, out_f); fig_time(run, out_f)
    fig_metrics(s, out_f); fig_fp(df, out_f); fig_fp_location(df, out_f); fig_paired(pr[pr.metric == 'dice_lesion'], out_f)
    fig_size(les, out_f); fig_reports(data, out_f); fig_example(run, data, df, out_f)
