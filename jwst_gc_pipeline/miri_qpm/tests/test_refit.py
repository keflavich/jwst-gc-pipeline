"""Numerical recovery, identifiability, and calibration bundle selection."""
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import fftconvolve

from jwst_gc_pipeline.miri_qpm import calib, pipeline, refit, psf
from jwst_gc_pipeline.miri_qpm.workflow import sha256


def test_response_recovers_affine_in_presence_of_outliers():
    x = np.linspace(30, 450, 200)
    y = .6*x + 1.1
    y[::20] += 50
    a,b,good = refit.robust_line(x,y)
    np.testing.assert_allclose([a,b], [.6,1.1], atol=1e-10)
    assert good.sum() >= 150 and not good[::20].any()
    with pytest.raises(refit.InsufficientCalibrationData):
        refit.robust_line(np.ones(50), np.ones(50))


def test_flat_and_response_recover_forward_model():
    rng = np.random.default_rng(9)
    region = np.repeat(np.arange(1,4), 16*32).reshape(48,32)
    ref = SimpleNamespace(region=region, x0=0, y0=0)
    flat = rng.uniform(.9,1.1,region.shape)
    for g in (1,2,3):
        flat[region == g] /= np.median(flat[region == g])
    n = flat.size
    pix = np.tile(np.arange(n), 24)
    exp = np.repeat(np.arange(24), n)
    truth_a = np.array([np.nan,.6,.61,.25])
    truth_b = np.array([np.nan,1.0,1.2,.5])
    M = rng.uniform(30,450,len(pix))
    Y = flat.ravel()[pix]*(truth_a[region.ravel()[pix]]*M+truth_b[region.ravel()[pix]])
    samples = dict(pix=pix,exp=exp,field=exp//8,Y=Y,M=M)
    fitted,count = refit.fit_flat(samples,region,n_iter=12)
    np.testing.assert_allclose(fitted,flat,rtol=1e-4)
    law = refit.fit_law(refit.block_samples(samples,fitted,ref),nboot=20)
    for g in (1,2,3):
        np.testing.assert_allclose([law[str(g)]['a'],law[str(g)]['b']], [truth_a[g],truth_b[g]], atol=2e-3)
    assert count.min() >= 3


def test_kernel_recovers_forward_smear():
    yy,xx = np.mgrid[-12:13,-12:13]
    P = np.exp(-(xx**2+yy**2)/2)
    P /= P.sum()
    K = np.zeros((5,21)); K[2,10]=.75; K[2,7]=.25
    pane = fftconvolve(P,K,mode='same')
    recovered = refit.fit_kernel(P,pane)
    assert recovered.min() >= 0
    np.testing.assert_allclose(recovered,K,atol=1e-6)


def test_geometry_uses_voting_and_rejects_unconstrained_panes():
    rng = np.random.default_rng(2)
    candidates=[]
    for g in (1,2,3):
        for i in range(80):
            x,y = rng.uniform(10,230), rng.uniform(220*(g-1)+22,220*g)
            dx,dy = -3.5+.05*(x-500)/500, .2+.07*(y-500)/500
            candidates.append([i%8,g,x,y,dx,dy])
            candidates.append([i%8,g,x,y,*rng.uniform(-10,10,2)])
    models,stats = refit._geometry(np.array(candidates))
    assert all(stats[g]['n'] >= 80 for g in (1,2,3))
    np.testing.assert_allclose(models[1][0],[-3.5,.05,0],atol=1e-10)
    with pytest.raises(refit.InsufficientCalibrationData):
        refit._geometry(np.array(candidates)[:10])


def test_candidate_bundle_metadata_and_variance_follow_selected_bundle(tmp_path):
    src=calib.DATA/'F770W'
    shutil.copytree(src,tmp_path/'bundle')
    bundle=tmp_path/'bundle'
    path=bundle/'law_R1_F770W.json'
    law=json.loads(path.read_text())
    law.update(recipe='R1-refit',status='candidate',rl_variance_factors={'1':2,'2':3,'3':4},rl_iterations=30)
    law['panes']['1']['a']=.8
    path.write_text(json.dumps(law))
    sums=''.join(f'{sha256(bundle/name)}  {name}\n' for name in sorted(['qpm_R1_F770W.fits','law_R1_F770W.json','kernels_F770W.npz','geometry_F770W.json']))
    (bundle/'SHA256SUMS').write_text(sums)
    ref=calib.refdata(calibration_dir=str(bundle))
    assert ref.a[1] == .8
    from astropy.io import fits
    hdr=pipeline._common_header(fits.Header(),pipeline.DEFAULTS,ref)
    assert hdr['QPMRECIP'] == 'R1-refit' and hdr['QPMCALST'] == 'candidate'
    assert hdr['QPMCSUM'] == ref.checksum
    var=psf.scale_variance(np.ones((1,3)),np.array([[1,2,3]]),law['rl_variance_factors'])
    np.testing.assert_array_equal(var,[[2,3,4]])
    calib.refdata.cache_clear()
    path.write_text('{}')
    with pytest.raises(ValueError,match='checksum mismatch'):
        calib.refdata(calibration_dir=str(bundle))


def test_refit_refuses_in_sample_only_validation(tmp_path):
    prepared={'config':{'calibration':{'output_dir':str(tmp_path/'candidate')}},'fields':[]}
    with pytest.raises(refit.InsufficientCalibrationData,match='training fields'):
        refit.refit(prepared)
    assert not (tmp_path/'candidate').exists()


def test_missing_stars_is_explicit_failure():
    with pytest.raises(refit.InsufficientCalibrationData,match='star stamps'):
        refit.fit_psfs({g:[] for g in range(4)})
