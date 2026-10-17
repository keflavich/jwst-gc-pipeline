#!/bin/bash
#SBATCH --job-name=eval-reffield-grouped-fit-grp0-vs-grp1
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default,hpg-milan,hpg-turin
#SBATCH --cpus-per-task=1
#SBATCH --mem=24gb
#SBATCH --time=8:00:00
#SBATCH --output=<scratch>/grouptest/logs/%x_%j.log
set -x
S=<scratch>
export PYTHONPATH=$S/wt-grouptest
O=$S/grouptest/out
cd $O
EV="python -m jwst_gc_pipeline.photometry.reference_fields.evaluate"
$EV --variant grp0 --json $O/reffield_grp0.json >| $O/eval_grp0.txt 2>&1
$EV --variant grp1 --baseline $O/reffield_grp0.json --json $O/reffield_grp1_vs_grp0.json >| $O/eval_grp1_vs_grp0.txt 2>&1
python $S/grouptest/ref_nn_oversub.py grp0 grp1 $O/nn_grp0_grp1 >| $O/nn_grp0_grp1.txt 2>&1
SC=$S/wt-grouptest/docs/evidence/vetting_hysteresis/scripts
SPECS=""
for fb in superdense:F212N superdense:F405N dense_bright:F212N dense_bright:F187N dark:F182M dark:F212N bright_modest:F187N bright_modest:F210M; do
  for s in 0 1 2 3 4 5 6 7 8 9 10; do SPECS="$SPECS $fb:$s:0"; done
done
python $SC/ab_gallery_m7.py $O/ab_counts_grp0_grp1.png grp0 grp1 $SPECS >| $O/ab_counts_grp0_grp1.txt 2>&1
python $SC/ab_gallery_m7.py $O/ab_counts_grp1_grp0.png grp1 grp0 $SPECS >| $O/ab_counts_grp1_grp0.txt 2>&1
python $SC/ab_gallery_m7.py $O/ab_grp0_grp1_superdense.png grp0 grp1 superdense:F212N:0:3 superdense:F405N:0:3 >| $O/ab_grp0_grp1_superdense.txt 2>&1
python $SC/ab_gallery_m7.py $O/ab_grp1_grp0_superdense.png grp1 grp0 superdense:F212N:0:3 superdense:F405N:0:3 >| $O/ab_grp1_grp0_superdense.txt 2>&1
echo EVAL GRP DONE
