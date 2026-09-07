import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opd_tools.qwen_production_controller import (
    compare_checkpoints, validate_iteration, verify_admission,
    verify_continuation, ProductionController, write_json,
)
from opd_tools.manifest import file_sha256


def iteration(arm='softgrpo_math_opd_s11', index=0, phase='full_dose'):
    standalone = arm == 'softopd_math_s11'
    enabled = standalone or arm.startswith('softgrpo_math_opd')
    ema = enabled and arm != 'softgrpo_math_opd_current_s11'
    base = .1 if arm.endswith('beta0p1_s11') else 1
    beta = base if enabled and (phase == 'full_dose' or standalone) else base * index / 11 if enabled else 0
    return {'rollout_iteration': index, 'trajectory_count': 64 if standalone else 512,
            'metrics': {'trainer/rollout_iteration': index, 'trainer/optimizer_steps_this_iteration': 2,
                        'trainer/optimizer_step': 2*(index+1), 'grad/total_norm': 1.6,
                        'actor/gradient_clipfrac': 1, 'integrity/continuous_replay_active': 1,
                        'replay/fallback_count': 0, 'replay/ratio_abs_error_max': 2e-6,
                        'latent/soft_to_hard_rate': .8, 'opd/beta_effective': beta,
                        'opd/ema_updates_this_iteration': int(ema),
                        'opd/ema_update_count': index+1 if ema else 0,
                        'opd/latent_slot_count': 100, 'opd/answer_slot_count': 20,
                        'opd/selected_slots': 120, 'grad/opd_norm': 1e-6, 'grad/grpo_norm': 1},
            'actor_update_timing': {'ranks': [{'rank': rank, 'optimizer_steps': 2,
                 'ema_updates_this_iteration': int(ema), 'ema_update_count': index+1 if ema else 0,
                 'worker_update_seconds': 10, 'policy_update_seconds': 8,
                 'max_memory_allocated_gib': 61, 'max_memory_reserved_gib': 75} for rank in range(4)]}}


@pytest.mark.parametrize('arm', ['hardgrpo_math_s11','softgrpo_math_s11','softopd_math_s11',
    'softgrpo_math_opd_s11','softgrpo_math_opd_posadv_s11','softgrpo_math_opd_current_s11','softgrpo_math_opd_beta0p1_s11'])
def test_four_rank_arm_acceptance(arm):
    record=iteration(arm)
    result=validate_iteration(record,arm_id=arm,phase='full_dose')
    assert result['accepted'] and not result['clipping_frequency_gate'] and not result['ratio_range_gate']


@pytest.mark.parametrize('field,value', [('replay/ratio_abs_error_max',1.001e-4),('grad/total_norm',float('nan')),
    ('opd/beta_effective',0),('trainer/optimizer_step',1),('opd/ema_updates_this_iteration',0),
    ('grad/opd_norm',0),('actor/gradient_clipfrac',1.1),('latent/soft_to_hard_rate',0)])
def test_full_dose_errors_remain_failures(field,value):
    record=iteration(); record['metrics'][field]=value
    with pytest.raises(ValueError): validate_iteration(record,arm_id='softgrpo_math_opd_s11',phase='full_dose')


def test_positive_advantage_empty_uses_selected_slots_not_denominator():
    record=iteration('softgrpo_math_opd_posadv_s11')
    record['metrics'].update({'opd/selected_slots':0,'grad/opd_norm':0})
    assert validate_iteration(record,arm_id='softgrpo_math_opd_posadv_s11',phase='full_dose')['accepted']
    with pytest.raises(ValueError): validate_iteration(record,arm_id='softgrpo_math_opd_s11',phase='full_dose')


def test_wrong_rank_and_group_counts_fail():
    record=iteration(); record['actor_update_timing']['ranks'][3]['rank']=2
    with pytest.raises(ValueError): validate_iteration(record,arm_id='softgrpo_math_opd_s11',phase='full_dose')
    record=iteration(); record['trajectory_count']=64
    with pytest.raises(ValueError): validate_iteration(record,arm_id='softgrpo_math_opd_s11',phase='full_dose')


def test_warmup_and_resume_cadence():
    record=iteration(index=1,phase='resume')
    assert validate_iteration(record,arm_id='softgrpo_math_opd_s11',phase='resume')['effective_beta']==1/11


def test_exact_resume_and_zero_dose_comparisons_differ_only_in_teacher_scope():
    left={'rollout_trajectory_sha256':'a','actor_model_optimizer_tree_sha256':'b','opd_teacher_tree_sha256':'c'}
    right={**left,'opd_teacher_tree_sha256':None}
    assert compare_checkpoints(left,right,include_teacher=False)['passed']
    with pytest.raises(ValueError,match='opd_teacher'): compare_checkpoints(left,right,include_teacher=True)
    right['actor_model_optimizer_tree_sha256']='different'
    with pytest.raises(ValueError,match='actor_model'): compare_checkpoints(left,right,include_teacher=False)


def test_admission_authenticates_evidence_not_only_status(tmp_path):
    manifest_path=tmp_path/'manifest.json'; manifest_path.write_text('{}')
    evidence=tmp_path/'measurement.json'; evidence.write_text('{}')
    manifest={'parent_commit':'a','fork_commit':'b'}; row={'run_root':str(tmp_path),'arm_id':'arm'}
    write_json(tmp_path/'admission.json',{'status':'passed','arm_id':'arm','submission_manifest_sha256':file_sha256(manifest_path),
        **manifest,'evidence_files':{'measurement.json':file_sha256(evidence)},'resume_parity':{'passed':True}},seal=True)
    assert verify_admission(manifest_path,manifest,row)['status']=='passed'
    evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='evidence changed'): verify_admission(manifest_path,manifest,row)


def test_admission_failure_prevents_production_invocation(tmp_path, monkeypatch):
    controller=ProductionController.__new__(ProductionController)
    controller.args=SimpleNamespace(signal_file=tmp_path/'signal',arm='softgrpo_math_opd_s11')
    controller.root=tmp_path; controller.child=None; controller.restart=0
    controller.report={}; controller.phase='uninterrupted'; controller.deadline=10**12
    controller.persist=lambda:None
    def fail(): raise ValueError('exact next-update parity failed')
    controller.admission=fail
    controller.invoke=lambda *a,**kw:pytest.fail('production must not run after failed admission')
    assert controller.run()==1
    assert controller.report['status']=='failed'


def test_cleanup_targets_child_process_group_even_when_leader_exited(monkeypatch):
    controller=ProductionController.__new__(ProductionController)
    calls=[]
    controller.child=SimpleNamespace(pid=1234,wait=lambda **kw:0)
    monkeypatch.setattr('os.killpg',lambda pid,sig:calls.append((pid,sig)))
    controller.terminate_child()
    assert len(calls)==2 and all(pid==1234 for pid,_ in calls) and controller.child is None
