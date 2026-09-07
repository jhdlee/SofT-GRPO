import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from opd_tools.qwen_production_controller import (
    compare_checkpoints, validate_iteration, verify_admission,
    verify_continuation, ProductionController, write_json, requires_semantic_checkpoint,
    verify_gpu_allocation, verify_gpu_evidence,
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


def semantic_checkpoint():
    return {'rollout_trajectory_sha256': '1' * 64, 'actor_model_optimizer_tree_sha256': 'archive',
            'opd_teacher_tree_sha256': 'teacher-archive', 'semantic_identity': {
                'schema': 'qwen_semantic_v1',
                **{field: str(index) * 64 for index, field in enumerate((
                    'actor_model_optimizer_scheduler_sha256', 'worker_rng_sha256',
                    'driver_rng_sha256', 'dataloader_sha256', 'teacher_model_ema_sha256'), 2)}}}


def test_exact_semantic_comparison_accepts_archive_metadata_differences():
    left = semantic_checkpoint(); right = copy.deepcopy(left)
    right['actor_model_optimizer_tree_sha256'] = 'different-pickle-cache'
    right['opd_teacher_tree_sha256'] = 'different-teacher-pickle-cache'
    result = compare_checkpoints(left, right, include_teacher=True, require_semantic=True)
    assert result['passed'] and result['schema'] == 'qwen_semantic_v1'
    assert 'worker_rng_sha256' in result['compared_fields']


@pytest.mark.parametrize('field', ['actor_model_optimizer_scheduler_sha256', 'worker_rng_sha256',
                                  'driver_rng_sha256', 'dataloader_sha256', 'teacher_model_ema_sha256'])
def test_each_semantic_next_update_component_must_match(field):
    left = semantic_checkpoint(); right = copy.deepcopy(left)
    right['semantic_identity'][field] = 'f' * 64
    with pytest.raises(ValueError, match=field):
        compare_checkpoints(left, right, include_teacher=True)


def test_semantic_zero_dose_excludes_only_teacher():
    left = semantic_checkpoint(); right = copy.deepcopy(left)
    right['semantic_identity']['teacher_model_ema_sha256'] = None
    assert compare_checkpoints(left, right, include_teacher=False)['passed']
    right['semantic_identity']['worker_rng_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='worker_rng'):
        compare_checkpoints(left, right, include_teacher=False)


def test_semantic_requirements_fail_closed_without_weakening_legacy():
    checkpoint = semantic_checkpoint()
    old = {k: v for k, v in checkpoint.items() if k != 'semantic_identity'}
    assert compare_checkpoints(old, old, include_teacher=True)['passed']
    with pytest.raises(ValueError, match='schemas'):
        compare_checkpoints(old, old, include_teacher=True, require_semantic=True)
    with pytest.raises(ValueError, match='schemas'):
        compare_checkpoints(checkpoint, old, include_teacher=True)
    assert requires_semantic_checkpoint({'profile_id': 'qwen3-math-seven-arm-lora-fa3-v1'})
    assert not requires_semantic_checkpoint({'profile_id': 'qwen3-math-seven-arm-v1'})


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


def gpu_allocation(tmp_path, *, job_id='470999', restart=0):
    manifest_path = tmp_path / 'manifest.json'; manifest_path.write_text('{}')
    manifest = {'profile_id': 'qwen3-math-seven-arm-lora-fa3-v1', 'parent_commit': 'a' * 40, 'fork_commit': 'b' * 40}
    unified = write_json(tmp_path / 'runtime.json', {'build_record': {'source': {
        key: manifest[key] for key in ('parent_commit', 'fork_commit')}}}, seal=True)
    manifest['runtime_manifest'] = {'manifest_content_sha256': unified['manifest_content_sha256']}
    row = {'run_root': str(tmp_path), 'arm_id': 'softgrpo_math_opd_s11'}
    devices = [{'name': 'NVIDIA H100 80GB HBM3', 'total_memory_bytes': 80 * 1024**3} for _ in range(4)]
    result = {'status': 'passed', 'job_id': job_id, 'restart_count': restart,
              'arm_id': row['arm_id'], 'manifest_sha256': file_sha256(manifest_path), 'devices': devices,
              'runtime': {'unified': unified, 'native_fa3_acceptance': [{
                  'device': index, 'name': devices[index]['name'], 'native_fa3_forward_backward': True,
                  'fresh_process_exact_match': True,
                  'native_kernel': 'opd_fa3._C', 'dtype': 'bfloat16', 'head_dimension': 128,
                  'vllm_flash_attention_version': 3, 'vllm_kernel': 'vllm._vllm_fa3_C',
                  'packed_causal_gradient_isolation': True, 'long_packed_lengths': [8192, 8192],
                  'long_output_gradient_sha256': ['c' * 64] * 4,
                  'native_lora': {'schema_version': 1, 'status': 'passed', 'optimizer_steps': 2,
                      'frozen_base_unchanged': True, 'dense_export_weight_exact': True,
                      'dense_export_projection_exact': True, 'teacher_or_training_data_used': False,
                      'adapter_gradient_norms': [12.5, 28.0], 'effective_update': {'changed_elements': 128}},
              } for index in range(4)]}}
    path = tmp_path / f'segments/allocation-{job_id}-{restart}.json'
    path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(result))
    return manifest_path, manifest, row, path, result


def test_four_gpu_preflight_identity_and_controlled_lora_evidence(tmp_path):
    manifest_path, manifest, row, path, record = gpu_allocation(tmp_path)
    actual = verify_gpu_allocation(manifest_path, manifest, row, job_id='470999', restart_count=0)
    assert actual == {'path': str(path.relative_to(tmp_path)), 'sha256': file_sha256(path),
                      'job_id': '470999', 'restart_count': 0, 'device_count': 4}
    # Gradient magnitudes are diagnostic; no clipping/ratio range is imposed.
    assert record['runtime']['native_fa3_acceptance'][0]['native_lora']['adapter_gradient_norms'][0] > 10


@pytest.mark.parametrize('mutation', ['job', 'restart', 'manifest', 'arm', 'status', 'missing_gpu',
    'duplicate_gpu', 'forward_backward', 'causal', 'long_hash', 'kernel', 'lora', 'base_change',
    'effective_update', 'nonfinite_gradient', 'runtime_source', 'missing_acceptance', 'fresh_process'])
def test_preflight_rejects_missing_or_mismatched_gpu_evidence(tmp_path, mutation):
    manifest_path, manifest, row, path, record = gpu_allocation(tmp_path)
    acceptance = record['runtime']['native_fa3_acceptance']; first = acceptance[0]
    if mutation == 'job': record['job_id'] = '1'
    elif mutation == 'restart': record['restart_count'] = 1
    elif mutation == 'manifest': record['manifest_sha256'] = 'f' * 64
    elif mutation == 'arm': record['arm_id'] = 'softopd_math_s11'
    elif mutation == 'status': record['status'] = 'failed'
    elif mutation == 'missing_gpu': acceptance.pop()
    elif mutation == 'duplicate_gpu': acceptance[-1]['device'] = 0
    elif mutation == 'forward_backward': first['native_fa3_forward_backward'] = False
    elif mutation == 'causal': first['packed_causal_gradient_isolation'] = False
    elif mutation == 'long_hash': first['long_output_gradient_sha256'] = ['c' * 64]
    elif mutation == 'kernel': first['vllm_flash_attention_version'] = 2
    elif mutation == 'lora': first['native_lora']['status'] = 'failed'
    elif mutation == 'base_change': first['native_lora']['frozen_base_unchanged'] = False
    elif mutation == 'effective_update': first['native_lora']['effective_update']['changed_elements'] = 0
    elif mutation == 'nonfinite_gradient': first['native_lora']['adapter_gradient_norms'][0] = float('nan')
    elif mutation == 'runtime_source': record['runtime']['unified']['build_record']['source']['fork_commit'] = 'f' * 40
    elif mutation == 'missing_acceptance': del record['runtime']['native_fa3_acceptance']
    elif mutation == 'fresh_process': del first['fresh_process_exact_match']
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, RuntimeError)):
        verify_gpu_allocation(manifest_path, manifest, row, job_id='470999', restart_count=0)


def test_new_admission_seals_gpu_file_and_rechecks_it_for_continuation(tmp_path):
    manifest_path, manifest, row, path, _ = gpu_allocation(tmp_path)
    allocation = verify_gpu_allocation(manifest_path, manifest, row, job_id='470999', restart_count=0)
    admission = {'status': 'passed', 'arm_id': row['arm_id'], 'job_id': '470999',
                 'submission_manifest_sha256': file_sha256(manifest_path),
                 'parent_commit': manifest['parent_commit'], 'fork_commit': manifest['fork_commit'],
                 'resume_parity': {'passed': True, 'schema': 'qwen_semantic_v1'},
                 'gpu_allocation': allocation, 'evidence_files': {allocation['path']: allocation['sha256']}}
    write_json(tmp_path / 'admission.json', admission, seal=True)
    assert verify_admission(manifest_path, manifest, row)['gpu_allocation'] == allocation
    missing = copy.deepcopy(admission); missing['evidence_files'] = {}
    write_json(tmp_path / 'admission.json', missing, seal=True)
    with pytest.raises(ValueError, match='sealed evidence'):
        verify_admission(manifest_path, manifest, row)
    write_json(tmp_path / 'admission.json', admission, seal=True)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='evidence changed'):
        verify_admission(manifest_path, manifest, row)


def test_continuation_requires_its_own_allocation_not_initial_gpu_proof(tmp_path):
    manifest_path, manifest, row, path, _ = gpu_allocation(tmp_path, restart=1)
    allocation = verify_gpu_allocation(manifest_path, manifest, row, job_id='470999', restart_count=1)
    record = {'job_id': '470999', 'gpu_allocation': allocation,
              'evidence_files': {allocation['path']: allocation['sha256']}}
    verify_gpu_evidence(manifest_path, manifest, row, record, restart_count=1)
    with pytest.raises(ValueError):
        verify_gpu_evidence(manifest_path, manifest, row, record, restart_count=0)


def test_missing_allocation_prevents_initial_controller_invocation(tmp_path, monkeypatch):
    manifest_path, manifest, row, path, _ = gpu_allocation(tmp_path)
    manifest['arms'] = [{**row, 'wandb_run_id': 'test'}]
    path.unlink()
    monkeypatch.setenv('SLURM_JOB_ID', '470999'); monkeypatch.setenv('SLURM_RESTART_COUNT', '0')
    monkeypatch.setattr('opd_tools.qwen_production.verify_manifest', lambda path: manifest)
    args = SimpleNamespace(manifest=manifest_path, arm=row['arm_id'], prologue_limit_seconds=7200)
    with pytest.raises(ValueError, match='regular JSON'):
        ProductionController(args)


def test_legacy_study_does_not_require_new_gpu_preflight_schema(tmp_path):
    assert verify_gpu_allocation(tmp_path / 'missing', {}, {}, job_id=None, restart_count=None) is None


@pytest.mark.parametrize('phase', ['production', 'split'])
@pytest.mark.parametrize('revised', [False, True])
def test_invocation_wandb_project_matches_manifest_and_phase(tmp_path, monkeypatch, phase, revised):
    from opd_tools.qwen_production import PRODUCTION_PROJECT
    arm = 'hardgrpo_math_s11'
    directory = tmp_path / phase
    output = directory / 'measurement.json'
    controller = ProductionController.__new__(ProductionController)
    controller.args = SimpleNamespace(arm=arm, signal_file=tmp_path / 'signal')
    controller.manifest = {'source_root': str(tmp_path)}
    controller.row = {'wandb_run_id': 'test-run', 'phases': {
        phase: {'directory': str(directory), 'output': str(output)}}}
    if revised:
        controller.row['wandb_project'] = PRODUCTION_PROJECT + '-lora-fa3'
    project = controller.row.get('wandb_project', PRODUCTION_PROJECT) + ('' if phase == 'production' else '-prologue')
    controller.restart = 0
    controller.deadline = time.monotonic() + 600
    controller.report = {'submission_manifest_sha256': 'a' * 64, 'phases': {}}
    controller.persist = lambda: None
    controller.terminate_child = lambda: None
    captured = {}

    def launch(command, **kwargs):
        captured.update(kwargs['env'])
        write_json(output, {'phase': phase, 'arm_id': arm, 'status': 'complete',
                            'wandb_run_id': kwargs['env']['WANDB_RUN_ID'],
                            'wandb_online': True, 'wandb_finished': True,
                            'iterations': [] if phase == 'production' else [iteration(arm, phase=phase)]})
        return SimpleNamespace(wait=lambda timeout: 0)

    monkeypatch.setattr('opd_tools.qwen_production.phase_command', lambda *args, **kwargs: ['python', 'trainer'])
    monkeypatch.setattr('opd_tools.qwen_production_controller.subprocess.Popen', launch)
    monkeypatch.setattr('opd_tools.icl_resource_monitor.ResourceMonitor', lambda **kwargs: SimpleNamespace(
        start=lambda: None, stop=lambda: SimpleNamespace(to_dict=lambda: {})))
    controller.invoke(phase)
    suffix = '-0' if phase == 'production' else ''
    invocation = json.loads((directory / f'invocation-{phase}{suffix}.json').read_text())
    assert captured['WANDB_PROJECT'] == project
    assert invocation['wandb_project'] == project
    assert controller.report['phases'][phase]['wandb_project'] == project
    assert invocation['wandb_run_id'] == captured['WANDB_RUN_ID']


def test_real_admission_publishes_gpu_certificate_and_continuation_reuses_it(tmp_path, monkeypatch):
    manifest_path, manifest, row, path, _ = gpu_allocation(tmp_path)
    row['phases'] = {phase: {'output': str(tmp_path / phase / 'measurement.json'),
                             'run_dir': str(tmp_path / phase / 'training')}
                     for phase in ('uninterrupted', 'split', 'resume', 'full_dose', 'zero_dose', 'production')}
    controller = ProductionController.__new__(ProductionController)
    controller.args = SimpleNamespace(manifest=manifest_path, arm=row['arm_id'])
    controller.manifest = manifest; controller.row = row; controller.root = tmp_path; controller.restart = 0
    controller.started = time.monotonic(); controller.deadline = controller.started + 600
    controller.report = {'submission_manifest_sha256': file_sha256(manifest_path), 'job_id': '470999', 'phases': {}}
    def invoke(phase, **kwargs):
        output = Path(row['phases'][phase]['output']); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{}'); controller.report['phases'][phase] = {}
    controller.invoke = invoke
    monkeypatch.setattr('opd_tools.qwen_production_controller.authenticate_checkpoint', lambda *args, **kw: semantic_checkpoint())
    admission = controller.admission()
    allocation = admission['gpu_allocation']
    assert admission['evidence_files'][allocation['path']] == file_sha256(path)
    accepted = verify_admission(manifest_path, manifest, row)
    checkpoint_path = Path(row['phases']['production']['run_dir']) / 'global_step_25/checkpoint_manifest.json'
    checkpoint_path.parent.mkdir(parents=True); checkpoint_path.write_text('{}')
    continuation = {'arm_id': row['arm_id'], 'global_step': 25, 'job_id': '470999', 'restart_count': 0,
                    'submission_manifest_sha256': file_sha256(manifest_path),
                    'admission_sha256': accepted['manifest_content_sha256'],
                    'checkpoint_manifest_sha256': file_sha256(checkpoint_path),
                    'gpu_allocation': allocation, 'evidence_files': {allocation['path']: allocation['sha256']}}
    write_json(tmp_path / 'continuation.json', continuation, seal=True)
    monkeypatch.setattr('opd_tools.qwen_production_controller.authenticate_checkpoint', lambda *args, **kw: {'reason': 'requeue_signal'})
    assert verify_continuation(manifest_path, manifest, row)['global_step'] == 25
    del continuation['gpu_allocation']
    write_json(tmp_path / 'continuation.json', continuation, seal=True)
    with pytest.raises(ValueError, match='sealed evidence'):
        verify_continuation(manifest_path, manifest, row)
