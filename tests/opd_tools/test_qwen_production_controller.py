import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from opd_tools.qwen_production_controller import (
    compare_checkpoints, validate_iteration, verify_admission,
    verify_continuation, ProductionController, write_json, requires_semantic_checkpoint,
    verify_gpu_allocation, verify_gpu_evidence, validate_measurement, verify_hard_long_replay,
)
from opd_tools.manifest import file_sha256
from opd_tools.qwen_acceptance import PHYSICAL_RESOURCE_POLICY, validate_fsdp_probe, validate_physical_memory


def iteration(arm='softgrpo_math_opd_s11', index=0, phase='full_dose'):
    arm = 'softgrpo_math_s11' if phase == 'zero_dose' else arm
    standalone = arm == 'softopd_math_s11'
    enabled = standalone or arm.startswith('softgrpo_math_opd')
    ema = enabled and arm != 'softgrpo_math_opd_current_s11'
    base = .1 if arm.endswith('beta0p1_s11') else 1
    beta = base if enabled and (phase == 'full_dose' or standalone) else base * min(1, index / 11) if enabled else 0
    return {'rollout_iteration': index, 'trajectory_count': 64 if standalone else 512,
            'metrics': {'trainer/rollout_iteration': index, 'trainer/optimizer_steps_this_iteration': 2,
                        'trainer/optimizer_step': 2*(index+1), 'grad/total_norm': 1.6,
                        'actor/gradient_clipfrac': 1, 'integrity/continuous_replay_active': int(arm != 'hardgrpo_math_s11'),
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


def physical_iteration(arm='softgrpo_math_opd_s11', index=0, phase='full_dose'):
    record = iteration(arm, index, phase)
    timing = record['actor_update_timing']
    scope = 'update_actor entry through policy completion; excludes rollout and final offload'
    for rank in timing['ranks']:
        total, used = 80 * 1024**3, (74 + rank['rank']) * 1024**3
        rank['physical_memory'] = {
            'source': 'cuda_mem_get_info', 'scope': scope,
            'sampling': 'start, periodic, final; observed peak may miss sub-interval spikes',
            'sample_interval_seconds': .1, 'sample_count': 6,
            'device_total_bytes': total, 'device_used_peak_bytes': used,
            'device_free_min_bytes': total - used, 'start_free_bytes': 20 * 1024**3,
            'final_free_bytes': 6 * 1024**3, 'observed_seconds': .42,
            'host_ram_scope': 'whole-node utilization is diagnostic; Slurm enforces the job memory allocation'}
    timing.update(resource_policy=copy.deepcopy(PHYSICAL_RESOURCE_POLICY), physical_memory_scope=scope,
                  logical_allocator_peaks_diagnostic_only=True, physical_device_used_peak_gib=77.,
                  physical_device_free_min_gib=3., physical_device_used_fraction_peak=77/80)
    return record


def measured_phase(row, phase):
    suffix = 'resume' if phase in ('split', 'resume') else phase
    indices = {'production': [0], 'uninterrupted': [0, 1], 'split': [0], 'resume': [1],
               'full_dose': [0], 'zero_dose': [0]}[phase]
    return {'phase': phase, 'arm_id': row['arm_id'], 'status': 'complete',
            'wandb_run_id': row['wandb_run_id'] + ('' if phase == 'production' else '-' + suffix),
            'wandb_online': True, 'wandb_finished': True,
            'configuration': {key: {'resource_policy': copy.deepcopy(PHYSICAL_RESOURCE_POLICY)}
                              for key in ('trainer', 'actor_rollout_ref')},
            'acceptance_policy': {'resource_policy': copy.deepcopy(PHYSICAL_RESOURCE_POLICY)},
            'iterations': [physical_iteration(row['arm_id'], index, phase) for index in indices]}


def fsdp_acceptance(world_size=4):
    return {'schema_version': 1, 'status': 'passed', 'world_size': world_size, 'ranks': [
        {'rank': rank, 'optimizer_steps': 2, 'dense_ema_updates': 1, 'wrapper_count': 3,
         'frozen_base_unchanged': True, 'base_gradients_absent': True,
         'adapter_gradients_finite': True, 'adapter_update_nonzero': True,
         'disabled_reference_exact': True, 'dense_export_exact': True,
         'current_actor_detached': True} for rank in range(world_size)]}


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
    row = {'run_root': str(tmp_path), 'arm_id': 'softgrpo_math_opd_s11', 'wandb_run_id': 'test-run',
           'production_overrides': ['++trainer.resource_policy=' + json.dumps(PHYSICAL_RESOURCE_POLICY)],
           'phases': {phase: {'output': str(tmp_path / phase / 'measurement.json')}
                      for phase in ('production', 'uninterrupted', 'split', 'resume', 'full_dose', 'zero_dose')}}
    for phase in row['phases']:
        write_json(row['phases'][phase]['output'], measured_phase(row, phase))
    devices = [{'name': 'NVIDIA H100 80GB HBM3', 'total_memory_bytes': 80 * 1024**3} for _ in range(4)]
    result = {'status': 'passed', 'job_id': job_id, 'restart_count': restart,
              'arm_id': row['arm_id'], 'manifest_sha256': file_sha256(manifest_path), 'devices': devices,
              'runtime': {'unified': unified, 'native_lora_fsdp_acceptance': fsdp_acceptance(), 'native_fa3_acceptance': [{
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
                 'zero_dose_parity': {'passed': True, 'schema': 'qwen_semantic_v1'},
                 'gpu_allocation': allocation, 'evidence_files': {allocation['path']: allocation['sha256']}}
    admission['evidence_files'].update({str(Path(details['output']).relative_to(tmp_path)): file_sha256(details['output'])
                                       for phase, details in row['phases'].items() if phase != 'production'})
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
def test_invocation_project_and_import_path_match_manifest_and_phase(tmp_path, monkeypatch, phase, revised):
    from opd_tools.qwen_production import PRODUCTION_PROJECT
    arm = 'hardgrpo_math_s11'
    directory = tmp_path / phase
    output = directory / 'measurement.json'
    controller = ProductionController.__new__(ProductionController)
    controller.args = SimpleNamespace(arm=arm, signal_file=tmp_path / 'signal')
    source_root = tmp_path / 'source'
    source_verl = source_root / '3rdparty/SofT-GRPO/verl-0.4.x'
    installed = tmp_path / 'environment/site-packages'
    for root, marker in ((source_verl, 'source'), (installed, 'wheel')):
        (root / 'verl').mkdir(parents=True)
        (root / 'verl/__init__.py').write_text(f'identity = {marker!r}\n')
    controller.manifest = {'source_root': str(source_root)}
    controller.row = {'arm_id': arm, 'wandb_run_id': 'test-run', 'phases': {
        phase: {'directory': str(directory), 'output': str(output)}}}
    if revised:
        controller.row['wandb_project'] = PRODUCTION_PROJECT + '-lora-fa3'
        controller.manifest['profile_id'] = 'qwen3-math-seven-arm-lora-fa3-v1'
    project = controller.row.get('wandb_project', PRODUCTION_PROJECT) + ('' if phase == 'production' else '-prologue')
    controller.restart = 0
    controller.deadline = time.monotonic() + 600
    controller.report = {'submission_manifest_sha256': 'a' * 64, 'phases': {}}
    controller.persist = lambda: None
    controller.terminate_child = lambda: None
    captured = {}
    original_popen = subprocess.Popen

    def launch(command, **kwargs):
        captured.update(kwargs['env'])
        captured['working_directory'] = kwargs['cwd']
        assert kwargs['cwd'].is_dir()  # Created before spawning the trainer.
        # A fresh interpreter verifies actual Python cwd precedence, without
        # importing Torch or relying on this test process's imported modules.
        probe = original_popen([sys.executable, '-c', 'import verl; print(verl.identity)'],
                               cwd=kwargs['cwd'], env={**os.environ, 'PYTHONPATH': str(installed)},
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = probe.communicate(timeout=10)
        assert probe.returncode == 0, stderr
        assert stdout.strip() == ('wheel' if revised else 'source')
        write_json(output, measured_phase(controller.row, phase))
        return SimpleNamespace(wait=lambda timeout: 0)

    monkeypatch.setattr('opd_tools.qwen_production.phase_command', lambda *args, **kwargs: ['python', '-m', 'trainer',
        '++trainer.resource_policy=' + json.dumps(PHYSICAL_RESOURCE_POLICY)])
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
    assert captured['working_directory'] == (directory if revised else source_verl)
    assert invocation['working_directory'] == str(captured['working_directory'])


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
        write_json(output, measured_phase(row, phase)); controller.report['phases'][phase] = {}
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
    measured = measured_phase(row, 'production')
    measured['iterations'] = [physical_iteration(row['arm_id'], 24, 'production')]
    output = Path(row['phases']['production']['output'])
    write_json(output, measured)
    relative, digest = str(output.relative_to(tmp_path)), file_sha256(output)
    continuation['production_measurement'] = {'path': relative, 'sha256': digest}
    continuation['evidence_files'][relative] = digest
    write_json(tmp_path / 'continuation.json', continuation, seal=True)
    monkeypatch.setattr('opd_tools.qwen_production_controller.authenticate_checkpoint', lambda *args, **kw: {'reason': 'requeue_signal'})
    assert verify_continuation(manifest_path, manifest, row)['global_step'] == 25
    del continuation['gpu_allocation']
    write_json(tmp_path / 'continuation.json', continuation, seal=True)
    with pytest.raises(ValueError, match='sealed evidence'):
        verify_continuation(manifest_path, manifest, row)


@pytest.mark.parametrize('phase', ['split', 'production'])
@pytest.mark.parametrize('field,value', [('integrity/continuous_replay_active', 1),
                                       ('replay/ratio_abs_error_max', 1.001e-4),
                                       ('replay/ratio_abs_error_max', float('nan'))])
def test_hard_categorical_replay_checked_in_prologue_and_production(phase, field, value):
    record = iteration('hardgrpo_math_s11', phase=phase)
    record['metrics'][field] = value
    with pytest.raises(ValueError, match='replay'):
        validate_iteration(record, arm_id='hardgrpo_math_s11', phase=phase)


@pytest.mark.parametrize('world_size', [2, 4])
@pytest.mark.parametrize('mutation', ['missing', 'world_size', 'duplicate', 'rank_bool', 'rank_failure', 'cadence_bool'])
def test_shared_fsdp_probe_rejects_missing_or_wrong_rank_evidence(world_size, mutation):
    value = fsdp_acceptance(world_size)
    assert validate_fsdp_probe(value, world_size=world_size) == value
    if mutation == 'missing': value['ranks'].pop()
    elif mutation == 'world_size': value['world_size'] = 4 if world_size == 2 else 2
    elif mutation == 'duplicate': value['ranks'][-1]['rank'] = 0
    elif mutation == 'rank_bool': value['ranks'][0]['rank'] = False
    elif mutation == 'rank_failure': value['ranks'][-1]['dense_export_exact'] = False
    elif mutation == 'cadence_bool': value['ranks'][-1]['dense_ema_updates'] = True
    with pytest.raises(ValueError, match='LoRA/FSDP'):
        validate_fsdp_probe(value, world_size=world_size)


@pytest.mark.parametrize('mutation', ['missing_fsdp', 'two_ranks', 'fourth_rank_failed'])
def test_production_allocation_requires_real_four_rank_fsdp(tmp_path, mutation):
    path, manifest, row, certificate_path, certificate = gpu_allocation(tmp_path)
    runtime = certificate['runtime']
    if mutation == 'missing_fsdp': del runtime['native_lora_fsdp_acceptance']
    elif mutation == 'two_ranks': runtime['native_lora_fsdp_acceptance'] = fsdp_acceptance(2)
    else: runtime['native_lora_fsdp_acceptance']['ranks'][3]['frozen_base_unchanged'] = False
    write_json(certificate_path, certificate)
    with pytest.raises(ValueError, match='LoRA/FSDP'):
        verify_gpu_allocation(path, manifest, row, job_id='470999', restart_count=0)


@pytest.mark.parametrize('phase', ['production', 'uninterrupted', 'split', 'resume', 'full_dose', 'zero_dose'])
@pytest.mark.parametrize('mutation', ['missing_rank', 'duplicate_rank', 'missing_policy', 'fraction', 'aggregate', 'scope'])
def test_every_phase_independently_checks_four_rank_physical_memory(tmp_path, phase, mutation):
    _, manifest, row, _, _ = gpu_allocation(tmp_path)
    measured = measured_phase(row, phase)
    validate_measurement(measured, manifest, row, phase, row['production_overrides'])
    timing = measured['iterations'][-1]['actor_update_timing']
    if mutation == 'missing_rank': del timing['ranks'][3]['physical_memory']
    elif mutation == 'duplicate_rank': timing['ranks'][3]['rank'] = 2
    elif mutation == 'missing_policy': del measured['configuration']['actor_rollout_ref']['resource_policy']
    elif mutation == 'fraction':
        observation = timing['ranks'][3]['physical_memory']
        observation.update(device_total_bytes=100, device_used_peak_bytes=98, device_free_min_bytes=2,
                           start_free_bytes=20, final_free_bytes=6)
    elif mutation == 'aggregate': timing['physical_device_used_fraction_peak'] = .8
    else: timing['physical_memory_scope'] = 'rollout only'
    with pytest.raises(ValueError):
        validate_measurement(measured, manifest, row, phase, row['production_overrides'])


def test_production_cannot_infer_or_downgrade_missing_sealed_policy(tmp_path):
    _, manifest, row, _, _ = gpu_allocation(tmp_path)
    measured = measured_phase(row, 'production')
    with pytest.raises(ValueError, match='missing sealed'):
        validate_measurement(measured, manifest, row, 'production', [])
    arguments = ['++trainer.resource_policy=' + json.dumps({**PHYSICAL_RESOURCE_POLICY, 'max_device_used_fraction': 1})]
    with pytest.raises(ValueError, match='unsupported sealed'):
        validate_measurement(measured, manifest, row, 'production', arguments)


def test_prologue_budget_is_read_from_sealed_manifest(tmp_path, monkeypatch):
    path, manifest, row, _, _ = gpu_allocation(tmp_path)
    manifest.update(prologue_limit_seconds=10800, arms=[row])
    monkeypatch.setattr('opd_tools.qwen_production.verify_manifest', lambda path: manifest)
    monkeypatch.setenv('SLURM_JOB_ID', '470999')
    monkeypatch.setenv('SLURM_RESTART_COUNT', '0')
    monkeypatch.delenv('OPD_PROLOGUE_STARTED_EPOCH', raising=False)
    args = SimpleNamespace(manifest=path, arm=row['arm_id'], prologue_limit_seconds=None)
    controller = ProductionController(args)
    assert 10799 < controller.deadline - controller.started <= 10800
    args.prologue_limit_seconds = 7200
    with pytest.raises(ValueError, match='budget differs'):
        ProductionController(args)


def long_replay_fixture(tmp_path):
    source, runtime, assets = [tmp_path / name for name in ('source', 'runtime', 'assets')]
    for path in (source / 'scripts', runtime, assets): path.mkdir(parents=True)
    (source / 'scripts/qwen_runtime.py').write_text('# sealed runtime verifier\n')
    (source / 'scripts/qwen_vllm_long_replay_diagnostic.py').write_text(
        'import json\ndef validate_report(path):\n    result = json.loads(path.read_text())["acceptance"]\n'
        '    if not result["candidate_passed"]: raise ValueError("candidate failed")\n    return result\n')
    (runtime / 'opd-runtime-manifest.json').write_text('{}')
    (assets / 'manifest.json').write_text('{}')
    manifest_path = tmp_path / 'manifest.json'; manifest_path.write_text('{}')
    manifest = {'source_root': str(source), 'assets_root': str(assets),
                'parent_commit': 'a' * 40, 'fork_commit': 'b' * 40}
    row = {'run_root': str(tmp_path / 'run'), 'environment_root': str(runtime), 'arm_id': 'hardgrpo_math_s11'}
    path = Path(row['run_root']) / 'segments/preflight-42-1/categorical-long-replay/measurement.json'
    acceptance = {'candidate_passed': True, 'candidate_sampled_max_ratio_error': 1e-6}
    report = {'runtime': {'root': str(runtime), 'manifest_sha256': file_sha256(runtime / 'opd-runtime-manifest.json'),
                         'source': {key: manifest[key] for key in ('parent_commit', 'fork_commit')},
                         'verifier_sha256': file_sha256(source / 'scripts/qwen_runtime.py')},
              'assets_manifest_sha256': file_sha256(assets / 'manifest.json'), 'acceptance': acceptance,
              'job_id': '42', 'time_limit_seconds': 1170,
              'source_sha256': file_sha256(source / 'scripts/qwen_vllm_long_replay_diagnostic.py')}
    write_json(path, report)
    log = path.with_name('probe.log'); log.write_text('diagnostic completed\n')
    evidence = {'path': str(path.relative_to(row['run_root'])), 'sha256': file_sha256(path),
                'time_limit_seconds': 1170, 'acceptance': copy.deepcopy(acceptance),
                'log_sha256': file_sha256(log),
                'probe_sha256': file_sha256(source / 'scripts/qwen_vllm_long_replay_diagnostic.py'),
                'binding': {'job_id': '42', 'restart_count': 1, 'arm_id': row['arm_id'],
                            'manifest_sha256': file_sha256(manifest_path),
                            **{key: manifest[key] for key in ('parent_commit', 'fork_commit')}}}
    record = {'job_id': '42', 'restart_count': 1, 'runtime': {'hard_long_replay_acceptance': evidence}}
    return manifest_path, manifest, row, record, path, report


def test_long_replay_is_bound_to_source_allocation_runtime_and_assets(tmp_path):
    path, manifest, row, record, _, _ = long_replay_fixture(tmp_path)
    verify_hard_long_replay(path, manifest, row, record)


@pytest.mark.parametrize('mutation', ['missing', 'job_id', 'restart_count', 'arm_id', 'manifest_sha256',
    'parent_commit', 'fork_commit', 'absolute_path', 'escape', 'hash', 'runtime', 'runtime_source',
    'assets', 'verifier', 'candidate', 'acceptance', 'budget', 'log_hash', 'log_missing', 'log_symlink',
    'probe_hash', 'report_source', 'report_job', 'report_budget', 'budget_too_short'])
def test_long_replay_rejects_reused_tampered_and_failed_evidence(tmp_path, mutation):
    path, manifest, row, record, output, report = long_replay_fixture(tmp_path)
    evidence = record['runtime']['hard_long_replay_acceptance']
    if mutation == 'missing': del record['runtime']['hard_long_replay_acceptance']
    elif mutation in evidence['binding']: evidence['binding'][mutation] = 'changed'
    elif mutation == 'absolute_path': evidence['path'] = str(output)
    elif mutation == 'escape': evidence['path'] = '../measurement.json'
    elif mutation == 'hash': output.write_text(output.read_text() + '\n')
    elif mutation == 'runtime': report['runtime']['root'] = '/other/runtime'
    elif mutation == 'runtime_source': report['runtime']['source']['fork_commit'] = 'f' * 40
    elif mutation == 'assets': report['assets_manifest_sha256'] = 'f' * 64
    elif mutation == 'verifier': report['runtime']['verifier_sha256'] = 'f' * 64
    elif mutation == 'candidate': report['acceptance']['candidate_passed'] = False
    elif mutation == 'acceptance': evidence['acceptance']['candidate_sampled_max_ratio_error'] = 0
    elif mutation == 'budget': evidence['time_limit_seconds'] = 1171
    elif mutation == 'budget_too_short': evidence['time_limit_seconds'] = report['time_limit_seconds'] = 59
    elif mutation == 'log_hash': output.with_name('probe.log').write_text('changed')
    elif mutation == 'log_missing': output.with_name('probe.log').unlink()
    elif mutation == 'log_symlink':
        log = output.with_name('probe.log'); log.unlink()
        elsewhere = tmp_path / 'other.log'; elsewhere.write_text('diagnostic completed\n')
        log.symlink_to(elsewhere)
    elif mutation == 'probe_hash': evidence['probe_sha256'] = 'f' * 64
    elif mutation == 'report_source': report['source_sha256'] = 'f' * 64
    elif mutation == 'report_job': report['job_id'] = '43'
    elif mutation == 'report_budget': report['time_limit_seconds'] = 1169
    if mutation in ('runtime', 'runtime_source', 'assets', 'verifier', 'candidate',
                     'report_source', 'report_job', 'report_budget', 'budget_too_short'):

        write_json(output, report); evidence['sha256'] = file_sha256(output)
    with pytest.raises(ValueError):
        verify_hard_long_replay(path, manifest, row, record)


@pytest.mark.parametrize('mutation', [None, 'missing_site', 'gpu_name', 'capability', 'missing_capability',
                                     'partition', 'qos', 'account', 'constraint'])
def test_local_allocation_checks_sealed_h200_site_and_scheduler(tmp_path, monkeypatch, mutation):
    from opd_tools.qwen_site import resolve_site
    monkeypatch.setattr('opd_tools.qwen_site.shared_storage_root', lambda: tmp_path)
    root = tmp_path / 'artifacts'; root.mkdir()
    path, manifest, row, certificate_path, certificate = gpu_allocation(root)
    site = resolve_site('mbzuai-h200', artifact_root=root)
    manifest['site'] = site
    row['account'] = 'k2m'
    certificate.update(site=copy.deepcopy(site), scheduler={
        'Partition': 'main', 'QOS': 'k2m', 'Account': 'k2m', 'Features': 'nvidia_h200'})
    for device, acceptance in zip(certificate['devices'], certificate['runtime']['native_fa3_acceptance']):
        device.update(name='NVIDIA H200', compute_capability=[9, 0])
        acceptance['name'] = device['name']
    if mutation == 'missing_site': del certificate['site']
    elif mutation == 'gpu_name': certificate['devices'][3]['name'] = 'NVIDIA H100'
    elif mutation == 'capability': certificate['devices'][3]['compute_capability'] = [8, 0]
    elif mutation == 'missing_capability': del certificate['devices'][3]['compute_capability']
    elif mutation == 'partition': certificate['scheduler']['Partition'] = 'batch'
    elif mutation == 'qos': certificate['scheduler']['QOS'] = 'medium'
    elif mutation == 'account': certificate['scheduler']['Account'] = 'other'
    elif mutation == 'constraint': certificate['scheduler']['Features'] = 'nvidia_h200|nvidia_h100'
    write_json(certificate_path, certificate)
    if mutation is None:
        assert verify_gpu_allocation(path, manifest, row, job_id='470999', restart_count=0)['device_count'] == 4
    else:
        with pytest.raises(ValueError):
            verify_gpu_allocation(path, manifest, row, job_id='470999', restart_count=0)


def test_resealed_admission_cannot_hide_missing_physical_rank_evidence(tmp_path):
    path, manifest, row, certificate_path, _ = gpu_allocation(tmp_path)
    allocation = verify_gpu_allocation(path, manifest, row, job_id='470999', restart_count=0)
    evidence = {str(Path(details['output']).relative_to(tmp_path)): file_sha256(details['output'])
                for phase, details in row['phases'].items() if phase != 'production'}
    evidence[allocation['path']] = file_sha256(certificate_path)
    admission = {'status': 'passed', 'arm_id': row['arm_id'], 'job_id': '470999',
                 'submission_manifest_sha256': file_sha256(path),
                 'parent_commit': manifest['parent_commit'], 'fork_commit': manifest['fork_commit'],
                 'resume_parity': {'passed': True, 'schema': 'qwen_semantic_v1'},
                 'zero_dose_parity': {'passed': True, 'schema': 'qwen_semantic_v1'},
                 'gpu_allocation': allocation, 'evidence_files': evidence}
    write_json(tmp_path / 'admission.json', admission, seal=True)
    verify_admission(path, manifest, row)
    output = Path(row['phases']['full_dose']['output'])
    measured = json.loads(output.read_text())
    del measured['iterations'][0]['actor_update_timing']['ranks'][3]['physical_memory']
    write_json(output, measured)
    admission['evidence_files'][str(output.relative_to(tmp_path))] = file_sha256(output)
    write_json(tmp_path / 'admission.json', admission, seal=True)
    with pytest.raises(ValueError, match='missing physical memory'):
        verify_admission(path, manifest, row)
