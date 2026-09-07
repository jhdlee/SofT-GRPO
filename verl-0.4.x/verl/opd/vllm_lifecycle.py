"""Frozen TP1 hard-rollout lifecycle for the revised Qwen production profile."""


def poison_vllm(engine, reason):
    engine._opd_poisoned = str(reason)
    if getattr(engine, '_opd_shutdown_after_failure', False):
        return
    engine._opd_shutdown_after_failure = True
    # vLLM's external-launcher executor runs inside this Ray worker. Its base
    # shutdown may be a no-op: the fatal collective error must also reach the
    # production controller, which terminates the job's Ray process tree.
    executor = getattr(getattr(engine, 'llm_engine', None), 'model_executor', None)
    shutdown = getattr(executor, 'shutdown', None) or getattr(engine, 'shutdown', None)
    if shutdown is not None:
        try:
            shutdown()
        except Exception as failure:
            engine._opd_shutdown_error = f'{type(failure).__name__}: {failure}'


def require_idle_vllm(engine):
    if getattr(engine, '_opd_poisoned', None):
        raise RuntimeError(f'vLLM engine is poisoned and cannot be reused: {engine._opd_poisoned}')
    if getattr(engine, '_opd_batch_outstanding', False):
        raise RuntimeError('vLLM rollout batch remains outstanding')
    llm = getattr(engine, 'llm_engine', None)
    unfinished = getattr(llm, 'has_unfinished_requests', None)
    if unfinished is not None and unfinished():
        raise RuntimeError('vLLM scheduler requests remain unfinished')


def finish_vllm_stage(engine, distributed, error, stage):
    """One matched world collective, including healthy ranks, before transition."""
    message = None if error is None else f'{type(error).__name__}: {error}'[:2000]
    errors = [message]
    try:
        if distributed.is_initialized():
            errors = [None] * distributed.get_world_size()
            distributed.all_gather_object(errors, message)
    except BaseException as failure:
        poison_vllm(engine, f'vLLM {stage} communication failed: {failure}')
        raise
    failures = [f'rank {rank}: {value}' for rank, value in enumerate(errors) if value is not None]
    if failures:
        reason = f'vLLM collective {stage} failed: ' + '; '.join(failures)
        poison_vllm(engine, reason)
        raise RuntimeError(reason) from error
