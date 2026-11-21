"""Shared capture guard and failure diagnostics. No torch import on module import."""
import math
import time


class MemoryLimitExceeded(RuntimeError):
    def __init__(self, phase, snapshot, limit_gib):
        super().__init__(f'Reserved GPU peak exceeded {limit_gib:g} GiB during {phase}')
        self.phase, self.snapshot, self.limit_gib = phase, snapshot, limit_gib


class MemoryGuard:
    """Check completed intervals, not allocations inside a running CUDA kernel."""
    def __init__(self, cuda, limit_gib=21.):
        if not math.isfinite(limit_gib) or not 0 < limit_gib <= 21:
            raise ValueError('Safety limit must be positive and at most 21 GiB')
        self.cuda, self.limit_gib = cuda, float(limit_gib)
        self.last = {}
        self.last_valid = None

    def snapshot(self):
        try:
            result={'allocated_bytes':self.cuda.memory_allocated(),
                    'reserved_bytes':self.cuda.memory_reserved(),
                    'peak_allocated_bytes':self.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':self.cuda.max_memory_reserved()}
            self.last_valid=result
            return result
        except Exception as exc:
            return {'allocated_bytes':None,'reserved_bytes':None,'peak_allocated_bytes':None,
                    'peak_reserved_bytes':None,'measurement_error':type(exc).__name__,
                    'last_successful_measurement':self.last_valid}

    def begin(self):
        self.cuda.reset_peak_memory_stats()
        self.start = self.snapshot()
        self.check('before_generation')

    def check(self, phase):
        self.last = self.snapshot()
        if self.last.get('measurement_error'):
            raise RuntimeError('GPU memory measurement failed; stop capture')
        if self.last['peak_reserved_bytes'] > self.limit_gib * 2**30:
            raise MemoryLimitExceeded(phase,self.last,self.limit_gib)


def failure_kind(exc):
    if isinstance(exc,MemoryLimitExceeded):return 'memory_limit_exceeded'
    if type(exc).__name__=='StopRequested':return 'interrupted'
    text=str(exc).lower()
    if 'memory measurement failed' in text:return 'memory_measurement_failed'
    if type(exc).__name__=='OutOfMemoryError' or 'out of memory' in text:return 'oom'
    if isinstance(exc,OSError):return 'io_error'
    if 'truncated' in text:return 'generation_truncated'
    if 'context exceeded' in text:return 'context_exceeded'
    if any(s in text for s in ('subquestions','processed token','alignment mismatch')):return 'parse_failure'
    if 'feature' in text:return 'invalid_features'
    return 'runtime_error'


def validate_features(cap, layers, hidden_size=4096):
    import numpy as np
    n=len(cap['steps'])
    if not 2<=n<=5 or len(cap['features'])!=n or len(cap['feature_positions'])!=n:
        raise ValueError('Feature/step count mismatch')
    if cap['feature_positions']!=sorted(set(cap['feature_positions'])):
        raise ValueError('Feature token positions are not strictly ordered')
    for feature in cap['features']:
        if set(feature)!={str(layer) for layer in layers}:raise ValueError('Feature layer mismatch')
        for vector in feature.values():
            array=np.asarray(vector)
            if array.shape!=(hidden_size,) or not np.isfinite(array).all():
                raise ValueError('Feature dimension or finite-value check failed')


def require_device(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one visible GPU required')
    name=torch.cuda.get_device_name(0)
    if '3090' not in name:raise RuntimeError('Require RTX 3090 before model loading')
    return name


def guarded_generate(model, tok, messages, guard, layers, seed=42, max_new_tokens=1024,
                     hidden_size=4096, capture=True):
    from .capture import generate
    from experiment_state import StopRequested
    trace={}; cap=None; error=None; status='running';started=time.perf_counter();validation_seconds=0.
    try:
        guard.begin()
        cap=generate(model,tok,messages,layers=layers,max_new_tokens=max_new_tokens,
            max_context=8192,temperature=.1,top_p=.9,seed=seed,capture=capture,
            diagnostics=trace,memory_check=guard.check)
        trace['phase']='feature_validation'
        validation_started=time.perf_counter()
        try:
            if capture:validate_features(cap,layers,hidden_size)
        finally:
            validation_seconds=time.perf_counter()-validation_started
        cap['metrics']['feature_validation_seconds']=validation_seconds
        guard.check('after_feature_validation')
        status='success'
    except (Exception,StopRequested) as exc:
        status=failure_kind(exc)
        error={'type':type(exc).__name__,'message':str(exc)[:1500],
               'phase':getattr(exc,'phase',trace.get('phase','before_generation'))}
        cap=None
    elapsed=time.perf_counter()-started
    tokens=trace.get('generated_ids',[])
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos or [])
    ended=bool(tokens and tokens[-1] in eos)
    memory=guard.snapshot()
    if memory.get('measurement_error') and status=='success':
        status='memory_measurement_failed';cap=None
        error={'type':'MemoryMeasurementError','message':'Final memory measurement failed','phase':'final_snapshot'}
    return {'status':status,'error':error,'result':cap,'diagnostics':trace,
        'start_memory':getattr(guard,'start',None),
        'parse_success':bool(trace.get('parse_success',False)),
        'features_valid':status=='success' and capture,
        'prompt_tokens':trace.get('prompt_tokens'),'completion_tokens':len(tokens),
        'finish_reason':'stop' if ended else 'length' if len(tokens)>=max_new_tokens else None,
        'feature_validation_seconds':validation_seconds,
        'extra_forward_passes':0,'generation_and_feature_wall_seconds':elapsed,
        'memory':memory,'memory_limit_gib':guard.limit_gib,
        'safe':status=='success' and memory['peak_reserved_bytes']<=guard.limit_gib*2**30,
        'stop_required':status in ('memory_limit_exceeded','oom','interrupted','memory_measurement_failed') or bool(memory.get('measurement_error')),
        'length_coverage':{'max_new_tokens':max_new_tokens,'observed_completion_tokens':len(tokens),
            'near_cap_90_percent':len(tokens)>=math.ceil(max_new_tokens*.9),
            'worst_case_verified':False,
            'note':'Only the observed trajectory is measured. Setting max_new_tokens does not validate a 1024-token worst case.'}}


def apply_attempt(doc, row, attempt, outcome):
    """Record failure before caller stops; successful rows are never retroactively edited."""
    if outcome.get('result') is not None:
        attempt['usage']=outcome['result']['usage']
    elif 'usage' not in attempt:
        attempt['usage']={'prompt_tokens':outcome.get('prompt_tokens'),
            'completion_tokens':outcome['completion_tokens'],
            'latency_seconds':outcome['diagnostics'].get('generation_seconds')}
    attempt.update(status=outcome['status'],memory=outcome['memory'],error=outcome['error'],
                   diagnostics=outcome['diagnostics'],completion_tokens=outcome['completion_tokens'],
                   length_coverage=outcome['length_coverage'],start_memory=outcome.get('start_memory'),
                   observations={key:outcome.get(key) for key in ('prompt_tokens','finish_reason',
                        'parse_success','features_valid','extra_forward_passes','generation_and_feature_wall_seconds')})
    row['memory']=outcome['memory']
    if outcome['status']!='success':
        row['status']='error';row['failure_kind']=outcome['status']
    else:
        row.pop('failure_kind',None)
    if outcome['stop_required']:
        doc['safety_stop']={'reason':outcome['status'],'problem_id':row['id'],
            'last_successful_problem_id':doc.get('last_successful_problem_id'),
            'attempt_number':len(row['attempts']),'memory':outcome['memory'],
            'error':outcome['error'],'requires_explicit_resume_acknowledgement':True}


def acknowledge_resume(doc, resume, acknowledged):
    if acknowledged and not resume:raise ValueError('Safety acknowledgement requires --resume')
    stop=doc.get('safety_stop')
    if stop:
        if not resume or not acknowledged:
            raise ValueError('Safety stop preserved; inspect it, then explicitly use --resume --resume-after-safety-stop')
        doc.setdefault('safety_stop_history',[]).append(stop)
        doc['safety_stop']=None
