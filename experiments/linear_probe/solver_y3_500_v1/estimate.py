"""Project workload using only measurements from the five-problem Pilot."""
import json

from .audit import CREDIT, OUT
from ..common import read, verify
from experiment_state import attempt_cost

Y1 = OUT.parent / 'solver_y1_v1'
N = 500
SCALE = N / 5


def estimate():
    solver = verify(read(Y1 / 'responses_solver_221051_main.json'))
    ledger = attempt_cost([row['stage'] for row in solver['records']])
    capture = verify(read(OUT.parent / 'pilot_v1/evaluation_recovered.json'))['records']
    teacher = verify(read(CREDIT / 'evaluation_labeled.json'))['records']
    diagnostic = verify(read(OUT.parent / 'analysis_credit_tp4_220316/extended_220319.json'))
    predictions = diagnostic['splits']['evaluation']['probe']['records']
    controls = [r['capture']['capture_control']['metrics'] for r in capture]
    features = [r['capture']['metrics'] for r in capture]
    sec = {name: SCALE * sum(m[key] for m in records) for name, records, key in (
        ('control_decomposition_wall', controls, 'generation_plus_feature_seconds'),
        ('capture_decomposition_wall', features, 'generation_plus_feature_seconds'),
        ('capture_validation', features, 'feature_validation_seconds'),
        ('hidden_retention_overlap', features, 'hidden_retention_seconds'),
        ('feature_alignment_overlap', features, 'alignment_seconds'),
        ('feature_transfer_overlap', features, 'feature_transfer_seconds'),
    )}
    for k in ('feature_read_seconds', 'inference_seconds', 'projection_seconds'):
        sec[f'probe_{k}'] = SCALE*sum(p['probe'][k] for p in predictions)
    sec['teacher_api'] = SCALE*attempt_cost([r['teacher_stage'] for r in teacher])['latency_seconds']['known_sum']
    sec['solver_api'] = SCALE*ledger['latency_seconds']['known_sum']
    sec['solver_session_wall'] = SCALE*solver['wall_seconds']
    sec['solver_startup_observed_once'] = verify(read(
        Y1 / 'ready_solver_221051_main.json'))['server_start_to_ready_seconds']
    sec['capture_model_load_observed_once'] = verify(read(
        OUT.parent/'pilot_v1/evaluation_recovered.json'))['sessions'][0]['model_load_seconds']
    total = sum(sec[k] for k in ('control_decomposition_wall','capture_decomposition_wall',
                'capture_validation','teacher_api','solver_session_wall','solver_startup_observed_once',
                'probe_feature_read_seconds','probe_inference_seconds','probe_projection_seconds'))
    total += 10*sec['capture_model_load_observed_once']
    return dict(scope='historical linear projection, teacher API from A2000 TP4',
                seconds=sec, wall_h_reference=total/3600, reserved_gpu_h_reference=4*total/3600,
                chunk_reference_wall_h=total/(10*3600), solver_api_3090_allocated_gpu_h=4*sec['solver_api']/3600,
                planner_single_gpu_stage_h_reference=(sec['control_decomposition_wall']+
                sec['capture_decomposition_wall']+sec['capture_validation'])/3600,
                teacher_api_4xA2000_gpu_h_reference=4*sec['teacher_api']/3600,
                solver_raw_json_pilot_scaled_bytes=SCALE*(Y1/'responses_solver_221051_main.json').stat().st_size,
                solver_server_log_pilot_scaled_bytes=SCALE*(Y1/'server_solver_221051_main.log').stat().st_size,
                selected_feature_float32_bytes_4_steps=N*4*4096*4)


if __name__ == '__main__':
    print(json.dumps(estimate(), indent=2))
