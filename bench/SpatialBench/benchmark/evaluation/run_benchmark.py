"""Evaluate GFM3D / GFM4D using the bundled models."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import re
import sys
import time
import traceback

BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH_ROOT))
GFM_ROOT = next((p for p in BENCH_ROOT.parents if (p / 'gfm').is_dir()), None)

import numpy as np
import torch
import yaml

from benchmark.datasets.benchmark_dataset import BenchmarkDataset
from benchmark.evaluation.model_adapters import get_adapter
import benchmark.evaluation.model_adapters.gfm_adapter  # registers the two supported models
from benchmark.evaluation.metrics import compute_depth_metrics, compute_pose_metrics, compute_trajectory_metrics
from benchmark.evaluation.report import build_report, write_json_report, write_overall, print_summary

ALL_EVAL_METRICS = ('depth', 'pose', 'trajectory')


def evaluate_scene(scene, predictions, adapter, depth_alignment='median', eval_metrics=None):
    """Preserve reference scene-wide depth alignment and camera/trajectory protocols."""
    selected = set(ALL_EVAL_METRICS if eval_metrics is None else eval_metrics)
    result = dict(scene_id=scene['scene_id'], source_dataset=scene['source_dataset'],
                  tags=scene['tags'], num_frames=len(scene['frame_indices']))
    if 'depth' in selected:
        mask = scene['valid_mask']
        pred, gt = predictions['pred_depth'][mask], scene['depth'][mask]
        if len(gt):
            if depth_alignment == 'lstsq':
                design = np.stack([pred, np.ones_like(pred)], axis=1)
                scale, shift = np.linalg.lstsq(design, gt, rcond=None)[0]
                pred = pred * scale + shift
            elif depth_alignment == 'median':
                pred = pred * (np.median(gt) / (np.median(pred) + 1e-8))
            else:
                raise ValueError(f'Unknown depth_alignment: {depth_alignment}')
            result['depth'] = compute_depth_metrics(pred, gt, np.ones(len(gt), dtype=bool))
    if 'pose' in selected:
        result['camera'] = compute_pose_metrics(predictions['w2c_extrinsics'], adapter.normalize_gt_poses(scene))
    if 'trajectory' in selected:
        result['trajectory'] = compute_trajectory_metrics(predictions['pred_pose'], scene['extrinsic'])
    return result


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument('--config', help='GFM evaluation YAML')
    parser.add_argument('--model', choices=('gfm3d', 'gfm4d'))
    parser.add_argument('--checkpoint')
    parser.add_argument('--device')
    parser.add_argument('--scene-index')
    parser.add_argument('--benchmark-root', help='SpatialBenchmark dataset root')
    parser.add_argument('--scene-id')
    parser.add_argument('--tags')
    parser.add_argument('--max-scenes', type=int)
    parser.add_argument('--eval-metrics', nargs='+', choices=ALL_EVAL_METRICS)
    parser.add_argument('--depth-alignment', choices=('median', 'lstsq'))
    parser.add_argument('--conf-threshold', type=float)
    parser.add_argument('--shuffle-seed', type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--output')
    cli = vars(parser.parse_args(argv))
    config = {}
    if cli.get('config'):
        with open(cli['config']) as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            parser.error('The YAML must contain a mapping')
    config.update(cli)  # Explicit CLI values always win, even if equal to defaults.
    defaults = dict(device='cuda', scene_index='benchmark/scene_indices/all_scenes.json',
                    benchmark_root='../SpatialBenchmark', scene_id=None, tags='sparse', max_scenes=None,
                    eval_metrics=list(ALL_EVAL_METRICS), depth_alignment='median',
                    conf_threshold=.5, shuffle_seed=None, output_dir='results', output=None)
    for key,value in defaults.items():
        config.setdefault(key,value)
    if config.get('model') not in ('gfm3d','gfm4d'):
        parser.error('Only model: gfm3d or gfm4d is supported')
    if not config.get('checkpoint'):
        parser.error('A local checkpoint is required')
    if config['depth_alignment'] not in ('median','lstsq'):
        parser.error('depth_alignment must be median or lstsq')
    if not config['eval_metrics'] or set(config['eval_metrics']) - set(ALL_EVAL_METRICS):
        parser.error('eval_metrics supports only depth, pose, trajectory')
    if config.get('visualize',False):
        parser.error('Visualization is not part of this evaluation-only package')
    if config.get('resolution') is not None:
        parser.error('Use resolution_override for dataset resizing')
    for key in ('scene_index','benchmark_root','checkpoint'):
        path=Path(config[key]).expanduser()
        if not path.is_absolute():
            if key in cli:
                base=Path.cwd()
            elif key=='checkpoint':
                if GFM_ROOT is None:
                    parser.error('Use an absolute checkpoint path outside the bundled GFM tree')
                base=GFM_ROOT
            else:
                base=BENCH_ROOT
            path=base/path
        config[key]=str(path.resolve())
    # Resolve the output split after YAML and CLI overrides have been merged.
    # --output remains an explicit filename override; --output-dir is the base.
    tags = str(config.get('tags') or '').strip()
    tag_dir = re.sub(r'[^A-Za-z0-9_.-]+', '_',
                    tags.replace('|', '_OR_').replace('+', '_AND_')).strip('._') or 'all'
    config['output_dir'] = str(Path(config['output_dir']) / tag_dir)
    return config


def main(argv=None):
    cfg=parse_config(argv)
    common={'config','model','checkpoint','device','scene_index','benchmark_root','scene_id','tags',
            'max_scenes','eval_metrics','depth_alignment','conf_threshold','shuffle_seed','output_dir',
            'output','resolution','resolution_override','visualize','vis_conf_percent'}
    adapter=get_adapter(cfg['model'])
    adapter.configure(**{k:v for k,v in cfg.items() if k not in common})
    tag_filter=cfg['tags'];operator='OR' if tag_filter and '|' in tag_filter else 'AND'
    tags=tag_filter.replace('|','+').split('+') if tag_filter else None
    dataset=BenchmarkDataset(scene_index_path=cfg['scene_index'], benchmark_root=cfg['benchmark_root'],
        tags=None if cfg['scene_id'] else tags, tag_operator=operator,
        max_scenes=None if cfg['scene_id'] else cfg['max_scenes'],
        conf_threshold=cfg['conf_threshold'], resolution_override=cfg.get('resolution_override'),
        shuffle_seed=cfg['shuffle_seed'])
    if cfg['scene_id']:
        dataset.scenes=[s for s in dataset.scenes if s['scene_id']==cfg['scene_id']]
    if not dataset.scenes:
        raise ValueError('No scenes match the supplied scene index/filter')
    if not Path(cfg['benchmark_root']).is_dir():
        raise FileNotFoundError(f"Dataset root not found: {cfg['benchmark_root']}")
    print(f"Loading {adapter.name()} from {cfg['checkpoint']}")
    adapter.load_model(cfg['checkpoint'],device=cfg['device'])
    output=Path(cfg['output']) if cfg['output'] else Path(cfg['output_dir'])/f'{adapter.name()}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix(output.suffix+'.partial')
    stat=Path(cfg['checkpoint']).stat()
    signature=hashlib.sha256(json.dumps(dict(config=cfg, checkpoint_stat=[stat.st_size,stat.st_mtime_ns],
        scenes=[s['scene_id'] for s in dataset.scenes]),sort_keys=True).encode()).hexdigest()
    results=[]
    if partial.is_file():
        saved=json.loads(partial.read_text())
        if saved.get('signature')==signature:
            results=[r for r in saved['results'] if 'error' not in r]
    done={r['scene_id'] for r in results}
    for i,meta in enumerate(dataset.scenes):
        if meta['scene_id'] in done:
            continue
        scene=predictions=None
        try:
            scene=dataset[i]
            print(f"[{i+1}/{len(dataset)}] {scene['scene_id']}")
            if adapter.device.type=='cuda':torch.cuda.synchronize(adapter.device)
            start=time.perf_counter()
            predictions=adapter.predict(scene)
            if adapter.device.type=='cuda':torch.cuda.synchronize(adapter.device)
            elapsed=time.perf_counter()-start
            selected=cfg['eval_metrics']
            if scene['tags'].get('view_density')=='single':selected=[m for m in selected if m=='depth']
            result=evaluate_scene(scene,predictions,adapter,cfg['depth_alignment'],selected)
            result['timing']={'inference_seconds':elapsed}
        except Exception as error:
            traceback.print_exc()
            result=dict(scene_id=meta['scene_id'],source_dataset=meta['source_dataset'],
                        tags=meta.get('tags',{}),error=str(error))
        results.append(result)
        temporary=partial.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(signature=signature,results=results),indent=2))
        temporary.replace(partial)
        scene=predictions=None
        gc.collect()
        if adapter.device.type=='cuda':torch.cuda.empty_cache()
    report=build_report(adapter.name(),cfg['tags'],results,checkpoint=cfg['checkpoint'],model_params=adapter.get_model_params())
    report['meta']['config']=cfg
    report['meta']['num_failed']=sum('error' in r for r in results)
    write_json_report(report,str(output))
    write_overall(report,str(output.parent))
    print_summary(report)
    if report['meta']['num_failed']:
        raise SystemExit(f"{report['meta']['num_failed']} scene(s) failed; partial results kept for retry")
    partial.unlink(missing_ok=True)


if __name__=='__main__':
    main()
