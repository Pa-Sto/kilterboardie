"""Compare deployed decoders, not incomparable architecture-specific losses."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from cvae_model import KilterCVAE
from cvae_generate import decode_route_with_priors
from diffusion_model import KilterDiffusionUNet, GaussianDiffusion
from graph_transformer_data import KilterGraphSequenceDataset, events_to_route_matrix
from graph_transformer_model import HierarchicalGraphTransformer
from graph_transformer_generate import generate_events


def route_scores(routes, xy, spacing):
    counts = routes.sum(1)
    valid_counts = ((counts[:, :2] >= 1) & (counts[:, :2] <= 2)).all(1)
    exclusive = (routes.sum(2) <= 1).all(1)
    reach = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1) / spacing <= 10
    connected = []
    for route in routes:
        usable = route[:, :3].any(1)
        visited = route[:, 0].copy()
        while True:
            new = visited | ((reach & visited[:, None]).any(0) & usable)
            if np.array_equal(new, visited):
                break
            visited = new
        connected.append(bool(route[:, 0].any() and route[:, 1].any() and visited[route[:, 1]].all()))
    return dict(start_finish_valid_rate=float(valid_counts.mean()),
                exclusive_roles_rate=float(exclusive.mean()),
                finish_reachable_rate=float(np.mean(connected)),
                unique_fraction=len({r.tobytes() for r in routes}) / len(routes),
                mean_counts=counts.mean(0).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--samples-per-grade', type=int, default=20)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--models', nargs='+', default=['cvae', 'diffusion', 'graph'],
                        choices=['cvae', 'diffusion', 'graph'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root
    manifest = json.loads((root / 'split.json').read_text())
    data = Path(manifest['data_dir'])
    ds = KilterGraphSequenceDataset(str(data), max_sequence_length=64)
    graph = ds.board_graph.to(torch.device(args.device))
    rows = graph.node_rows.cpu().numpy()
    cols = graph.node_cols.cpu().numpy()
    xy = graph.raw_xy.cpu().numpy()
    reference = np.load(ds.samples[0].npy_path)
    static_np = reference[..., 4:]
    static = torch.tensor(static_np.transpose(2, 0, 1)[None], device=args.device)
    hist = defaultdict(lambda: {k: Counter() for k in ('start', 'finish', 'hand', 'foot')})
    test = defaultdict(list)
    training_layouts = set()
    for partition in ('train', 'test'):
        for stem in manifest['splits'][partition]:
            a = np.load(data / f'{stem}.npy')[rows, cols, :4] > 0
            grade = int(json.loads((data / f'{stem}.json').read_text())['grade_v'])
            if partition == 'train':
                training_layouts.add(a.tobytes())
                for ch, key in enumerate(hist[grade]):
                    hist[grade][key][int(a[:, ch].sum())] += 1
            else:
                test[grade].append(a)
    test = {g: np.array(a) for g, a in test.items()}
    gen_args = SimpleNamespace(temperature=.9, top_k=24, greedy=False, start_min=1,
        start_max=2, finish_min=1, finish_max=2, min_body_holds=4, max_body_holds=20,
        max_group_size=4, pair_max_distance=8.)
    results = {}
    examples = {}
    for name in args.models:
        checkpoints = list((root / name).glob('*/best.pt'))
        if len(checkpoints) != 1:
            raise ValueError(f'Expected one checkpoint for {name}, got {checkpoints}')
        path = checkpoints[0]
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        if name == 'cvae':
            model = KilterCVAE(11, static_channels=6, emb_dim=cfg['emb_dim'], latent_dim=cfg['latent_dim'])
        elif name == 'diffusion':
            model = KilterDiffusionUNet(11, 6, base_channels=cfg['base_channels'],
                grade_emb_dim=cfg['grade_emb_dim'], time_emb_dim=cfg['time_emb_dim'])
            diffusion = GaussianDiffusion(timesteps=cfg['timesteps'], beta_start=cfg['beta_start'], beta_end=cfg['beta_end'])
        else:
            model = HierarchicalGraphTransformer(11, graph.node_feature_dim, graph.edge_feature_dim,
                **{k: cfg[k] for k in ('hidden_dim', 'graph_layers', 'transformer_layers',
                    'attention_heads', 'feedforward_dim', 'dropout', 'max_sequence_length')})
        model.load_state_dict(ckpt['model_state'])
        model.to(args.device).eval()

        @torch.no_grad()
        def generate(grade, seed):
            torch.manual_seed(seed)
            g = torch.tensor([grade-3], device=args.device)
            if name == 'graph':
                return events_to_route_matrix(generate_events(model, graph, grade-3, gen_args), graph, static_np)
            if name == 'cvae':
                probs = model.sample(g, static).sigmoid()
            else:
                x = diffusion.sample(model=model, shape=(1, 4, *reference.shape[:2]),
                    static=static, grade=g, device=torch.device(args.device), hold_mask=static[:, :1])
                probs = ((x+1)*.5).clamp(0, 1)
            route = decode_route_with_priors(probs, static[:, :1], g+3, hist, seed,
                .5, 1, 2, 1, 2, 8., 8., foot_count_mode='median')
            return np.concatenate([route[0].cpu().numpy().transpose(1, 2, 0), static_np], axis=2)

        generate(6, 0)  # Exclude warm-up and model loading from UI latency.
        generated, grades, durations, matrices = [], [], [], []
        for grade in range(3, 14):
            for i in range(args.samples_per_grade):
                started = time.perf_counter()
                a = generate(grade, 10000 + grade*100 + i)
                durations.append(time.perf_counter()-started)
                matrices.append(a)
                generated.append(a[rows, cols, :4] > 0)
                grades.append(grade)
        generated, grades = np.array(generated), np.array(grades)
        examples[name] = np.array(matrices)
        np.savez_compressed(root / f'{name}_generated.npz', matrices=examples[name], grades=grades)
        scores = route_scores(generated, xy, graph.coordinate_spacing)
        scores['exact_training_copy_rate'] = float(np.mean([r.tobytes() in training_layouts for r in generated]))
        scores['latency_median_seconds'] = float(np.median(durations))
        scores['latency_p95_seconds'] = float(np.percentile(durations, 95))
        scores['parameters'] = sum(p.numel() for p in model.parameters())
        scores['checkpoint'] = str(path)
        scores['best_epoch'] = ckpt.get('epoch')
        per_grade = {}
        for grade in range(3, 14):
            a, b = generated[grades == grade], test[grade]
            per_grade[grade] = dict(generated_counts=a.sum(1).mean(0).tolist(),
                test_counts=b.sum(1).mean(0).tolist(),
                count_mean_absolute_error=float(np.abs(a.sum(1).mean(0)-b.sum(1).mean(0)).mean()),
                hold_role_frequency_mae=float(np.abs(a.mean(0)-b.mean(0)).mean()))
        scores['per_grade'] = per_grade
        scores['grade_macro_count_mae'] = float(np.mean([s['count_mean_absolute_error'] for s in per_grade.values()]))
        scores['grade_macro_hold_role_frequency_mae'] = float(np.mean([s['hold_role_frequency_mae'] for s in per_grade.values()]))
        metrics = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text().splitlines()]
        scores['training_epoch_seconds'] = (sum(m['epoch_seconds'] for m in metrics)
            if all('epoch_seconds' in m for m in metrics) else None)
        results[name] = scores
        (root / 'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
        print(name, json.dumps({k: v for k, v in scores.items() if k != 'per_grade'}), flush=True)
        del model
        if args.device == 'mps':
            torch.mps.empty_cache()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 4.3*len(examples)), squeeze=False)
    colors = ['#24ad54', '#e84957', '#159dcc', '#e5ab23']
    for r, name in enumerate(examples):
        for c, grade in enumerate((3, 6, 9, 12)):
            ax = axes[r, c]
            ax.scatter(xy[:, 0], xy[:, 1], s=3, color='#d4d4d4')
            a = examples[name][(grade-3)*args.samples_per_grade][rows, cols, :4]
            for ch, color in enumerate(colors):
                selected = a[:, ch] > 0
                ax.scatter(xy[selected, 0], xy[selected, 1], s=38, facecolors='none', edgecolors=color, linewidths=1.8)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_title(f'{name} / V{grade}')
            ax.axis('off')
    fig.suptitle('Start: green | finish: red | hands: blue | feet: yellow')
    fig.tight_layout()
    fig.savefig(root / 'routes.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(results), figsize=(4.7*len(results), 4), squeeze=False)
    for ax, name in zip(axes[0], results):
        path = Path(results[name]['checkpoint']).parent / 'metrics.jsonl'
        metrics = [json.loads(line) for line in path.read_text().splitlines()]
        for split in ('train', 'val'):
            values = [m.get(f'{split}_loss', m.get(split, {}).get('loss')) for m in metrics]
            ax.plot(range(1, len(values)+1), values, label=split)
        ax.set_title(name + ' (own loss scale)')
        ax.set_xlabel('Epoch')
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'training.png', dpi=160)
    lines = ['# Three-model comparison', '',
        'All models: 30 epochs; same occupancy-grouped train/validation/test split.',
        'Generation uses existing constrained decoders. Count priors use training data only.',
        'Scores measure structural/distribution similarity, NOT true climbing difficulty or climbability.',
        'Latency includes single-route sampling and decoding, excludes model loading; Mac GPU.', '',
        '| Model | Count MAE (lower) | Valid start/finish | Exclusive roles | Reachable finish | Median / p95 seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for name, s in results.items():
        lines.append(f"| {name} | {s['grade_macro_count_mae']:.3f} | {s['start_finish_valid_rate']:.1%} | {s['exclusive_roles_rate']:.1%} | {s['finish_reachable_rate']:.1%} | {s['latency_median_seconds']:.3f} / {s['latency_p95_seconds']:.3f} |")
    lines += ['', 'Reach uses calibrated image coordinates and a 10-unit Euclidean threshold, excluding feet.',
        'CVAE/diffusion count matching is partly imposed by priors; validity is not evidence of learned rules.',
        'Transformer sequences are geometric pseudo-orderings, not observed climbing moves.',
        'One seed per model; 20 samples per grade by default. Treat rankings as preliminary.',
        '', '![Training](training.png)', '![Routes](routes.png)']
    (root / 'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
