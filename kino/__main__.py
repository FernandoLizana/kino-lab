from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import (
    DEFAULT_OUTPUT,
    build_local_dataset,
    frequency_table,
    frame_to_draws,
    load_canonical,
    load_historico,
    load_kino_csv,
    merge_draws,
    save_draws,
)
from .remote import KinoHistoricClient
from .simulate import run_streaming_simulation


def cmd_build(_: argparse.Namespace) -> int:
    build_local_dataset()
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    local = []
    if not args.remote_only:
        local = merge_draws(load_historico(), load_kino_csv())
        print(f"Local fusionado: {len(local)} sorteos")

    client = KinoHistoricClient(delay_seconds=args.delay)
    if args.latest_only:
        remote = [client.fetch_latest()]
        print(
            f"Último sorteo remoto: #{remote[0].draw_number} {remote[0].draw_date} "
            f"{remote[0].sorted_numbers()}"
        )
        if DEFAULT_OUTPUT.exists():
            existing = frame_to_draws(load_canonical())
            draws = merge_draws(existing, local, remote)
        else:
            draws = merge_draws(local, remote)
    else:
        remote = client.fetch_all()
        draws = merge_draws(local, remote)

    path = save_draws(draws, Path(args.output))
    print(f"Dataset sincronizado: {len(draws)} sorteos -> {path}")
    if draws:
        print(f"Rango: {draws[0].draw_date} .. {draws[-1].draw_date}")
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    frame = load_canonical()
    print(f"Sorteos: {len(frame)}")
    print(f"Rango: {frame['draw_date'].iloc[0]} .. {frame['draw_date'].iloc[-1]}")
    if "source" in frame.columns:
        print("Fuentes:")
        print(frame["source"].value_counts().to_string())
    freq = frequency_table(frame)
    print("Frecuencia por número:")
    print(freq.to_string())
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    run_streaming_simulation(
        n_simulations=args.n,
        batch_size=args.batch_size,
        top=args.top,
        seed=args.seed,
        weighted=args.weighted,
        dataset_path=Path(args.dataset),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kino",
        description=(
            "Dataset Kino unificado + sync desde kinohistorico.cl "
            "+ Monte Carlo en streaming (sin volcar simulaciones al disco)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Fusiona historico.csv + kino.csv")
    p_build.set_defaults(func=cmd_build)

    p_sync = sub.add_parser(
        "sync",
        help="Complementa el dataset local con la API de kinohistorico.cl",
    )
    p_sync.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Ruta del CSV canónico",
    )
    p_sync.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Pausa entre requests (segundos)",
    )
    p_sync.add_argument(
        "--latest-only",
        action="store_true",
        help="Solo trae el último sorteo y lo fusiona",
    )
    p_sync.add_argument(
        "--remote-only",
        action="store_true",
        help="Ignora CSVs locales y usa solo la API",
    )
    p_sync.set_defaults(func=cmd_sync)

    p_stats = sub.add_parser("stats", help="Resumen del dataset canónico")
    p_stats.set_defaults(func=cmd_stats)

    p_sim = sub.add_parser(
        "simulate",
        help="Monte Carlo en streaming (solo guarda el top N)",
    )
    p_sim.add_argument("--n", type=int, default=1_000_000, help="Simulaciones")
    p_sim.add_argument(
        "--batch-size",
        type=int,
        default=20_000,
        help="Tamaño de lote en RAM",
    )
    p_sim.add_argument("--top", type=int, default=100, help="Combinaciones a exportar")
    p_sim.add_argument("--seed", type=int, default=42, help="Semilla RNG")
    p_sim.add_argument(
        "--weighted",
        action="store_true",
        help="Pondera por frecuencias históricas (exploratorio)",
    )
    p_sim.add_argument(
        "--dataset",
        default=str(DEFAULT_OUTPUT),
        help="CSV canónico para modo weighted",
    )
    p_sim.set_defaults(func=cmd_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
