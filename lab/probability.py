from __future__ import annotations

import math
from dataclasses import dataclass

from .profiles import GameProfile


class ProbabilityError(ValueError):
    pass


def comb(n: int, k: int) -> int:
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)


def hit_support(profile: GameProfile) -> tuple[int, int]:
    """Rango válido de aciertos j para extracción sin reposición."""
    low = max(0, profile.r + profile.k - profile.n)
    high = min(profile.r, profile.k)
    return low, high


def hypergeometric_pmf(profile: GameProfile) -> dict[int, float]:
    """
    P(X=j) = C(k,j) * C(N-k, r-j) / C(N, r)
    k = extraídos, r = boleto, N = universo.
    """
    if not profile.supports_exact_hypergeometric():
        raise ProbabilityError(
            "Este perfil tiene orden o reposición: no aplica la hipergeométrica. "
            "Usa simulación etiquetada."
        )
    n, k, r = profile.n, profile.k, profile.r
    denom = comb(n, r)
    if denom == 0:
        raise ProbabilityError("Combinaciones totales = 0; revisa N y r.")
    low, high = hit_support(profile)
    pmf: dict[int, float] = {}
    for j in range(low, high + 1):
        pmf[j] = comb(k, j) * comb(n - k, r - j) / denom
    return pmf


def expected_hits(profile: GameProfile) -> float:
    return profile.r * profile.k / profile.n


def jackpot_odds(profile: GameProfile) -> int:
    if profile.r != profile.k:
        return comb(profile.n, profile.r)
    return comb(profile.n, profile.k)


@dataclass(frozen=True)
class ProbabilityTable:
    profile_slug: str
    label: str
    n: int
    k: int
    r: int
    total_combinations: int
    expected_hits: float
    support: tuple[int, int]
    pmf: dict[int, float]
    one_in: dict[int, float]
    assumptions: list[str]
    explanation: str

    def as_dict(self) -> dict:
        rows = []
        for j in range(self.support[0], self.support[1] + 1):
            p = self.pmf[j]
            rows.append(
                {
                    "hits": j,
                    "probability": p,
                    "percent": p * 100,
                    "one_in": self.one_in[j],
                }
            )
        return {
            "kind": "calculo_teorico",
            "profile": self.profile_slug,
            "label": self.label,
            "n": self.n,
            "k": self.k,
            "r": self.r,
            "total_combinations": self.total_combinations,
            "expected_hits": self.expected_hits,
            "support": list(self.support),
            "rows": rows,
            "sum_check": sum(self.pmf.values()),
            "assumptions": self.assumptions,
            "explanation": self.explanation,
        }


def build_probability_table(profile: GameProfile) -> ProbabilityTable:
    pmf = hypergeometric_pmf(profile)
    one_in = {j: (1.0 / p if p > 0 else float("inf")) for j, p in pmf.items()}
    total = jackpot_odds(profile)
    low, high = hit_support(profile)
    return ProbabilityTable(
        profile_slug=profile.slug,
        label=profile.name,
        n=profile.n,
        k=profile.k,
        r=profile.r,
        total_combinations=total,
        expected_hits=expected_hits(profile),
        support=(low, high),
        pmf=pmf,
        one_in=one_in,
        assumptions=[
            "Extracción uniforme sin reposición.",
            "El orden no importa.",
            "Un boleto de r números distintos.",
            "Independencia entre sorteos; dependencia entre bolillas del mismo sorteo.",
        ],
        explanation=(
            "P(X=j) = C(k,j)·C(N−k, r−j) / C(N,r). "
            f"Con N={profile.n}, k={profile.k}, r={profile.r} la esperanza es "
            f"{profile.r * profile.k / profile.n:.4f} aciertos. "
            "Esto no es un premio monetario."
        ),
    )


def joint_independent_tickets(single_p: float, tickets: int, unique: bool) -> dict:
    """
    Varios boletos: si hay duplicados o el sorteo es el mismo, no son independientes.
    Solo informamos el producto cuando unique=False se declara explícitamente
    como 'aprox. si se ignorara la dependencia' — y lo etiquetamos.
    """
    if tickets < 1:
        raise ProbabilityError("La cantidad de boletos debe ser ≥ 1.")
    if not unique:
        return {
            "kind": "calculo_teorico",
            "warning": (
                "Hay boletos repetidos o se comparte el mismo sorteo: "
                "no se muestra una probabilidad conjunta como si fueran independientes."
            ),
            "tickets": tickets,
            "single_p": single_p,
        }
    fail = (1.0 - single_p) ** tickets
    return {
        "kind": "calculo_teorico",
        "assumption": (
            "Aproximación: boletos distintos evaluados sobre sorteos independientes. "
            "En un mismo sorteo los eventos no son independientes."
        ),
        "tickets": tickets,
        "p_at_least_one": 1.0 - fail,
        "single_p": single_p,
    }
