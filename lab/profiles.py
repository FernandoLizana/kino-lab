from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GameProfile:
    slug: str
    name: str
    version: str
    n: int  # universo
    k: int  # extraídos en el sorteo
    r: int  # números del boleto
    ordered: bool
    replacement: bool
    official: bool
    notes: str
    synthetic: bool = False
    valid_from: str | None = None  # observado o declarado; None = desconocido
    valid_to: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def supports_exact_hypergeometric(self) -> bool:
        return (not self.ordered) and (not self.replacement) and self.k > 0 and self.r > 0


PROFILES: dict[str, GameProfile] = {
    "kino_moderno": GameProfile(
        slug="kino_moderno",
        name="Kino moderno (14 de 25)",
        version="1",
        n=25,
        k=14,
        r=14,
        ordered=False,
        replacement=False,
        official=True,
        notes=(
            "Modalidad actual. En historico.csv las filas de 14 números "
            "aparecen desde 08-01-2006. Esa fecha es observada en el archivo, "
            "no una fecha oficial inventada."
        ),
        valid_from="2006-01-08",
    ),
    "kino_clasico": GameProfile(
        slug="kino_clasico",
        name="Kino clásico (15 de 25)",
        version="1",
        n=25,
        k=15,
        r=15,
        ordered=False,
        replacement=False,
        official=True,
        notes=(
            "Filas de 15 números en historico.csv entre 19-09-1990 y 01-01-2006. "
            "No se convierten a 14. Se analizan solo con este perfil."
        ),
        valid_from="1990-09-19",
        valid_to="2006-01-01",
    ),
    "demo_6_3": GameProfile(
        slug="demo_6_3",
        name="Demostración 3 de 6 (sintético)",
        version="1",
        n=6,
        k=3,
        r=3,
        ordered=False,
        replacement=False,
        official=False,
        synthetic=True,
        notes="Juego diminuto para enumerar las 20 combinaciones y validar la hipergeométrica.",
    ),
    "demo_uniforme": GameProfile(
        slug="demo_uniforme",
        name="Kino sintético uniforme (demo)",
        version="1",
        n=25,
        k=14,
        r=14,
        ordered=False,
        replacement=False,
        official=False,
        synthetic=True,
        notes="Sorteos inventados con muestreo uniforme. No son resultados reales.",
    ),
}


def get_profile(slug: str) -> GameProfile:
    if slug not in PROFILES:
        raise KeyError(f"Perfil desconocido: {slug}")
    return PROFILES[slug]


def list_profiles() -> list[GameProfile]:
    return list(PROFILES.values())


def profile_for_draw_size(ball_count: int, n: int = 25) -> GameProfile | None:
    for profile in PROFILES.values():
        if profile.k == ball_count and profile.n == n and not profile.synthetic:
            return profile
    return None


def duplicate_profile(slug: str, new_slug: str, name: str) -> GameProfile:
    base = get_profile(slug)
    copy = GameProfile(
        slug=new_slug,
        name=name,
        version=str(int(base.version) + 1),
        n=base.n,
        k=base.k,
        r=base.r,
        ordered=base.ordered,
        replacement=base.replacement,
        official=False,
        notes=f"Copia editable de {slug}.",
        synthetic=base.synthetic,
        valid_from=base.valid_from,
        valid_to=base.valid_to,
    )
    PROFILES[new_slug] = copy
    return copy
