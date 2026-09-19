"""
Méthode des nuées dynamiques (Diday, 1971)
==========================================
Implémentation from scratch, sans dépendance externe.

L'utilisateur choisit explicitement le type de noyau (représentant)
de ses nuées : point, ensemble de points, axe factoriel, distribution,
ou structure représentative.
"""
from __future__ import annotations

import csv
import json
import math
import pickle
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


# =============================================================
# Types
# =============================================================
Point = List[float]
Dataset = List[Point]


# =============================================================
# Distances de base
# =============================================================
def dist_euclidienne(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dist_mahalanobis_diag(x: Sequence[float], mu: Sequence[float],
                          sigma: Sequence[float]) -> float:
    return math.sqrt(sum(
        ((x[k] - mu[k]) / sigma[k]) ** 2 if sigma[k] > 1e-12 else 0.0
        for k in range(len(x))
    ))


def dist_point_segment(p: Sequence[float], a: Sequence[float],
                       b: Sequence[float]) -> float:
    ab2 = sum((b[k] - a[k]) ** 2 for k in range(len(p)))
    if ab2 < 1e-12:
        return dist_euclidienne(p, a)
    t = sum((p[k] - a[k]) * (b[k] - a[k]) for k in range(len(p))) / ab2
    t = max(0.0, min(1.0, t))
    proj = [a[k] + t * (b[k] - a[k]) for k in range(len(p))]
    return dist_euclidienne(p, proj)


# =============================================================
# Noyaux (représentants de nuées)
# =============================================================
class Noyau:
    """Classe de base pour tous les noyaux."""
    nom: str = "abstrait"

    def distance(self, x: Sequence[float]) -> float: ...
    def mettre_a_jour(self, classe: Dataset, X: Dataset, **kw) -> "Noyau": ...
    def egal(self, autre: "Noyau", tol: float = 1e-6) -> bool: ...
    def to_dict(self) -> dict: ...


class NoyauPoint(Noyau):
    nom = "point"

    def __init__(self, point: Optional[Point] = None):
        self.point = list(point) if point is not None else None

    def distance(self, x):
        return float('inf') if self.point is None else dist_euclidienne(x, self.point)

    def mettre_a_jour(self, classe, X, **kw):
        if not classe:
            return NoyauPoint(self.point)
        n, d = len(classe), len(classe[0])
        return NoyauPoint([sum(p[k] for p in classe) / n for k in range(d)])

    def egal(self, autre, tol=1e-6):
        if not isinstance(autre, NoyauPoint) or self.point is None or autre.point is None:
            return False
        return all(abs(a - b) < tol for a, b in zip(self.point, autre.point))

    def to_dict(self):
        return {"type": "point", "point": self.point}


class NoyauEnsemblePoints(Noyau):
    nom = "ensemble_points"

    def __init__(self, points: Optional[Dataset] = None,
                 distance_to_set: str = 'sum'):
        self.points = [list(p) for p in points] if points else []
        self.distance_to_set = distance_to_set

    def distance(self, x):
        if not self.points:
            return float('inf')
        ds = [dist_euclidienne(x, p) for p in self.points]
        if self.distance_to_set == 'min':  return min(ds)
        if self.distance_to_set == 'mean': return sum(ds) / len(ds)
        return sum(ds)

    def mettre_a_jour(self, classe, X, n_etalons=1,
                      R_func: Optional[Callable] = None,
                      noyaux=None, i=None, **kw):
        if R_func is not None and noyaux is not None and i is not None:
            valeurs = [(R_func(x, i, noyaux), idx) for idx, x in enumerate(X)]
            valeurs.sort(key=lambda t: t[0])
            indices = [idx for _, idx in valeurs[:n_etalons]]
            return NoyauEnsemblePoints([X[idx] for idx in indices],
                                       self.distance_to_set)
        if not classe:
            return NoyauEnsemblePoints(self.points, self.distance_to_set)
        d = len(classe[0]); n = len(classe)
        centre = [sum(p[k] for p in classe) / n for k in range(d)]
        tries = sorted(classe, key=lambda p: dist_euclidienne(p, centre))
        return NoyauEnsemblePoints(tries[:n_etalons], self.distance_to_set)

    def egal(self, autre, tol=1e-6):
        if not isinstance(autre, NoyauEnsemblePoints):
            return False
        if len(self.points) != len(autre.points):
            return False
        s1 = sorted(tuple(p) for p in self.points)
        s2 = sorted(tuple(p) for p in autre.points)
        return all(all(abs(a - b) < tol for a, b in zip(p1, p2))
                   for p1, p2 in zip(s1, s2))

    def to_dict(self):
        return {"type": "ensemble_points", "n": len(self.points),
                "points": self.points,
                "distance_to_set": self.distance_to_set}


class NoyauAxeFactoriel(Noyau):
    nom = "axe_factoriel"

    def __init__(self, centre: Optional[Point] = None,
                 direction: Optional[Point] = None):
        self.centre = list(centre) if centre is not None else None
        self.direction = list(direction) if direction is not None else None

    def distance(self, x):
        if self.centre is None or self.direction is None:
            return float('inf')
        v = self.direction
        w = [x[k] - self.centre[k] for k in range(len(x))]
        proj = sum(w[k] * v[k] for k in range(len(x)))
        ortho = [w[k] - proj * v[k] for k in range(len(x))]
        return math.sqrt(sum(o * o for o in ortho))

    def mettre_a_jour(self, classe, X, **kw):
        if len(classe) < 2:
            c = classe[0] if classe else self.centre
            return NoyauAxeFactoriel(c, self.direction)
        d = len(classe[0]); n = len(classe)
        centre = [sum(p[k] for p in classe) / n for k in range(d)]
        cov = [[0.0] * d for _ in range(d)]
        for p in classe:
            e = [p[k] - centre[k] for k in range(d)]
            for a in range(d):
                for b in range(d):
                    cov[a][b] += e[a] * e[b]
        for a in range(d):
            for b in range(d):
                cov[a][b] /= n
        # Power iteration pour le vecteur propre principal
        v = [random.gauss(0, 1) for _ in range(d)]
        nv = math.sqrt(sum(x * x for x in v)) or 1.0
        v = [x / nv for x in v]
        for _ in range(200):
            w = [sum(cov[a][b] * v[b] for b in range(d)) for a in range(d)]
            nw = math.sqrt(sum(x * x for x in w))
            if nw < 1e-12:
                break
            v_new = [x / nw for x in w]
            if all(abs(v_new[k] - v[k]) < 1e-10 for k in range(d)):
                v = v_new; break
            v = v_new
        return NoyauAxeFactoriel(centre, v)

    def egal(self, autre, tol=1e-6):
        if not isinstance(autre, NoyauAxeFactoriel):
            return False
        if self.centre is None or autre.centre is None:
            return False
        if not all(abs(a - b) < tol for a, b in zip(self.centre, autre.centre)):
            return False
        s = sum(a * b for a, b in zip(self.direction, autre.direction))
        return abs(abs(s) - 1.0) < tol

    def to_dict(self):
        return {"type": "axe_factoriel",
                "centre": self.centre, "direction": self.direction}


class NoyauDistribution(Noyau):
    nom = "distribution"

    def __init__(self, centre: Optional[Point] = None,
                 ecarts_types: Optional[Point] = None):
        self.centre = list(centre) if centre is not None else None
        self.ecarts_types = list(ecarts_types) if ecarts_types is not None else None

    def distance(self, x):
        if self.centre is None or self.ecarts_types is None:
            return float('inf')
        return dist_mahalanobis_diag(x, self.centre, self.ecarts_types)

    def mettre_a_jour(self, classe, X, **kw):
        if not classe:
            return NoyauDistribution(self.centre, self.ecarts_types)
        d = len(classe[0]); n = len(classe)
        centre = [sum(p[k] for p in classe) / n for k in range(d)]
        ecarts = []
        for k in range(d):
            var = sum((p[k] - centre[k]) ** 2 for p in classe) / n
            ecarts.append(math.sqrt(var) if var > 1e-12 else 1e-6)
        return NoyauDistribution(centre, ecarts)

    def egal(self, autre, tol=1e-6):
        if not isinstance(autre, NoyauDistribution):
            return False
        if self.centre is None or autre.centre is None:
            return False
        return (all(abs(a - b) < tol for a, b in zip(self.centre, autre.centre))
                and all(abs(a - b) < tol
                        for a, b in zip(self.ecarts_types, autre.ecarts_types)))

    def to_dict(self):
        return {"type": "distribution",
                "centre": self.centre, "ecarts_types": self.ecarts_types}


class NoyauStructureRepresentative(Noyau):
    nom = "structure_representative"

    def __init__(self, points: Optional[Dataset] = None):
        self.points = [list(p) for p in points] if points else []

    def distance(self, x):
        if not self.points:
            return float('inf')
        if len(self.points) == 1:
            return dist_euclidienne(x, self.points[0])
        return min(dist_point_segment(x, self.points[i], self.points[i + 1])
                   for i in range(len(self.points) - 1))

    def mettre_a_jour(self, classe, X, n_etalons=1, **kw):
        if not classe:
            return NoyauStructureRepresentative(self.points)
        d = len(classe[0]); n = len(classe)
        centre = [sum(p[k] for p in classe) / n for k in range(d)]
        tries = sorted(classe, key=lambda p: dist_euclidienne(p, centre))
        etalons = tries[:n_etalons]
        if len(etalons) <= 2:
            return NoyauStructureRepresentative(etalons)
        chaine = [etalons[0]]
        restants = etalons[1:]
        while restants:
            dernier = chaine[-1]
            suivant = min(restants, key=lambda p: dist_euclidienne(dernier, p))
            chaine.append(suivant); restants.remove(suivant)
        return NoyauStructureRepresentative(chaine)

    def egal(self, autre, tol=1e-6):
        if not isinstance(autre, NoyauStructureRepresentative):
            return False
        if len(self.points) != len(autre.points):
            return False
        return all(all(abs(a - b) < tol for a, b in zip(p1, p2))
                   for p1, p2 in zip(self.points, autre.points))

    def to_dict(self):
        return {"type": "structure_representative",
                "n": len(self.points), "points": self.points}


# Registre des types de noyaux
TYPES_NOYAUX = {
    "point": NoyauPoint,
    "ensemble_points": NoyauEnsemblePoints,
    "axe_factoriel": NoyauAxeFactoriel,
    "distribution": NoyauDistribution,
    "structure_representative": NoyauStructureRepresentative,
}

DESCRIPTIONS = {
    "point": "Un seul point (revient à k-means)",
    "ensemble_points": "n1 points représentatifs (forme multi-exemplaires)",
    "axe_factoriel": "Axe factoriel (centre + direction principale)",
    "distribution": "Distribution gaussienne (centre + écarts-types)",
    "structure_representative": "Chaîne polygonale de n1 points (forme curviligne)",
}


# =============================================================
# Configuration
# =============================================================
@dataclass
class Config:
    n_clusters: int = 2
    type_noyau: str = "point"
    n_etalons: int = 1
    distance_to_set: str = "sum"
    R: str = "ex2"
    max_iter: int = 100
    tol: float = 1e-6
    n_init: int = 1
    normaliser: bool = False
    random_state: Optional[int] = None

    def valider(self, n_points: int, dim: int) -> None:
        if self.n_clusters < 1:
            raise ValueError("n_clusters doit être ≥ 1")
        if self.n_clusters > n_points:
            raise ValueError("n_clusters > nombre de points")
        if self.type_noyau not in TYPES_NOYAUX:
            raise ValueError(f"type_noyau inconnu : {self.type_noyau}")
        if self.type_noyau in ("ensemble_points", "structure_representative"):
            if self.n_etalons < 1:
                raise ValueError("n_etalons doit être ≥ 1")
            if self.n_etalons > n_points:
                raise ValueError("n_etalons > nombre de points")
        if self.distance_to_set not in ("min", "sum", "mean"):
            raise ValueError("distance_to_set ∈ {min, sum, mean}")
        if self.R not in ("ex1", "ex2"):
            raise ValueError("R ∈ {ex1, ex2}")
        if self.max_iter < 1:
            raise ValueError("max_iter doit être ≥ 1")
        if self.n_init < 1:
            raise ValueError("n_init doit être ≥ 1")


# =============================================================
# Modèle des nuées dynamiques
# =============================================================
class NueesDynamiques:
    def __init__(self, config: Config):
        self.cfg = config
        self.noyaux_: Optional[List[Noyau]] = None
        self.labels_: Optional[List[int]] = None
        self.classes_: Optional[List[Dataset]] = None
        self.n_iter_: int = 0
        self.S_: Optional[float] = None
        self.mu_: Optional[Point] = None      # moyenne (si normalisation)
        self.sigma_: Optional[Point] = None   # écart-type (si normalisation)

    # ---------- Normalisation ----------
    def _normaliser(self, X: Dataset) -> Dataset:
        d = len(X[0]); n = len(X)
        self.mu_ = [sum(p[k] for p in X) / n for k in range(d)]
        self.sigma_ = []
        for k in range(d):
            var = sum((p[k] - self.mu_[k]) ** 2 for p in X) / n
            self.sigma_.append(math.sqrt(var) if var > 1e-12 else 1.0)
        return [[(p[k] - self.mu_[k]) / self.sigma_[k] for k in range(d)]
                for p in X]

    def _denormaliser_point(self, p: Point) -> Point:
        if self.mu_ is None:
            return p
        return [p[k] * self.sigma_[k] + self.mu_[k] for k in range(len(p))]

    # ---------- Utilitaires ----------
    def _distance_classe(self, x: Point, classe: Dataset) -> float:
        if not classe:
            return float('inf')
        ds = [dist_euclidienne(x, p) for p in classe]
        if self.cfg.distance_to_set == 'min':  return min(ds)
        if self.cfg.distance_to_set == 'mean': return sum(ds) / len(ds)
        return sum(ds)

    def _R(self, x, i, noyaux):
        if self.cfg.R == 'ex1':
            num = noyaux[i].distance(x) * self._distance_classe(
                x, self.classes_[i])
            denom = sum(noyaux[j].distance(x)
                        for j in range(self.cfg.n_clusters)) ** 2
            return num / denom if denom != 0 else float('inf')
        return self._distance_classe(x, self.classes_[i])

    # ---------- Initialisation ----------
    def _initialiser_un_noyau(self, X: Dataset,
                              forcer_indice: Optional[int] = None) -> Noyau:
        n, d = len(X), len(X[0])
        t = self.cfg.type_noyau
        if t == "point":
            idx = forcer_indice if forcer_indice is not None \
                  else random.randrange(n)
            return NoyauPoint(list(X[idx]))
        if t == "ensemble_points":
            pts = random.sample(X, min(self.cfg.n_etalons, n))
            return NoyauEnsemblePoints(pts, self.cfg.distance_to_set)
        if t == "axe_factoriel":
            idx = forcer_indice if forcer_indice is not None \
                  else random.randrange(n)
            centre = list(X[idx])
            direction = [random.gauss(0, 1) for _ in range(d)]
            nv = math.sqrt(sum(v * v for v in direction)) or 1.0
            return NoyauAxeFactoriel(centre, [v / nv for v in direction])
        if t == "distribution":
            idx = forcer_indice if forcer_indice is not None \
                  else random.randrange(n)
            return NoyauDistribution(list(X[idx]), [1.0] * d)
        if t == "structure_representative":
            pts = random.sample(X, min(self.cfg.n_etalons, n))
            return NoyauStructureRepresentative(pts)
        raise ValueError(f"Type inconnu : {t}")

    def _initialiser_kmeans_plus_plus(self, X: Dataset) -> List[Noyau]:
        """Init k-means++ adaptée à tous les types de noyaux."""
        n = len(X)
        premiers_idx = [random.randrange(n)]
        while len(premiers_idx) < self.cfg.n_clusters:
            d2 = []
            for p in X:
                dmin = min(dist_euclidienne(p, X[i]) ** 2 for i in premiers_idx)
                d2.append(dmin)
            total = sum(d2)
            if total == 0:
                idx = random.randrange(n)
            else:
                r = random.random() * total
                cum = 0.0
                idx = 0
                for i, val in enumerate(d2):
                    cum += val
                    if cum >= r:
                        idx = i; break
            premiers_idx.append(idx)
        return [self._initialiser_un_noyau(X, forcer_indice=i)
                for i in premiers_idx]

    def _initialiser(self, X: Dataset) -> List[Noyau]:
        if self.cfg.n_clusters <= 1:
            return [self._initialiser_un_noyau(X, forcer_indice=0)]
        return self._initialiser_kmeans_plus_plus(X)

    # ---------- Étapes ----------
    def _affecter(self, X: Dataset, noyaux: List[Noyau]) -> List[int]:
        labels = []
        for x in X:
            best, best_d = 0, float('inf')
            for i, n in enumerate(noyaux):
                d = n.distance(x)
                if d < best_d:
                    best_d, best = d, i
            labels.append(best)
        return labels

    def _construire_classes(self, X: Dataset,
                            labels: List[int]) -> List[Dataset]:
        classes: List[Dataset] = [[] for _ in range(self.cfg.n_clusters)]
        for x, lab in zip(X, labels):
            classes[lab].append(x)
        return classes

    def _mettre_a_jour(self, noyaux, classes, X):
        nouveaux = []
        for i in range(self.cfg.n_clusters):
            n = noyaux[i]
            if isinstance(n, NoyauEnsemblePoints):
                nouveaux.append(n.mettre_a_jour(
                    classes[i], X,
                    n_etalons=self.cfg.n_etalons,
                    R_func=self._R, noyaux=noyaux, i=i))
            elif isinstance(n, NoyauStructureRepresentative):
                nouveaux.append(n.mettre_a_jour(
                    classes[i], X, n_etalons=self.cfg.n_etalons))
            else:
                nouveaux.append(n.mettre_a_jour(classes[i], X))
        return nouveaux

    # ---------- Un run ----------
    def _un_run(self, X: Dataset, verbose: bool) -> Tuple[List[Noyau], int, float]:
        noyaux = self._initialiser(X)
        K = self.cfg.n_clusters
        for iteration in range(self.cfg.max_iter):
            labels = self._affecter(X, noyaux)
            self.classes_ = self._construire_classes(X, labels)

            # Réinitialisation des classes vides
            for i in range(K):
                if not self.classes_[i]:
                    noyaux[i] = self._initialiser_un_noyau(X)

            nouveaux = self._mettre_a_jour(noyaux, self.classes_, X)

            if verbose and (iteration + 1) % 5 == 0:
                print(f"    itération {iteration + 1:4d} ...")

            if all(noyaux[i].egal(nouveaux[i], tol=self.cfg.tol)
                   for i in range(K)):
                noyaux = nouveaux
                labels = self._affecter(X, noyaux)
                self.classes_ = self._construire_classes(X, labels)
                S = sum(self._R(x, labels[k], noyaux)
                        for k, x in enumerate(X))
                return noyaux, iteration + 1, S

            noyaux = nouveaux

        labels = self._affecter(X, noyaux)
        self.classes_ = self._construire_classes(X, labels)
        S = sum(self._R(x, labels[k], noyaux) for k, x in enumerate(X))
        return noyaux, self.cfg.max_iter, S

    # ---------- Fit global ----------
    def fit(self, X: Dataset, verbose: bool = False) -> "NueesDynamiques":
        if not X:
            raise ValueError("X est vide")
        d = len(X[0])
        if any(len(p) != d for p in X):
            raise ValueError("Tous les points doivent avoir la même dimension")
        self.cfg.valider(len(X), d)

        if self.cfg.random_state is not None:
            random.seed(self.cfg.random_state)

        # Normalisation optionnelle
        if self.cfg.normaliser:
            X_work = self._normaliser(X)
        else:
            X_work = [list(p) for p in X]

        meilleur = None
        for run in range(self.cfg.n_init):
            if verbose:
                print(f"  Run {run + 1}/{self.cfg.n_init}")
            noyaux, n_iter, S = self._un_run(X_work, verbose)
            if meilleur is None or S < meilleur[2]:
                meilleur = (noyaux, n_iter, S)

        self.noyaux_, self.n_iter_, self.S_ = meilleur
        self.labels_ = self._affecter(X_work, self.noyaux_)
        self.classes_ = self._construire_classes(X_work, self.labels_)
        return self

    def predict(self, X: Dataset) -> List[int]:
        if self.noyaux_ is None:
            raise RuntimeError("Modèle non entraîné")
        X_work = X
        if self.cfg.normaliser and self.mu_ is not None:
            X_work = [[(p[k] - self.mu_[k]) / self.sigma_[k]
                       for k in range(len(p))] for p in X]
        return self._affecter(X_work, self.noyaux_)

    def fit_predict(self, X: Dataset, verbose: bool = False) -> List[int]:
        self.fit(X, verbose=verbose)
        return self.labels_

    # ---------- Résumé ----------
    def resume(self) -> dict:
        if self.noyaux_ is None:
            raise RuntimeError("Modèle non entraîné")
        return {
            "config": asdict(self.cfg),
            "n_iter": self.n_iter_,
            "S": self.S_,
            "tailles_classes": [len(c) for c in self.classes_],
            "noyaux": [n.to_dict() for n in self.noyaux_],
        }

    # ---------- Sauvegarde / chargement ----------
    def save(self, chemin: str | Path) -> None:
        with open(chemin, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(chemin: str | Path) -> "NueesDynamiques":
        with open(chemin, "rb") as f:
            return pickle.load(f)


# =============================================================
# Chargement de données
# =============================================================
def charger_csv(chemin: str | Path, entete: bool = True) -> Dataset:
    X: Dataset = []
    with open(chemin, newline="") as f:
        lecteur = csv.reader(f)
        if entete:
            next(lecteur, None)
        for ligne in lecteur:
            try:
                X.append([float(v) for v in ligne])
            except ValueError:
                continue
    return X


def donnees_exemple(n1: int = 80, n2: int = 80,
                    seed: int = 0) -> Dataset:
    random.seed(seed)
    X: Dataset = []
    for _ in range(n1):
        t = random.uniform(0, 10)
        X.append([t + random.gauss(0, 0.2), 0.5 * t + random.gauss(0, 0.2)])
    for _ in range(n2):
        t = random.uniform(0, 10)
        X.append([t + random.gauss(0, 0.2),
                  -0.5 * t + 10 + random.gauss(0, 0.2)])
    return X


# =============================================================
# Visualisation ASCII (2D)
# =============================================================
def afficher_ascii(X: Dataset, labels: List[int],
                   symboles: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                   largeur: int = 70, hauteur: int = 30) -> None:
    if len(X[0]) != 2:
        print("(visualisation ASCII réservée aux données 2D)")
        return
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0

    grille = [[" "] * largeur for _ in range(hauteur)]
    for (x, y), lab in zip(X, labels):
        cx = int((x - xmin) / dx * (largeur - 1))
        cy = int((y - ymin) / dy * (hauteur - 1))
        grille[hauteur - 1 - cy][cx] = symboles[lab % len(symboles)]

    print("+" + "-" * largeur + "+")
    for ligne in grille:
        print("|" + "".join(ligne) + "|")
    print("+" + "-" * largeur + "+")


# =============================================================
# Interface utilisateur interactive
# =============================================================
def _demander_entier(msg: str, defaut: Optional[int] = None,
                     mini: Optional[int] = None,
                     maxi: Optional[int] = None) -> int:
    while True:
        suffixe = f" [défaut={defaut}]" if defaut is not None else ""
        rep = input(f"{msg}{suffixe} : ").strip()
        if not rep and defaut is not None:
            return defaut
        try:
            v = int(rep)
        except ValueError:
            print("  Entier attendu."); continue
        if mini is not None and v < mini:
            print(f"  Valeur ≥ {mini} attendue."); continue
        if maxi is not None and v > maxi:
            print(f"  Valeur ≤ {maxi} attendue."); continue
        return v


def _demander_flottant(msg: str, defaut: Optional[float] = None) -> float:
    while True:
        suffixe = f" [défaut={defaut}]" if defaut is not None else ""
        rep = input(f"{msg}{suffixe} : ").strip()
        if not rep and defaut is not None:
            return defaut
        try:
            return float(rep)
        except ValueError:
            print("  Nombre attendu.")


def _demander_oui_non(msg: str, defaut: bool = False) -> bool:
    rep = input(f"{msg} [{'O/n' if defaut else 'o/N'}] : ").strip().lower()
    if not rep:
        return defaut
    return rep.startswith("o")


def _choisir_menu(titre: str, options: List[Tuple[str, str]],
                  defaut_idx: int = 0) -> int:
    print(f"\n{titre}")
    for i, (cle, desc) in enumerate(options, 1):
        marque = " *" if i - 1 == defaut_idx else "  "
        print(f" {marque}{i}) {cle:30s} → {desc}")
    while True:
        rep = input(f"Votre choix [1-{len(options)}, défaut={defaut_idx + 1}] : ").strip()
        if not rep:
            return defaut_idx
        try:
            idx = int(rep) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("  Choix invalide.")


def construire_config_interactive(X: Dataset) -> Config:
    n, d = len(X), len(X[0])
    print("=" * 72)
    print(" MÉTHODE DES NUÉES DYNAMIQUES (Diday, 1971)")
    print(" C'est VOUS qui choisissez le type de représentant de vos nuées.")
    print("=" * 72)
    print(f"\nDonnées : {n} points en {d} dimension(s).")

    K = _demander_entier("\nNombre de classes K", defaut=2, mini=1, maxi=n)

    options_noyaux = [(t, DESCRIPTIONS[t]) for t in TYPES_NOYAUX]
    idx = _choisir_menu("Type de noyau :", options_noyaux, defaut_idx=0)
    type_noyau = options_noyaux[idx][0]

    n_etalons = 1
    if type_noyau in ("ensemble_points", "structure_representative"):
        n_etalons = _demander_entier(
            f"Nombre d'étalons n1 par noyau", defaut=8, mini=1, maxi=n)

    distance_to_set = "sum"
    if type_noyau in ("ensemble_points", "structure_representative"):
        idx = _choisir_menu(
            "Distance d'un point à un ensemble :",
            [("min", "distance minimale aux étalons"),
             ("sum", "somme des distances aux étalons"),
             ("mean", "moyenne des distances aux étalons")],
            defaut_idx=1)
        distance_to_set = ["min", "sum", "mean"][idx]

    idx = _choisir_menu(
        "Fonction d'agrégation-écartement R :",
        [("ex2", "R(x,i,L) = D(x, C_i)  (compactage)"),
         ("ex1", "R = D(x,E_i)·D(x,C_i) / (Σ_j D(x,E_j))²")],
        defaut_idx=0)
    R = ["ex2", "ex1"][idx]

    max_iter = _demander_entier(
        "\nNombre maximum d'itérations", defaut=100, mini=1)
    tol = _demander_flottant("Tolérance de convergence", defaut=1e-6)
    n_init = _demander_entier(
        "Nombre de redémarrages (n_init)", defaut=3, mini=1)

    normaliser = _demander_oui_non(
        "Normaliser les données (z-score) ?", defaut=True)

    graine = _demander_entier(
        "Graine aléatoire (vide = aléatoire)", defaut=42)

    return Config(
        n_clusters=K, type_noyau=type_noyau, n_etalons=n_etalons,
        distance_to_set=distance_to_set, R=R,
        max_iter=max_iter, tol=tol, n_init=n_init,
        normaliser=normaliser, random_state=graine,
    )


def afficher_resultats(model: NueesDynamiques) -> None:
    info = model.resume()
    print("\n" + "=" * 72)
    print(" RÉSULTATS")
    print("=" * 72)
    print(f"  Type de noyau    : {info['config']['type_noyau']}")
    print(f"  K                : {info['config']['n_clusters']}")
    print(f"  Itérations       : {info['n_iter']}")
    print(f"  S(L, L)          : {info['S']:.4f}")
    print(f"  Tailles classes  : {info['tailles_classes']}")
    print("\n  Détail des noyaux :")
    for i, n in enumerate(info["noyaux"]):
        t = n["type"]
        if t == "point":
            print(f"    [{i}] point          = "
                  f"{[round(v, 3) for v in n['point']]}")
        elif t == "axe_factoriel":
            print(f"    [{i}] axe            : centre="
                  f"{[round(v, 3) for v in n['centre']]}, "
                  f"direction={[round(v, 3) for v in n['direction']]}")
        elif t == "distribution":
            print(f"    [{i}] distribution   : centre="
                  f"{[round(v, 3) for v in n['centre']]}, "
                  f"σ={[round(v, 3) for v in n['ecarts_types']]}")
        elif t == "ensemble_points":
            print(f"    [{i}] {n['n']} étalons "
                  f"(distance_to_set={n['distance_to_set']})")
        elif t == "structure_representative":
            print(f"    [{i}] chaîne de {n['n']} étalons")


def main() -> None:
    # 1) Données
    print("Source des données :")
    print("  1) Exemple synthétique 2D")
    print("  2) Fichier CSV")
    rep = input("Votre choix [1/2, défaut=1] : ").strip() or "1"
    if rep == "2":
        chemin = input("Chemin du fichier CSV : ").strip()
        try:
            X = charger_csv(chemin)
            print(f"  {len(X)} points chargés.")
        except Exception as e:
            print(f"  Erreur : {e}. Utilisation de l'exemple.")
            X = donnees_exemple()
    else:
        X = donnees_exemple()

    if not X:
        print("Aucune donnée. Abandon.")
        return

    # 2) Configuration interactive
    cfg = construire_config_interactive(X)
    print("\nConfiguration retenue :")
    for k, v in asdict(cfg).items():
        print(f"  {k:18s} = {v}")

    # 3) Fit
    print("\n" + "=" * 72)
    print(" APPRENTISSAGE")
    print("=" * 72)
    model = NueesDynamiques(cfg)
    model.fit(X, verbose=True)

    # 4) Résultats
    afficher_resultats(model)

    # 5) Visualisation 2D si possible
    if len(X[0]) == 2:
        print("\nVisualisation ASCII (lettres = classes) :")
        afficher_ascii(X, model.labels_)

    # 6) Sauvegarde éventuelle
    if _demander_oui_non("\nSauvegarder le modèle ?", defaut=False):
        chemin = input("Chemin du fichier (.pkl) : ").strip() or "modele.pkl"
        try:
            model.save(chemin)
            print(f"  Modèle sauvegardé dans {chemin}")
        except Exception as e:
            print(f"  Erreur : {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.")
        sys.exit(1)