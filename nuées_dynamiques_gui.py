"""
Méthode des nuées dynamiques (Diday, 1971)
Algorithme + interface graphique modernisée (Tkinter).
"""
from __future__ import annotations

import csv
import math
import pickle
import random
import sys
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, asdict
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

Point = List[float]
Dataset = List[Point]


# =============================================================
# Design system
# =============================================================
COLORS = {
    "bg":             "#f1f5f9",
    "surface":        "#ffffff",
    "surface_alt":    "#f8fafc",
    "sidebar":        "#0f172a",
    "sidebar_hover":  "#1e293b",
    "sidebar_text":   "#cbd5e1",
    "sidebar_active": "#6366f1",
    "primary":        "#6366f1",
    "primary_hover":  "#4f46e5",
    "primary_text":   "#ffffff",
    "success":        "#10b981",
    "success_hover":  "#059669",
    "danger":         "#ef4444",
    "danger_hover":   "#dc2626",
    "warning":        "#f59e0b",
    "text":           "#0f172a",
    "text_muted":     "#64748b",
    "text_light":     "#94a3b8",
    "border":         "#e2e8f0",
    "border_strong":  "#cbd5e1",
    "canvas_bg":      "#fbfdff",
    "grid":           "#eef2f7",
    "accent_soft":    "#eef2ff",
}

FONT_FAMILY = None  # détecté au lancement


PALETTE = ["#6366f1", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
           "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#14b8a6",
           "#a855f7", "#eab308", "#3b82f6", "#f43f5e", "#22c55e",
           "#0ea5e9", "#d946ef", "#64748b"]

SHAPES = ["●", "■", "▲", "◆", "★", "✚", "⬢", "◐", "◭", "▣"]


# =============================================================
# Distances
# =============================================================
def dist_euclidienne(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dist_mahalanobis_diag(x, mu, sigma):
    return math.sqrt(sum(
        ((x[k] - mu[k]) / sigma[k]) ** 2 if sigma[k] > 1e-12 else 0.0
        for k in range(len(x))
    ))


def dist_point_segment(p, a, b):
    ab2 = sum((b[k] - a[k]) ** 2 for k in range(len(p)))
    if ab2 < 1e-12:
        return dist_euclidienne(p, a)
    t = sum((p[k] - a[k]) * (b[k] - a[k]) for k in range(len(p))) / ab2
    t = max(0.0, min(1.0, t))
    proj = [a[k] + t * (b[k] - a[k]) for k in range(len(p))]
    return dist_euclidienne(p, proj)


# =============================================================
# Noyaux
# =============================================================
class Noyau:
    nom = "abstrait"
    def distance(self, x): raise NotImplementedError
    def mettre_a_jour(self, classe, X, **kw): raise NotImplementedError
    def egal(self, autre, tol=1e-6): raise NotImplementedError
    def to_dict(self): raise NotImplementedError


class NoyauPoint(Noyau):
    nom = "point"
    def __init__(self, point=None):
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
    def __init__(self, points=None, distance_to_set='sum'):
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
                      R_func=None, noyaux=None, i=None, **kw):
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
    def __init__(self, centre=None, direction=None):
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
    def __init__(self, centre=None, ecarts_types=None):
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
    def __init__(self, points=None):
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
        chaine = [etalons[0]]; restants = etalons[1:]
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


TYPES_NOYAUX = {
    "point": NoyauPoint,
    "ensemble_points": NoyauEnsemblePoints,
    "axe_factoriel": NoyauAxeFactoriel,
    "distribution": NoyauDistribution,
    "structure_representative": NoyauStructureRepresentative,
}

DESCRIPTIONS = {
    "point": "Un seul point — équivalent k-means",
    "ensemble_points": "n1 points représentatifs",
    "axe_factoriel": "Centre + direction principale",
    "distribution": "Centre + écarts-types",
    "structure_representative": "Chaîne polygonale de n1 points",
}

ICONES_NOYAUX = {
    "point": "●",
    "ensemble_points": "⬢",
    "axe_factoriel": "⟋",
    "distribution": "◎",
    "structure_representative": "⛓",
}

AIDES = {
    "point": "Noyau = point unique.\n\n"
             "D(x, Eᵢ) : distance euclidienne au point.\n\n"
             "Mise à jour : centre de gravité de la classe.\n\n"
             "→ Équivalent à l'algorithme des k-moyennes.",
    "ensemble_points": "Noyau = ensemble de n₁ points représentatifs.\n\n"
             "D(x, Eᵢ) : min, somme ou moyenne des distances aux étalons.\n\n"
             "Mise à jour : sélection des n₁ points qui minimisent R(x, i, L) "
             "sur tout le jeu de données.",
    "axe_factoriel": "Noyau = axe factoriel (centre μ + direction v).\n\n"
             "D(x, Eᵢ) : distance orthogonale à l'axe.\n\n"
             "Mise à jour : μ = moyenne, v = premier vecteur propre de la "
             "matrice de covariance (power iteration).\n\n"
             "→ Adapté aux formes allongées.",
    "distribution": "Noyau = distribution gaussienne à covariance diagonale.\n\n"
             "D(x, Eᵢ) : distance de Mahalanobis diagonale.\n\n"
             "Mise à jour : μ = moyenne, σ = écart-type par dimension.\n\n"
             "→ Adapté aux classes de variabilité différente.",
    "structure_representative": "Noyau = chaîne polygonale de n₁ étalons.\n\n"
             "D(x, Eᵢ) : distance min aux segments de la chaîne.\n\n"
             "Mise à jour : n₁ étalons proches du centre, ordonnés par plus "
             "proche voisin.\n\n"
             "→ Adapté aux formes curvilignes.",
}


# =============================================================
# Configuration + Modèle
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


class NueesDynamiques:
    def __init__(self, config: Config):
        self.cfg = config
        self.noyaux_: Optional[List[Noyau]] = None
        self.labels_: Optional[List[int]] = None
        self.classes_: Optional[List[Dataset]] = None
        self.n_iter_: int = 0
        self.S_: Optional[float] = None
        self.mu_: Optional[Point] = None
        self.sigma_: Optional[Point] = None

    def _normaliser(self, X):
        d = len(X[0]); n = len(X)
        self.mu_ = [sum(p[k] for p in X) / n for k in range(d)]
        self.sigma_ = []
        for k in range(d):
            var = sum((p[k] - self.mu_[k]) ** 2 for p in X) / n
            self.sigma_.append(math.sqrt(var) if var > 1e-12 else 1.0)
        return [[(p[k] - self.mu_[k]) / self.sigma_[k] for k in range(d)]
                for p in X]

    def _distance_classe(self, x, classe):
        if not classe:
            return float('inf')
        ds = [dist_euclidienne(x, p) for p in classe]
        if self.cfg.distance_to_set == 'min':  return min(ds)
        if self.cfg.distance_to_set == 'mean': return sum(ds) / len(ds)
        return sum(ds)

    def _R(self, x, i, noyaux):
        if self.cfg.R == 'ex1':
            num = noyaux[i].distance(x) * self._distance_classe(x, self.classes_[i])
            denom = sum(noyaux[j].distance(x)
                        for j in range(self.cfg.n_clusters)) ** 2
            return num / denom if denom != 0 else float('inf')
        return self._distance_classe(x, self.classes_[i])

    def _initialiser_un_noyau(self, X, forcer_indice=None):
        n, d = len(X), len(X[0])
        t = self.cfg.type_noyau
        if t == "point":
            idx = forcer_indice if forcer_indice is not None else random.randrange(n)
            return NoyauPoint(list(X[idx]))
        if t == "ensemble_points":
            pts = random.sample(X, min(self.cfg.n_etalons, n))
            return NoyauEnsemblePoints(pts, self.cfg.distance_to_set)
        if t == "axe_factoriel":
            idx = forcer_indice if forcer_indice is not None else random.randrange(n)
            centre = list(X[idx])
            direction = [random.gauss(0, 1) for _ in range(d)]
            nv = math.sqrt(sum(v * v for v in direction)) or 1.0
            return NoyauAxeFactoriel(centre, [v / nv for v in direction])
        if t == "distribution":
            idx = forcer_indice if forcer_indice is not None else random.randrange(n)
            return NoyauDistribution(list(X[idx]), [1.0] * d)
        if t == "structure_representative":
            pts = random.sample(X, min(self.cfg.n_etalons, n))
            return NoyauStructureRepresentative(pts)
        raise ValueError(t)

    def _initialiser_kmeans_plus_plus(self, X):
        n = len(X)
        premiers = [random.randrange(n)]
        while len(premiers) < self.cfg.n_clusters:
            d2 = []
            for p in X:
                dmin = min(dist_euclidienne(p, X[i]) ** 2 for i in premiers)
                d2.append(dmin)
            total = sum(d2)
            if total == 0:
                idx = random.randrange(n)
            else:
                r = random.random() * total
                cum = 0.0; idx = 0
                for i, val in enumerate(d2):
                    cum += val
                    if cum >= r:
                        idx = i; break
            premiers.append(idx)
        return [self._initialiser_un_noyau(X, forcer_indice=i)
                for i in premiers]

    def _initialiser(self, X):
        if self.cfg.n_clusters <= 1:
            return [self._initialiser_un_noyau(X, forcer_indice=0)]
        return self._initialiser_kmeans_plus_plus(X)

    def _affecter(self, X, noyaux):
        labels = []
        for x in X:
            best, best_d = 0, float('inf')
            for i, n in enumerate(noyaux):
                d = n.distance(x)
                if d < best_d:
                    best_d, best = d, i
            labels.append(best)
        return labels

    def _construire_classes(self, X, labels):
        classes = [[] for _ in range(self.cfg.n_clusters)]
        for x, lab in zip(X, labels):
            classes[lab].append(x)
        return classes

    def _mettre_a_jour(self, noyaux, classes, X):
        nouveaux = []
        for i in range(self.cfg.n_clusters):
            n = noyaux[i]
            if isinstance(n, NoyauEnsemblePoints):
                nouveaux.append(n.mettre_a_jour(
                    classes[i], X, n_etalons=self.cfg.n_etalons,
                    R_func=self._R, noyaux=noyaux, i=i))
            elif isinstance(n, NoyauStructureRepresentative):
                nouveaux.append(n.mettre_a_jour(
                    classes[i], X, n_etalons=self.cfg.n_etalons))
            else:
                nouveaux.append(n.mettre_a_jour(classes[i], X))
        return nouveaux

    def _un_run(self, X):
        noyaux = self._initialiser(X)
        K = self.cfg.n_clusters
        for iteration in range(self.cfg.max_iter):
            labels = self._affecter(X, noyaux)
            self.classes_ = self._construire_classes(X, labels)
            for i in range(K):
                if not self.classes_[i]:
                    noyaux[i] = self._initialiser_un_noyau(X)
            nouveaux = self._mettre_a_jour(noyaux, self.classes_, X)
            if all(noyaux[i].egal(nouveaux[i], tol=self.cfg.tol)
                   for i in range(K)):
                noyaux = nouveaux
                labels = self._affecter(X, noyaux)
                self.classes_ = self._construire_classes(X, labels)
                S = sum(self._R(x, labels[k], noyaux) for k, x in enumerate(X))
                return noyaux, iteration + 1, S
            noyaux = nouveaux
        labels = self._affecter(X, noyaux)
        self.classes_ = self._construire_classes(X, labels)
        S = sum(self._R(x, labels[k], noyaux) for k, x in enumerate(X))
        return noyaux, self.cfg.max_iter, S

    def fit(self, X):
        if not X:
            raise ValueError("X est vide")
        d = len(X[0])
        if any(len(p) != d for p in X):
            raise ValueError("Dimensions incohérentes")
        self.cfg.valider(len(X), d)
        if self.cfg.random_state is not None:
            random.seed(self.cfg.random_state)
        X_work = self._normaliser(X) if self.cfg.normaliser else [list(p) for p in X]
        meilleur = None
        for _ in range(self.cfg.n_init):
            noyaux, n_iter, S = self._un_run(X_work)
            if meilleur is None or S < meilleur[2]:
                meilleur = (noyaux, n_iter, S)
        self.noyaux_, self.n_iter_, self.S_ = meilleur
        self.labels_ = self._affecter(X_work, self.noyaux_)
        self.classes_ = self._construire_classes(X_work, self.labels_)
        return self

    def predict(self, X):
        if self.noyaux_ is None:
            raise RuntimeError("Modèle non entraîné")
        X_work = X
        if self.cfg.normaliser and self.mu_ is not None:
            X_work = [[(p[k] - self.mu_[k]) / self.sigma_[k]
                       for k in range(len(p))] for p in X]
        return self._affecter(X_work, self.noyaux_)

    def resume(self):
        return {
            "config": asdict(self.cfg),
            "n_iter": self.n_iter_,
            "S": self.S_,
            "tailles_classes": [len(c) for c in self.classes_],
            "noyaux": [n.to_dict() for n in self.noyaux_],
        }

    def save(self, chemin):
        with open(chemin, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(chemin):
        with open(chemin, "rb") as f:
            return pickle.load(f)


# =============================================================
# Données
# =============================================================
def charger_csv(chemin, entete=True):
    X = []
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


def donnees_exemple(n1=120, n2=120, seed=0):
    random.seed(seed)
    X = []
    for _ in range(n1):
        t = random.uniform(0, 10)
        X.append([t + random.gauss(0, 0.25), 0.6 * t + random.gauss(0, 0.25)])
    for _ in range(n2):
        t = random.uniform(0, 10)
        X.append([t + random.gauss(0, 0.25),
                  -0.6 * t + 12 + random.gauss(0, 0.25)])
    for _ in range(60):
        X.append([random.gauss(5, 0.4), random.gauss(3, 0.4)])
    return X


# =============================================================
# Widgets personnalisés
# =============================================================
class ModernButton(tk.Frame):
    """Bouton stylisé avec effet de survol."""
    def __init__(self, master, text="", command=None,
                 kind="primary", icon=None, **kw):
        palette = {
            "primary": (COLORS["primary"], COLORS["primary_hover"], COLORS["primary_text"]),
            "success": (COLORS["success"], COLORS["success_hover"], "#ffffff"),
            "danger":  (COLORS["danger"],  COLORS["danger_hover"],  "#ffffff"),
            "ghost":   (COLORS["surface"], COLORS["surface_alt"],   COLORS["text"]),
            "dark":    (COLORS["sidebar_hover"], COLORS["sidebar"], COLORS["sidebar_text"]),
        }
        bg, hover, fg = palette.get(kind, palette["primary"])
        super().__init__(master, bg=bg, cursor="hand2", **kw)
        self._bg, self._hover, self._fg = bg, hover, fg
        self._command = command
        self._enabled = True

        self.label = tk.Label(self, text=(f"{icon}  {text}" if icon else text),
                              bg=bg, fg=fg, padx=16, pady=9,
                              font=(FONT_FAMILY, 10, "bold"))
        self.label.pack(fill="both", expand=True)

        for w in (self, self.label):
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)
            w.bind("<Button-1>", self._on_click)

    def _on_enter(self, _):
        if self._enabled:
            self.configure(bg=self._hover)
            self.label.configure(bg=self._hover)

    def _on_leave(self, _):
        if self._enabled:
            self.configure(bg=self._bg)
            self.label.configure(bg=self._bg)

    def _on_click(self, _):
        if self._enabled and self._command:
            self._command()

    def set_enabled(self, enabled):
        self._enabled = enabled
        state_bg = self._bg if enabled else COLORS["border"]
        state_fg = self._fg if enabled else COLORS["text_light"]
        self.configure(bg=state_bg, cursor="hand2" if enabled else "arrow")
        self.label.configure(bg=state_bg, fg=state_fg)

    def set_text(self, text):
        self.label.configure(text=text)


class NavButton(tk.Frame):
    """Élément de navigation dans la sidebar."""
    def __init__(self, master, text, icon, command):
        super().__init__(master, bg=COLORS["sidebar"], cursor="hand2")
        self._command = command
        self._active = False

        self.accent = tk.Frame(self, bg=COLORS["sidebar"], width=4)
        self.accent.pack(side="left", fill="y")

        self.icon_lbl = tk.Label(self, text=icon, bg=COLORS["sidebar"],
                                 fg=COLORS["sidebar_text"],
                                 font=(FONT_FAMILY, 13), width=3)
        self.icon_lbl.pack(side="left", pady=10)

        self.text_lbl = tk.Label(self, text=text, bg=COLORS["sidebar"],
                                 fg=COLORS["sidebar_text"], anchor="w",
                                 font=(FONT_FAMILY, 10, "bold"))
        self.text_lbl.pack(side="left", fill="x", expand=True, pady=10,
                           padx=(4, 12))

        for w in (self, self.icon_lbl, self.text_lbl):
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)
            w.bind("<Button-1>", lambda e: self._command())

    def _on_enter(self, _):
        if not self._active:
            for w in (self, self.icon_lbl, self.text_lbl):
                w.configure(bg=COLORS["sidebar_hover"])

    def _on_leave(self, _):
        if not self._active:
            for w in (self, self.icon_lbl, self.text_lbl):
                w.configure(bg=COLORS["sidebar"])

    def set_active(self, active: bool):
        self._active = active
        bg = COLORS["sidebar_hover"] if active else COLORS["sidebar"]
        fg = "#ffffff" if active else COLORS["sidebar_text"]
        accent = COLORS["primary"] if active else COLORS["sidebar"]
        self.configure(bg=bg)
        self.icon_lbl.configure(bg=bg, fg=fg)
        self.text_lbl.configure(bg=bg, fg=fg)
        self.accent.configure(bg=accent)


class Card(tk.Frame):
    """Carte avec titre et contenu."""
    def __init__(self, master, title=None, subtitle=None, **kw):
        super().__init__(master, bg=COLORS["surface"],
                         highlightbackground=COLORS["border"],
                         highlightthickness=1, **kw)
        if title:
            header = tk.Frame(self, bg=COLORS["surface"])
            header.pack(fill="x", padx=20, pady=(16, 8))
            tk.Label(header, text=title, bg=COLORS["surface"],
                     fg=COLORS["text"], font=(FONT_FAMILY, 12, "bold"),
                     anchor="w").pack(anchor="w")
            if subtitle:
                tk.Label(header, text=subtitle, bg=COLORS["surface"],
                         fg=COLORS["text_muted"],
                         font=(FONT_FAMILY, 9), anchor="w",
                         justify="left").pack(anchor="w", pady=(2, 0))
            tk.Frame(self, bg=COLORS["border"], height=1).pack(fill="x")
        self.body = tk.Frame(self, bg=COLORS["surface"])
        self.body.pack(fill="both", expand=True, padx=20, pady=16)


class FormField(tk.Frame):
    """Champ de formulaire avec label, widget et texte d'aide."""
    def __init__(self, master, label, widget_factory, aide=None):
        super().__init__(master, bg=COLORS["surface"])
        tk.Label(self, text=label, bg=COLORS["surface"], fg=COLORS["text"],
                 font=(FONT_FAMILY, 10, "bold"),
                 anchor="w").pack(fill="x", pady=(0, 4))
        self.widget = widget_factory(self)
        self.widget.pack(fill="x")
        if aide:
            tk.Label(self, text=aide, bg=COLORS["surface"],
                     fg=COLORS["text_muted"], font=(FONT_FAMILY, 8),
                     anchor="w", justify="left",
                     wraplength=400).pack(fill="x", pady=(4, 0))


# =============================================================
# APPLICATION
# =============================================================
class NuéesDynamiquesApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nuées Dynamiques — Diday (1971)")
        self.geometry("1440x900")
        self.minsize(1180, 760)
        self.configure(bg=COLORS["bg"])

        # Police système
        global FONT_FAMILY
        FONT_FAMILY = tkfont.nametofont("TkDefaultFont").actual("family")

        self.X: Optional[Dataset] = None
        self.model: Optional[NueesDynamiques] = None
        self._current_page = None
        self._nav_buttons = {}
        self._pages = {}
        self._tooltip_id = None

        # Variables de paramètres
        self.var_K = tk.IntVar(value=3)
        self.var_type = tk.StringVar(value="point")
        self.var_n1 = tk.IntVar(value=8)
        self.var_dist = tk.StringVar(value="sum")
        self.var_R = tk.StringVar(value="ex2")
        self.var_maxiter = tk.IntVar(value=100)
        self.var_tol = tk.StringVar(value="1e-6")
        self.var_ninit = tk.IntVar(value=3)
        self.var_norm = tk.BooleanVar(value=True)
        self.var_seed = tk.StringVar(value="42")
        self.var_show_noyaux = tk.BooleanVar(value=True)
        self.var_show_grid = tk.BooleanVar(value=True)
        self.var_show_legend = tk.BooleanVar(value=True)

        self._configurer_styles()
        self._build_layout()
        self._charger_exemple()
        self._show_page("donnees")

        # Raccourcis clavier
        self.bind("<Control-o>", lambda e: self._charger_csv())
        self.bind("<Control-s>", lambda e: self._sauvegarder())
        self.bind("<Control-r>", lambda e: self._run())
        self.bind("<F5>", lambda e: self._draw_plot())

    # ---------------------------------------------------------
    def _configurer_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TCombobox",
                        fieldbackground=COLORS["surface"],
                        background=COLORS["surface"],
                        foreground=COLORS["text"],
                        bordercolor=COLORS["border_strong"],
                        lightcolor=COLORS["border"],
                        darkcolor=COLORS["border"],
                        arrowcolor=COLORS["text_muted"],
                        padding=6)
        style.map("TCombobox",
                  fieldbackground=[("readonly", COLORS["surface"])],
                  bordercolor=[("focus", COLORS["primary"])])
        style.configure("TSpinbox",
                        fieldbackground=COLORS["surface"],
                        foreground=COLORS["text"],
                        bordercolor=COLORS["border_strong"],
                        arrowcolor=COLORS["text_muted"],
                        padding=6)
        style.configure("TEntry",
                        fieldbackground=COLORS["surface"],
                        bordercolor=COLORS["border_strong"],
                        padding=6)
        style.configure("TRadiobutton",
                        background=COLORS["surface"],
                        foreground=COLORS["text"],
                        font=(FONT_FAMILY, 10))
        style.map("TRadiobutton",
                  background=[("active", COLORS["surface"])])
        style.configure("TCheckbutton",
                        background=COLORS["surface"],
                        foreground=COLORS["text"],
                        font=(FONT_FAMILY, 10))
        style.map("TCheckbutton",
                  background=[("active", COLORS["surface"])])
        style.configure("Modern.Horizontal.TProgressbar",
                        troughcolor=COLORS["border"],
                        background=COLORS["primary"],
                        borderwidth=0,
                        thickness=6)

    # ---------------------------------------------------------
    # Layout principal
    # ---------------------------------------------------------
    def _build_layout(self):
        # Header
        self._build_header()

        # Container milieu
        milieu = tk.Frame(self, bg=COLORS["bg"])
        milieu.pack(fill="both", expand=True)

        self._build_sidebar(milieu)

        self.content = tk.Frame(milieu, bg=COLORS["bg"])
        self.content.pack(side="left", fill="both", expand=True)

        # Status bar
        self._build_status()

        # Pages
        self._pages["donnees"] = self._build_page_donnees()
        self._pages["parametres"] = self._build_page_parametres()
        self._pages["visualisation"] = self._build_page_visualisation()
        self._pages["resultats"] = self._build_page_resultats()

    def _build_header(self):
        header = tk.Frame(self, bg=COLORS["surface"], height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Frame(header, bg=COLORS["border"], height=1).pack(
            side="bottom", fill="x")

        gauche = tk.Frame(header, bg=COLORS["surface"])
        gauche.pack(side="left", fill="y", padx=20)

        # Logo stylisé
        logo = tk.Frame(gauche, bg=COLORS["primary"], width=36, height=36)
        logo.pack(side="left", pady=14)
        logo.pack_propagate(False)
        tk.Label(logo, text="◈", bg=COLORS["primary"], fg="#ffffff",
                 font=(FONT_FAMILY, 18, "bold")).pack(expand=True)

        titres = tk.Frame(gauche, bg=COLORS["surface"])
        titres.pack(side="left", padx=(12, 0), pady=14)
        tk.Label(titres, text="Nuées Dynamiques",
                 bg=COLORS["surface"], fg=COLORS["text"],
                 font=(FONT_FAMILY, 14, "bold"),
                 anchor="w").pack(anchor="w")
        tk.Label(titres, text="Classification automatique — Diday, 1971",
                 bg=COLORS["surface"], fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 9), anchor="w").pack(anchor="w")

        droite = tk.Frame(header, bg=COLORS["surface"])
        droite.pack(side="right", fill="y", padx=20, pady=14)

        self.stat_pill = tk.Frame(droite, bg=COLORS["accent_soft"],
                                  highlightbackground=COLORS["primary"],
                                  highlightthickness=1)
        self.stat_pill.pack(side="right")
        self.stat_label = tk.Label(self.stat_pill,
                                   text="  Aucune donnée  ",
                                   bg=COLORS["accent_soft"],
                                   fg=COLORS["primary"],
                                   font=(FONT_FAMILY, 9, "bold"),
                                   pady=6)
        self.stat_label.pack()

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=COLORS["sidebar"], width=230)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="NAVIGATION", bg=COLORS["sidebar"],
                 fg=COLORS["text_light"], font=(FONT_FAMILY, 8, "bold"),
                 anchor="w").pack(fill="x", padx=20, pady=(20, 8))

        items = [
            ("donnees",       "◫", "Données"),
            ("parametres",    "⚙", "Paramètres"),
            ("visualisation", "◉", "Visualisation"),
            ("resultats",     "▤", "Résultats"),
        ]
        for key, icon, label in items:
            btn = NavButton(sidebar, label, icon,
                            command=lambda k=key: self._show_page(k))
            btn.pack(fill="x")
            self._nav_buttons[key] = btn

        # Bas de sidebar : bouton principal
        tk.Frame(sidebar, bg=COLORS["sidebar"]).pack(
            fill="both", expand=True)

        pied = tk.Frame(sidebar, bg=COLORS["sidebar"])
        pied.pack(fill="x", side="bottom", pady=20, padx=16)

        self.btn_run_sidebar = ModernButton(
            pied, text="LANCER", icon="▶", kind="success",
            command=self._run)
        self.btn_run_sidebar.pack(fill="x")

        tk.Label(sidebar, text="Ctrl+O · Ctrl+S · Ctrl+R",
                 bg=COLORS["sidebar"], fg=COLORS["text_light"],
                 font=(FONT_FAMILY, 8)).pack(side="bottom", pady=(0, 4))

    def _build_status(self):
        bar = tk.Frame(self, bg=COLORS["surface"], height=32)
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=COLORS["border"], height=1).pack(side="top", fill="x")

        self.status_var = tk.StringVar(value="Prêt")
        tk.Label(bar, textvariable=self.status_var, bg=COLORS["surface"],
                 fg=COLORS["text_muted"], font=(FONT_FAMILY, 9),
                 anchor="w", padx=16).pack(side="left", fill="y")

        self.progress = ttk.Progressbar(bar, mode="indeterminate",
                                        length=160,
                                        style="Modern.Horizontal.TProgressbar")
        # Le progressbar apparaît seulement pendant l'entraînement

    # ---------------------------------------------------------
    # Navigation
    # ---------------------------------------------------------
    def _show_page(self, key):
        if self._current_page == key:
            return
        for k, page in self._pages.items():
            page.pack_forget()
        self._pages[key].pack(fill="both", expand=True, padx=20, pady=20)
        for k, btn in self._nav_buttons.items():
            btn.set_active(k == key)
        self._current_page = key

        if key == "visualisation":
            self.after(50, self._draw_plot)
        if key == "resultats":
            self._afficher_resultats()

    # ---------------------------------------------------------
    # Page : Données
    # ---------------------------------------------------------
    def _build_page_donnees(self):
        page = tk.Frame(self.content, bg=COLORS["bg"])

        # Titre de page
        self._page_header(page, "Données",
                          "Chargez votre jeu de données ou utilisez l'exemple.")

        grille = tk.Frame(page, bg=COLORS["bg"])
        grille.pack(fill="both", expand=True, pady=(16, 0))

        # Carte gauche : source
        card_src = Card(grille, title="Source des données",
                        subtitle="CSV : une ligne par individu, une colonne par variable.")
        card_src.pack(side="left", fill="both", expand=True, padx=(0, 12))

        actions = tk.Frame(card_src.body, bg=COLORS["surface"])
        actions.pack(fill="x")
        ModernButton(actions, text="Charger un CSV",
                     icon="⬆", kind="primary",
                     command=self._charger_csv).pack(side="left")
        ModernButton(actions, text="Exemple",
                     icon="✦", kind="ghost",
                     command=self._charger_exemple).pack(side="left", padx=8)

        self.lbl_donnees = tk.Label(card_src.body,
                                    text="Aucune donnée chargée.",
                                    bg=COLORS["surface"],
                                    fg=COLORS["text_muted"],
                                    font=(FONT_FAMILY, 10),
                                    anchor="w", justify="left")
        self.lbl_donnees.pack(fill="x", pady=(20, 12))

        tk.Label(card_src.body, text="Aperçu",
                 bg=COLORS["surface"], fg=COLORS["text"],
                 font=(FONT_FAMILY, 10, "bold"),
                 anchor="w").pack(fill="x", pady=(8, 4))

        txt_wrap = tk.Frame(card_src.body, bg=COLORS["surface"],
                            highlightbackground=COLORS["border"],
                            highlightthickness=1)
        txt_wrap.pack(fill="both", expand=True)
        self.txt_apercu = tk.Text(txt_wrap, height=10, font=("Courier", 9),
                                  bg=COLORS["surface_alt"],
                                  fg=COLORS["text"],
                                  relief="flat", bd=0, padx=10, pady=8)
        self.txt_apercu.pack(fill="both", expand=True)

        # Carte droite : infos
        card_info = Card(grille, title="Résumé",
                         subtitle="Caractéristiques du jeu de données chargé.")
        card_info.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.info_labels = {}
        for key, label in [("n", "Nombre d'individus"),
                           ("d", "Nombre de dimensions"),
                           ("source", "Source"),
                           ("min", "Minimum (global)"),
                           ("max", "Maximum (global)")]:
            self._ajouter_info_ligne(card_info.body, key, label)

        return page

    def _ajouter_info_ligne(self, parent, key, label):
        ligne = tk.Frame(parent, bg=COLORS["surface"])
        ligne.pack(fill="x", pady=6)
        tk.Label(ligne, text=label, bg=COLORS["surface"],
                 fg=COLORS["text_muted"], font=(FONT_FAMILY, 9),
                 anchor="w").pack(side="left")
        val = tk.Label(ligne, text="—", bg=COLORS["surface"],
                       fg=COLORS["text"], font=(FONT_FAMILY, 10, "bold"),
                       anchor="e")
        val.pack(side="right")
        self.info_labels[key] = val

    # ---------------------------------------------------------
    # Page : Paramètres
    # ---------------------------------------------------------
    def _build_page_parametres(self):
        page = tk.Frame(self.content, bg=COLORS["bg"])
        self._page_header(page, "Paramètres",
                          "L'utilisateur choisit explicitement le type de noyau.")

        grille = tk.Frame(page, bg=COLORS["bg"])
        grille.pack(fill="both", expand=True, pady=(16, 0))
        grille.columnconfigure(0, weight=1, uniform="col")
        grille.columnconfigure(1, weight=1, uniform="col")
        grille.rowconfigure(0, weight=1)

        # Colonne gauche : Noyau
        card_g = Card(grille, title="Noyau (représentant des nuées)",
                      subtitle="Choisissez la forme du représentant de chaque classe.")
        card_g.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        self._champ_K(card_g.body)
        self._champ_type(card_g.body)
        self._champ_n1(card_g.body)
        self._champ_dist(card_g.body)

        # Colonne droite : convergence + R
        card_d = Card(grille, title="Convergence et critère",
                      subtitle="Règle d'agrégation-écartement et paramètres d'arrêt.")
        card_d.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        self._champ_R(card_d.body)
        self._champ_convergence(card_d.body)
        self._champ_divers(card_d.body)

        # Aide dynamique en bas
        card_aide = Card(page, title="À propos du noyau sélectionné")
        card_aide.pack(fill="x", pady=(16, 0))
        self.txt_aide = tk.Label(card_aide.body, text="",
                                 bg=COLORS["surface"],
                                 fg=COLORS["text_muted"],
                                 font=(FONT_FAMILY, 10),
                                 anchor="w", justify="left",
                                 wraplength=1100)
        self.txt_aide.pack(fill="x")
        self.var_type.trace_add("write", lambda *_: self._update_aide())
        self._update_aide()

        return page

    def _champ_K(self, parent):
        def make(p):
            sb = ttk.Spinbox(p, from_=1, to=10000,
                             textvariable=self.var_K, width=10)
            return sb
        FormField(parent, "Nombre de classes (K)", make,
                  "Nombre de nuées à découvrir.").pack(fill="x", pady=8)

    def _champ_type(self, parent):
        def make(p):
            cb = ttk.Combobox(p, textvariable=self.var_type,
                              values=list(TYPES_NOYAUX.keys()),
                              state="readonly")
            cb.bind("<<ComboboxSelected>>",
                    lambda e: self._update_dependent_fields())
            return cb
        FormField(parent, "Type de noyau", make,
                  "  ·  ".join(f"{k} : {v}" for k, v in DESCRIPTIONS.items())
                  ).pack(fill="x", pady=8)

    def _champ_n1(self, parent):
        def make(p):
            self.sp_n1 = ttk.Spinbox(p, from_=1, to=10000,
                                     textvariable=self.var_n1, width=10)
            return self.sp_n1
        self.ff_n1 = FormField(parent, "Nombre d'étalons (n₁)", make,
                               "Actif pour 'ensemble_points' et "
                               "'structure_representative'.")
        self.ff_n1.pack(fill="x", pady=8)

    def _champ_dist(self, parent):
        def make(p):
            self.cb_dist = ttk.Combobox(p, textvariable=self.var_dist,
                                        values=["min", "sum", "mean"],
                                        state="readonly")
            return self.cb_dist
        self.ff_dist = FormField(parent, "Distance à un ensemble", make,
                                 "min : plus proche étalon  ·  "
                                 "sum : somme  ·  mean : moyenne.")
        self.ff_dist.pack(fill="x", pady=8)

    def _champ_R(self, parent):
        tk.Label(parent, text="Fonction d'agrégation-écartement R",
                 bg=COLORS["surface"], fg=COLORS["text"],
                 font=(FONT_FAMILY, 10, "bold"),
                 anchor="w").pack(fill="x", pady=(8, 4))
        for val, text in [("ex2", "ex2  :  R(x,i,L) = D(x, Cᵢ)"),
                          ("ex1", "ex1  :  R = D(x,Eᵢ)·D(x,Cᵢ) / (Σⱼ D(x,Eⱼ))²")]:
            rb = ttk.Radiobutton(parent, text=text, value=val,
                                 variable=self.var_R)
            rb.pack(anchor="w", pady=2)

    def _champ_convergence(self, parent):
        ligne = tk.Frame(parent, bg=COLORS["surface"])
        ligne.pack(fill="x", pady=8)

        g = tk.Frame(ligne, bg=COLORS["surface"])
        g.pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Label(g, text="Itérations max", bg=COLORS["surface"],
                 fg=COLORS["text"], font=(FONT_FAMILY, 10, "bold"),
                 anchor="w").pack(fill="x", pady=(0, 4))
        ttk.Spinbox(g, from_=1, to=100000, textvariable=self.var_maxiter,
                    width=10).pack(fill="x")

        d = tk.Frame(ligne, bg=COLORS["surface"])
        d.pack(side="left", fill="x", expand=True, padx=(6, 0))
        tk.Label(d, text="Tolérance", bg=COLORS["surface"],
                 fg=COLORS["text"], font=(FONT_FAMILY, 10, "bold"),
                 anchor="w").pack(fill="x", pady=(0, 4))
        ttk.Entry(d, textvariable=self.var_tol).pack(fill="x")

    def _champ_divers(self, parent):
        ligne = tk.Frame(parent, bg=COLORS["surface"])
        ligne.pack(fill="x", pady=8)

        g = tk.Frame(ligne, bg=COLORS["surface"])
        g.pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Label(g, text="Redémarrages (n_init)", bg=COLORS["surface"],
                 fg=COLORS["text"], font=(FONT_FAMILY, 10, "bold"),
                 anchor="w").pack(fill="x", pady=(0, 4))
        ttk.Spinbox(g, from_=1, to=100, textvariable=self.var_ninit,
                    width=10).pack(fill="x")

        d = tk.Frame(ligne, bg=COLORS["surface"])
        d.pack(side="left", fill="x", expand=True, padx=(6, 0))
        tk.Label(d, text="Graine (vide = aléatoire)", bg=COLORS["surface"],
                 fg=COLORS["text"], font=(FONT_FAMILY, 10, "bold"),
                 anchor="w").pack(fill="x", pady=(0, 4))
        ttk.Entry(d, textvariable=self.var_seed).pack(fill="x")

        ttk.Checkbutton(parent, text="Normaliser les données (z-score)",
                        variable=self.var_norm).pack(anchor="w", pady=(10, 0))

    def _update_dependent_fields(self):
        t = self.var_type.get()
        besoin = t in ("ensemble_points", "structure_representative")
        for w in (self.sp_n1, self.cb_dist):
            try:
                w.configure(state=("normal" if w is self.sp_n1 else
                                   ("readonly" if besoin else "disabled")))
                if not besoin:
                    w.configure(state="disabled")
                elif w is self.cb_dist:
                    w.configure(state="readonly")
            except Exception:
                pass
        self._update_aide()

    def _update_aide(self):
        self.txt_aide.configure(text=AIDES.get(self.var_type.get(), ""))

    # ---------------------------------------------------------
    # Page : Visualisation
    # ---------------------------------------------------------
    def _build_page_visualisation(self):
        page = tk.Frame(self.content, bg=COLORS["bg"])
        self._page_header(page, "Visualisation",
                          "Vue 2D interactive avec noyaux et classes.")

        # Barre d'outils
        barre = tk.Frame(page, bg=COLORS["bg"])
        barre.pack(fill="x", pady=(16, 12))

        ttk.Checkbutton(barre, text="Noyaux",
                        variable=self.var_show_noyaux,
                        command=self._draw_plot).pack(side="left")
        ttk.Checkbutton(barre, text="Grille",
                        variable=self.var_show_grid,
                        command=self._draw_plot).pack(side="left", padx=12)
        ttk.Checkbutton(barre, text="Légende",
                        variable=self.var_show_legend,
                        command=self._draw_plot).pack(side="left", padx=12)

        self.lbl_coords = tk.Label(barre, text="", bg=COLORS["bg"],
                                   fg=COLORS["text_muted"],
                                   font=(FONT_FAMILY, 9))
        self.lbl_coords.pack(side="right")

        # Canvas
        cadre = tk.Frame(page, bg=COLORS["surface"],
                         highlightbackground=COLORS["border"],
                         highlightthickness=1)
        cadre.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(cadre, background=COLORS["canvas_bg"],
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=1, pady=1)
        self.canvas.bind("<Configure>", lambda e: self._draw_plot())
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self._hide_tooltip())

        return page

    # ---------------------------------------------------------
    # Page : Résultats
    # ---------------------------------------------------------
    def _build_page_resultats(self):
        page = tk.Frame(self.content, bg=COLORS["bg"])
        self._page_header(page, "Résultats",
                          "Synthèse de la partition et des noyaux obtenus.")

        # Cartes stats
        self.stats_frame = tk.Frame(page, bg=COLORS["bg"])
        self.stats_frame.pack(fill="x", pady=(16, 12))

        self.stat_cards = {}
        for key, titre in [("K", "Classes"), ("n_iter", "Itérations"),
                           ("S", "Critère S"), ("total", "Individus")]:
            c = tk.Frame(self.stats_frame, bg=COLORS["surface"],
                         highlightbackground=COLORS["border"],
                         highlightthickness=1)
            c.pack(side="left", fill="x", expand=True,
                   padx=(0 if key == "K" else 8, 0))
            tk.Label(c, text=titre.upper(), bg=COLORS["surface"],
                     fg=COLORS["text_muted"],
                     font=(FONT_FAMILY, 8, "bold")).pack(
                anchor="w", padx=16, pady=(12, 0))
            val = tk.Label(c, text="—", bg=COLORS["surface"],
                           fg=COLORS["primary"],
                           font=(FONT_FAMILY, 20, "bold"))
            val.pack(anchor="w", padx=16, pady=(2, 12))
            self.stat_cards[key] = val

        # Zone scrollable pour les noyaux
        cadre = tk.Frame(page, bg=COLORS["bg"])
        cadre.pack(fill="both", expand=True)
        self.resultats_canvas = tk.Canvas(cadre, bg=COLORS["bg"],
                                          highlightthickness=0)
        vsb = ttk.Scrollbar(cadre, orient="vertical",
                            command=self.resultats_canvas.yview)
        self.resultats_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.resultats_canvas.pack(side="left", fill="both", expand=True)

        self.resultats_inner = tk.Frame(self.resultats_canvas,
                                        bg=COLORS["bg"])
        self.resultats_canvas.create_window((0, 0),
                                            window=self.resultats_inner,
                                            anchor="nw")
        self.resultats_inner.bind(
            "<Configure>",
            lambda e: self.resultats_canvas.configure(
                scrollregion=self.resultats_canvas.bbox("all")))

        return page

    # ---------------------------------------------------------
    def _page_header(self, parent, titre, sous_titre):
        h = tk.Frame(parent, bg=COLORS["bg"])
        h.pack(fill="x")
        tk.Label(h, text=titre, bg=COLORS["bg"], fg=COLORS["text"],
                 font=(FONT_FAMILY, 22, "bold"),
                 anchor="w").pack(anchor="w")
        tk.Label(h, text=sous_titre, bg=COLORS["bg"],
                 fg=COLORS["text_muted"], font=(FONT_FAMILY, 10),
                 anchor="w").pack(anchor="w", pady=(2, 0))

    # ---------------------------------------------------------
    # Données
    # ---------------------------------------------------------
    def _charger_exemple(self):
        self.X = donnees_exemple()
        self._maj_infos_donnees(source="Exemple synthétique")

    def _charger_csv(self):
        chemin = filedialog.askopenfilename(
            title="Choisir un fichier CSV",
            filetypes=[("CSV", "*.csv"), ("Tous", "*.*")])
        if not chemin:
            return
        try:
            X = charger_csv(chemin, entete=True)
            if not X:
                raise ValueError("Fichier vide ou invalide")
            self.X = X
            self._maj_infos_donnees(source=chemin.split("/")[-1])
            self.status_var.set(f"{len(X)} individus chargés.")
        except Exception as e:
            messagebox.showerror("Erreur de chargement", str(e))

    def _maj_infos_donnees(self, source=""):
        if not self.X:
            return
        n, d = len(self.X), len(self.X[0])
        self.lbl_donnees.configure(
            text=f"{n} individus × {d} variables.")
        self.info_labels["n"].configure(text=str(n))
        self.info_labels["d"].configure(text=str(d))
        self.info_labels["source"].configure(text=source or "—")
        toutes = [v for p in self.X for v in p]
        self.info_labels["min"].configure(text=f"{min(toutes):.3f}")
        self.info_labels["max"].configure(text=f"{max(toutes):.3f}")

        # Aperçu
        self.txt_apercu.configure(state="normal")
        self.txt_apercu.delete("1.0", "end")
        for p in self.X[:8]:
            self.txt_apercu.insert("end", "  ".join(f"{v:>10.4f}" for v in p) + "\n")
        if len(self.X) > 8:
            self.txt_apercu.insert(
                "end", f"... ({len(self.X) - 8} lignes supplémentaires)\n")
        self.txt_apercu.configure(state="disabled")

        self.stat_label.configure(text=f"  {n} individus · {d}D  ")
        self.status_var.set(f"Données prêtes : {n} individus, {d} dimensions.")

    # ---------------------------------------------------------
    # Apprentissage
    # ---------------------------------------------------------
    def _run(self):
        if not self.X:
            messagebox.showwarning("Aucune donnée",
                                   "Chargez d'abord des données.")
            return
        try:
            seed_str = self.var_seed.get().strip()
            seed = int(seed_str) if seed_str else None
            cfg = Config(
                n_clusters=self.var_K.get(),
                type_noyau=self.var_type.get(),
                n_etalons=self.var_n1.get(),
                distance_to_set=self.var_dist.get(),
                R=self.var_R.get(),
                max_iter=self.var_maxiter.get(),
                tol=float(self.var_tol.get()),
                n_init=self.var_ninit.get(),
                normaliser=self.var_norm.get(),
                random_state=seed,
            )
            cfg.valider(len(self.X), len(self.X[0]))
        except Exception as e:
            messagebox.showerror("Paramètres invalides", str(e))
            return

        self._start_progress()
        self.update_idletasks()

        try:
            self.model = NueesDynamiques(cfg)
            self.model.fit(self.X)
        except Exception as e:
            self._stop_progress()
            messagebox.showerror("Erreur", str(e))
            return

        self._stop_progress()
        self.status_var.set(
            f"Terminé : {self.model.n_iter_} itérations, "
            f"S = {self.model.S_:.4f}")
        self._afficher_resultats()
        self._draw_plot()
        self._show_page("visualisation")

    def _start_progress(self):
        self.progress.pack(side="right", padx=16, pady=8)
        self.progress.start(12)
        self.btn_run_sidebar.set_enabled(False)

    def _stop_progress(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.btn_run_sidebar.set_enabled(True)

    # ---------------------------------------------------------
    # Résultats
    # ---------------------------------------------------------
    def _afficher_resultats(self):
        for w in self.resultats_inner.winfo_children():
            w.destroy()
        if not self.model:
            return

        info = self.model.resume()
        cfg = info["config"]

        self.stat_cards["K"].configure(text=str(cfg["n_clusters"]))
        self.stat_cards["n_iter"].configure(text=str(info["n_iter"]))
        self.stat_cards["S"].configure(text=f"{info['S']:.3f}")
        self.stat_cards["total"].configure(text=str(len(self.X)))

        # Carte config
        card_cfg = Card(self.resultats_inner, title="Configuration utilisée")
        card_cfg.pack(fill="x", pady=(0, 12))
        grid = tk.Frame(card_cfg.body, bg=COLORS["surface"])
        grid.pack(fill="x")
        for i, (k, v) in enumerate(cfg.items()):
            col = i % 3
            ligne = i // 3
            cell = tk.Frame(grid, bg=COLORS["surface"])
            cell.grid(row=ligne, column=col, sticky="w", padx=(0, 40), pady=4)
            tk.Label(cell, text=k, bg=COLORS["surface"],
                     fg=COLORS["text_muted"], font=(FONT_FAMILY, 9),
                     anchor="w").pack(anchor="w")
            tk.Label(cell, text=str(v), bg=COLORS["surface"],
                     fg=COLORS["text"], font=(FONT_FAMILY, 10, "bold"),
                     anchor="w").pack(anchor="w")

        # Cartes noyaux
        for i, n in enumerate(info["noyaux"]):
            self._carte_noyau(i, n, info["tailles_classes"][i])

    def _carte_noyau(self, idx, noyau, taille):
        color = PALETTE[idx % len(PALETTE)]
        carte = tk.Frame(self.resultats_inner, bg=COLORS["surface"],
                         highlightbackground=COLORS["border"],
                         highlightthickness=1)
        carte.pack(fill="x", pady=(0, 10))

        # Accent coloré gauche
        tk.Frame(carte, bg=color, width=6).pack(side="left", fill="y")

        body = tk.Frame(carte, bg=COLORS["surface"])
        body.pack(side="left", fill="both", expand=True, padx=16, pady=14)

        # Entête
        top = tk.Frame(body, bg=COLORS["surface"])
        top.pack(fill="x")
        tk.Label(top, text=f"Noyau {idx}", bg=COLORS["surface"],
                 fg=COLORS["text"],
                 font=(FONT_FAMILY, 12, "bold")).pack(side="left")
        tk.Label(top, text=f"  {noyau['type']}  ",
                 bg=COLORS["accent_soft"], fg=COLORS["primary"],
                 font=(FONT_FAMILY, 9, "bold"),
                 padx=8, pady=2).pack(side="left", padx=8)

        tk.Label(top, text=f"{taille} individus", bg=COLORS["surface"],
                 fg=COLORS["text_muted"],
                 font=(FONT_FAMILY, 10)).pack(side="right")

        # Détails
        t = noyau["type"]
        if t == "point":
            self._ligne_info(body, "Point",
                             self._fmt(noyau["point"]))
        elif t == "axe_factoriel":
            self._ligne_info(body, "Centre", self._fmt(noyau["centre"]))
            self._ligne_info(body, "Direction", self._fmt(noyau["direction"]))
        elif t == "distribution":
            self._ligne_info(body, "Centre", self._fmt(noyau["centre"]))
            self._ligne_info(body, "Écarts-types", self._fmt(noyau["ecarts_types"]))
        elif t in ("ensemble_points", "structure_representative"):
            self._ligne_info(body, "Nombre d'étalons", str(noyau["n"]))
            for j, p in enumerate(noyau["points"][:4]):
                self._ligne_info(body, f"Étalon {j + 1}", self._fmt(p))
            if noyau["n"] > 4:
                self._ligne_info(body, "...",
                                 f"+ {noyau['n'] - 4} autres")

    def _ligne_info(self, parent, label, valeur):
        ligne = tk.Frame(parent, bg=COLORS["surface"])
        ligne.pack(fill="x", pady=2)
        tk.Label(ligne, text=label, bg=COLORS["surface"],
                 fg=COLORS["text_muted"], font=(FONT_FAMILY, 9),
                 anchor="w", width=18).pack(side="left")
        tk.Label(ligne, text=valeur, bg=COLORS["surface"],
                 fg=COLORS["text"], font=("Courier", 9),
                 anchor="w").pack(side="left")

    @staticmethod
    def _fmt(vec):
        return "[" + ", ".join(f"{v:.3f}" for v in vec) + "]"

    # ---------------------------------------------------------
    # Visualisation canvas
    # ---------------------------------------------------------
    def _draw_plot(self):
        c = self.canvas
        c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 20 or H < 20:
            return

        if not self.X:
            c.create_text(W // 2, H // 2, text="Chargez des données",
                          fill=COLORS["text_muted"],
                          font=(FONT_FAMILY, 13))
            return

        d = len(self.X[0])
        if d != 2:
            c.create_text(W // 2, H // 2,
                          text=f"Visualisation 2D uniquement\n"
                               f"(données en {d} dimensions)",
                          fill=COLORS["text_muted"],
                          font=(FONT_FAMILY, 13), justify="center")
            return

        margin = 60
        xs = [p[0] for p in self.X]
        ys = [p[1] for p in self.X]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        dx = (xmax - xmin) or 1.0
        dy = (ymax - ymin) or 1.0
        xmin -= 0.06 * dx; xmax += 0.06 * dx
        ymin -= 0.06 * dy; ymax += 0.06 * dy
        dx = xmax - xmin; dy = ymax - ymin

        self._plot_bounds = (xmin, xmax, ymin, ymax, margin, W, H)

        def to_canvas(x, y):
            cx = margin + (x - xmin) / dx * (W - 2 * margin)
            cy = H - margin - (y - ymin) / dy * (H - 2 * margin)
            return cx, cy

        # Grille
        if self.var_show_grid.get():
            for i in range(1, 12):
                gx = margin + (W - 2 * margin) * i / 12
                gy = margin + (H - 2 * margin) * i / 12
                c.create_line(gx, margin, gx, H - margin, fill=COLORS["grid"])
                c.create_line(margin, gy, W - margin, gy, fill=COLORS["grid"])

        # Cadre
        c.create_rectangle(margin, margin, W - margin, H - margin,
                           outline=COLORS["border_strong"], width=1)

        # Axes gradués
        for i in range(6):
            x_val = xmin + dx * i / 5
            gx = margin + (W - 2 * margin) * i / 5
            c.create_text(gx, H - margin + 16,
                          text=f"{x_val:.1f}", fill=COLORS["text_muted"],
                          font=(FONT_FAMILY, 8))
            y_val = ymin + dy * i / 5
            gy = H - margin - (H - 2 * margin) * i / 5
            c.create_text(margin - 18, gy, text=f"{y_val:.1f}",
                          fill=COLORS["text_muted"],
                          font=(FONT_FAMILY, 8))

        # Points
        labels = self.model.labels_ if self.model else [0] * len(self.X)
        self._point_index = {}
        for i, ((x, y), lab) in enumerate(zip(self.X, labels)):
            cx, cy = to_canvas(x, y)
            color = PALETTE[lab % len(PALETTE)]
            r = 4
            pid = c.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=color, outline="white", width=1)
            self._point_index[pid] = i

        # Noyaux
        if self.model and self.var_show_noyaux.get():
            for i, noyau in enumerate(self.model.noyaux_):
                self._draw_noyau(c, noyau, i, to_canvas, max(dx, dy))

        # Légende
        if self.var_show_legend.get() and self.model:
            self._draw_legend(c, W, H)

    def _draw_noyau(self, c, noyau, idx, to_canvas, scale):
        color = PALETTE[idx % len(PALETTE)]
        if isinstance(noyau, NoyauPoint):
            if noyau.point:
                cx, cy = to_canvas(noyau.point[0], noyau.point[1])
                c.create_polygon(cx, cy - 9, cx + 9, cy, cx, cy + 9, cx - 9, cy,
                                 outline="white", fill=color, width=2)

        elif isinstance(noyau, NoyauEnsemblePoints):
            for p in noyau.points:
                cx, cy = to_canvas(p[0], p[1])
                c.create_polygon(cx, cy - 7, cx + 7, cy, cx, cy + 7, cx - 7, cy,
                                 outline="white", fill=color, width=2)

        elif isinstance(noyau, NoyauAxeFactoriel):
            if noyau.centre and noyau.direction:
                L = scale * 0.45
                p1 = [noyau.centre[0] + L * noyau.direction[0],
                      noyau.centre[1] + L * noyau.direction[1]]
                p2 = [noyau.centre[0] - L * noyau.direction[0],
                      noyau.centre[1] - L * noyau.direction[1]]
                c1 = to_canvas(*p1); c2 = to_canvas(*p2)
                c.create_line(c1[0], c1[1], c2[0], c2[1],
                              fill=color, width=4, capstyle="round")
                for (cx, cy) in (c1, c2):
                    c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5,
                                  fill=color, outline="white", width=2)

        elif isinstance(noyau, NoyauDistribution):
            if noyau.centre and noyau.ecarts_types:
                pts = []
                for k in range(40):
                    ang = 2 * math.pi * k / 40
                    px = noyau.centre[0] + noyau.ecarts_types[0] * math.cos(ang)
                    py = noyau.centre[1] + noyau.ecarts_types[1] * math.sin(ang)
                    pts.extend(to_canvas(px, py))
                c.create_polygon(*pts, outline=color, fill="", width=3,
                                 smooth=True)
                cx, cy = to_canvas(noyau.centre[0], noyau.centre[1])
                c.create_polygon(cx, cy - 8, cx + 8, cy, cx, cy + 8, cx - 8, cy,
                                 outline="white", fill=color, width=2)

        elif isinstance(noyau, NoyauStructureRepresentative):
            if len(noyau.points) >= 2:
                coords = []
                for p in noyau.points:
                    coords.extend(to_canvas(p[0], p[1]))
                c.create_line(*coords, fill=color, width=3, smooth=True,
                              capstyle="round")
            for p in noyau.points:
                cx, cy = to_canvas(p[0], p[1])
                c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5,
                              outline="white", fill=color, width=2)

    def _draw_legend(self, c, W, H):
        x0 = W - 220
        y0 = 80
        padding = 12
        entries = []
        for i in range(len(self.model.noyaux_)):
            taille = len(self.model.classes_[i])
            entries.append((i, taille))
        hauteur = padding * 2 + 20 + 22 * len(entries)

        c.create_rectangle(x0, y0, W - 20, y0 + hauteur,
                           fill=COLORS["surface"],
                           outline=COLORS["border_strong"], width=1)
        c.create_text(x0 + padding, y0 + padding,
                      text="CLASSES", anchor="nw",
                      fill=COLORS["text_muted"],
                      font=(FONT_FAMILY, 8, "bold"))

        for i, (idx, taille) in enumerate(entries):
            yy = y0 + padding + 22 + 22 * i
            color = PALETTE[idx % len(PALETTE)]
            c.create_oval(x0 + padding, yy + 5, x0 + padding + 10, yy + 15,
                          fill=color, outline="white")
            c.create_text(x0 + padding + 20, yy + 10,
                          text=f"Classe {idx}  ·  {taille} ind.",
                          anchor="w", fill=COLORS["text"],
                          font=(FONT_FAMILY, 9))

    # ---------------------------------------------------------
    # Tooltip
    # ---------------------------------------------------------
    def _on_motion(self, event):
        if not hasattr(self, "_plot_bounds") or not self.X:
            return
        xmin, xmax, ymin, ymax, margin, W, H = self._plot_bounds
        dx = xmax - xmin; dy = ymax - ymin
        # Cherche le point le plus proche en pixels
        best_dist = 12
        best_i = None
        for pid, i in getattr(self, "_point_index", {}).items():
            coords = self.canvas.coords(pid)
            if not coords:
                continue
            cx, cy = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
            dd = math.hypot(cx - event.x, cy - event.y)
            if dd < best_dist:
                best_dist, best_i = dd, i

        self._hide_tooltip()
        if best_i is None:
            self.lbl_coords.configure(text="")
            return

        x, y = self.X[best_i]
        lab = self.model.labels_[best_i] if self.model else "?"
        self.lbl_coords.configure(
            text=f"Individu #{best_i}  ·  ({x:.3f}, {y:.3f})  ·  Classe {lab}")

        # Infobulle
        tx, ty = event.x + 14, event.y + 14
        rect = self.canvas.create_rectangle(
            tx, ty, tx + 160, ty + 48, fill=COLORS["sidebar"],
            outline="", tags="tooltip")
        t1 = self.canvas.create_text(
            tx + 10, ty + 14, text=f"Individu #{best_i}", anchor="w",
            fill="#ffffff", font=(FONT_FAMILY, 9, "bold"), tags="tooltip")
        t2 = self.canvas.create_text(
            tx + 10, ty + 32,
            text=f"({x:.3f}, {y:.3f})  ·  Classe {lab}", anchor="w",
            fill=COLORS["sidebar_text"], font=(FONT_FAMILY, 8),
            tags="tooltip")
        self._tooltip_id = (rect, t1, t2)

    def _hide_tooltip(self):
        if self._tooltip_id:
            for i in self._tooltip_id:
                self.canvas.delete(i)
            self._tooltip_id = None

    # ---------------------------------------------------------
    # Fichier
    # ---------------------------------------------------------
    def _sauvegarder(self):
        if not self.model:
            messagebox.showwarning("Aucun modèle",
                                   "Lancez d'abord un apprentissage.")
            return
        chemin = filedialog.asksaveasfilename(
            title="Sauvegarder le modèle",
            defaultextension=".pkl",
            filetypes=[("Pickle", "*.pkl")])
        if not chemin:
            return
        try:
            self.model.save(chemin)
            self.status_var.set(f"Modèle sauvegardé : {chemin}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _charger_modele(self):
        chemin = filedialog.askopenfilename(
            title="Charger un modèle",
            filetypes=[("Pickle", "*.pkl"), ("Tous", "*.*")])
        if not chemin:
            return
        try:
            self.model = NueesDynamiques.load(chemin)
            self._afficher_resultats()
            self._draw_plot()
            self.status_var.set(f"Modèle chargé : {chemin}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))


# =============================================================
def main():
    app = NuéesDynamiquesApp()
    app.mainloop()


if __name__ == "__main__":
    main()