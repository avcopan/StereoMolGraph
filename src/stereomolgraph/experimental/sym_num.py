from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Literal, TypeVar

from typing_extensions import Self

from stereomolgraph import AtomId, Bond, StereoMolGraph
from stereomolgraph.algorithms.circular import color_refine_smg
from stereomolgraph.algorithms.isomorphism import vf2pp_all_isomorphisms
from stereomolgraph.stereodescriptors import (
    AtropBond,
    OInt,
    PlanarBond,
    Tetrahedral,
    _StereoMixin,
)


def topological_symmetry_number(graph: StereoMolGraph, atom_labels=None) -> int:
    """
    Calculated from the number of graph isomorphisms which conserve the
    stereo information.
    symmetry_number = internal_symmetry_number * rotational_symmetry_number
    """
    if atom_labels is None:
        atom_labels = color_refine_smg(graph, atom_labels=atom_labels)

    mappings = vf2pp_all_isomorphisms(
        graph, graph, stereo=True, atom_labels=(atom_labels, atom_labels)
    )
    return deque(enumerate(mappings, 1), maxlen=1)[0][0]


class HinderedBond33(
    _StereoMixin[
        tuple[OInt, OInt, OInt, int, int, OInt, OInt, OInt], None | Literal[1, -1]
    ],
):
    r"""
    Represents a bond that cannot freely rotate::

            parity = 1
             0    5
             |    |
        1  ▷ 3 - 4 ◁ 6
            ◀     ▶
           2        7

            parity = -1
             0    5
             |    |
        1  ▷ 3 - 4 ◁ 7
            ◀     ▶
           2        6


    """

    parity = 0
    inversion = (0, 2, 1, 3, 4, 5, 7, 6)
    _bond: Bond
    PERMUTATION_GROUP = (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (5, 7, 6, 4, 3, 0, 2, 1),
        (1, 2, 0, 3, 4, 6, 7, 5),
        (6, 5, 7, 4, 3, 1, 0, 2),
        (2, 0, 1, 3, 4, 7, 5, 6),
        (7, 6, 5, 4, 3, 2, 1, 0),
    )

    def get_isomers(self) -> set[Self]:
        return {self}

    @property
    def bond(self) -> Bond:
        bond = frozenset(self.atoms[3:5])
        assert len(bond) == 2
        return bond


class HinderedBond23(
    _StereoMixin[tuple[OInt, OInt, int, int, OInt, OInt, OInt], None | Literal[1, -1]],
):
    r"""
    Represents a bond that cannot freely rotate::
            parity = 1
           0     4
            \    |
             2 - 3 ◁ 5
            /     ▶
           1        6

            parity = -1
           0     4
            \    |
             2 - 3 ◁ 6
            /     ▶
           1        5
    """

    parity = 0
    inversion = (0, 1, 2, 3, 4, 6, 5)
    _bond: Bond
    PERMUTATION_GROUP = ((0, 1, 2, 3, 4, 5, 6),)

    def get_isomers(self) -> set[Self]:
        return {self}

    @property
    def bond(self) -> Bond:
        bond = frozenset(self.atoms[2:4])
        assert len(bond) == 2
        return bond


class HinderedBond13(
    _StereoMixin[tuple[OInt, int, int, OInt, OInt, OInt], None | Literal[1, -1]],
):
    r"""
    Represents a bond that cannot freely rotate::
            parity = 1
          0     3
           \    |
            1 - 2 ◁ 4
                  ▶
                   5

            parity = -1
          0     3
           \    |
            1 - 2 ◁ 5
                  ▶
                   4
    """

    parity = 0
    inversion = (0, 1, 2, 3, 5, 4)
    _bond: Bond
    PERMUTATION_GROUP = ((0, 1, 2, 3, 4, 5),)

    def get_isomers(self) -> set[Self]:
        return {self}

    @property
    def bond(self) -> Bond:
        bond = frozenset(self.atoms[1:3])
        assert len(bond) == 2
        return bond


HinderedBond = HinderedBond33 | HinderedBond23 | HinderedBond13 | PlanarBond | AtropBond

ItemT = TypeVar("ItemT", bound=Hashable)


def get_automorphism_classes(
    items: Iterable[ItemT],
    mappings: Iterable[Mapping[AtomId, AtomId]],
    map_item: Callable[[ItemT, Mapping[AtomId, AtomId]], ItemT],
) -> dict[ItemT, set[ItemT]]:
    items = tuple(items)

    parent: dict[ItemT, ItemT] = {item: item for item in items}
    rank: dict[ItemT, int] = {item: 0 for item in items}

    def find(item: ItemT) -> ItemT:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    def union(item1: ItemT, item2: ItemT) -> None:
        root1 = find(item1)
        root2 = find(item2)
        if root1 == root2:
            return
        if rank[root1] < rank[root2]:
            root1, root2 = root2, root1
        parent[root2] = root1
        if rank[root1] == rank[root2]:
            rank[root1] += 1

    for mapping in mappings:
        for item in items:
            mapped_item = map_item(item, mapping)
            if mapped_item in parent:
                union(item, mapped_item)

    classes: dict[ItemT, set[ItemT]] = {}
    for item in items:
        root = find(item)
        classes.setdefault(root, set()).add(item)

    return {item: set(classes[find(item)]) for item in items}


def get_atom_automorphism_classes(
    smg: StereoMolGraph,
    mappings: Iterable[Mapping[AtomId, AtomId]] | None = None,
) -> dict[AtomId, set[AtomId]]:
    if mappings is None:
        mappings = vf2pp_all_isomorphisms(smg, smg, stereo=True)

    return get_automorphism_classes(
        smg.atoms,
        mappings,
        lambda atom, mapping: mapping[atom],
    )


def get_bond_automorphism_classes(
    smg: StereoMolGraph,
    mappings: Iterable[Mapping[AtomId, AtomId]] | None = None,
) -> dict[Bond, set[Bond]]:
    """
    Bonds are only considered if both of its atoms have other substituents
    and therefore can have a internal rotation.
    """
    if mappings is None:
        mappings = vf2pp_all_isomorphisms(smg, smg, stereo=True)

    classes = get_automorphism_classes(
        {bond for bond in smg.bonds},
        mappings,
        lambda bond, mapping: Bond(mapping[atom] for atom in bond),
    )

    return classes


def bond_symmetry_number(
    graph: StereoMolGraph, bond: Bond, mappings=None | Mapping[int, int]
) -> int:
    mappings = vf2pp_all_isomorphisms(graph, graph, stereo=True)

    a1, a2 = bond

    nbrs1 = tuple(a for a in graph.bonded_to(a1) if a != a2)
    nbrs2 = tuple(a for a in graph.bonded_to(a2) if a != a1)

    if len(nbrs1) == 0 or len(nbrs2) == 0:
        return 1

    if len(nbrs1) > len(nbrs2):
        a1, a2 = a2, a1
        nbrs1, nbrs2 = nbrs2, nbrs1

    if len(nbrs1) == 3 and len(nbrs2) == 3:
        s = HinderedBond33(atoms=(*nbrs1, a1, a2, *nbrs2), parity=1)
    elif len(nbrs1) == 2 and len(nbrs2) == 3:
        s = HinderedBond23(atoms=(*nbrs1, a1, a2, *nbrs2), parity=1)
    elif len(nbrs1) == 1 and len(nbrs2) == 3:
        s = HinderedBond13(atoms=(*nbrs1, a1, a2, *nbrs2), parity=1)
    elif len(nbrs1) == 2 and len(nbrs2) == 2:
        s = PlanarBond(atoms=(*nbrs1, a1, a2, *nbrs2), parity=0)

    unique_reorderings: set[HinderedBond] = set()
    bond_class = s.__class__

    for mapping in mappings:
        if mapping[a1] != a1 or mapping[a2] != a2:
            continue
        map_nrbrs1 = tuple(mapping[nbr] for nbr in nbrs1)
        map_nrbrs2 = tuple(mapping[nbr] for nbr in nbrs2)

        if any(map_nrbr not in nbrs1 for map_nrbr in map_nrbrs1) or any(
            map_nrbr not in nbrs2 for map_nrbr in map_nrbrs2
        ):
            continue

        reordering = bond_class(atoms=(*map_nrbrs1, a1, a2, *map_nrbrs2), parity=1)
        unique_reorderings.add(reordering)

    return len(unique_reorderings)


def ext_sym_num(
    graph: StereoMolGraph, mappings: Iterable[dict[int, int]] | None = None
) -> int:
    """Calculate the upper bound of the external symmetry number for StereoMolGraph"""

    # if any(stereo.parity is None for stereo in graph.stereo.values()):
    #    raise ValueError("Stereodescriptor parity has to be assigned")

    # non hindered bonds will be added to the graph.
    graph = graph.copy()

    if mappings is None:
        mappings = vf2pp_all_isomorphisms(g1=graph, g2=graph, stereo=True)
    mappings: tuple[dict[int | None, int | None], ...] = tuple(mappings)
    for mapping in mappings:
        mapping[None] = None

    atom_eq_classes = get_atom_automorphism_classes(graph, mappings=mappings)
    bond_eq_classes = get_bond_automorphism_classes(graph, mappings=mappings)

    for eq_cls in {frozenset(eq_cls) for eq_cls in bond_eq_classes.values()}:
        eq_cls_iterator = iter(eq_cls)
        a1, a2 = next(eq_cls_iterator)

        if graph.get_bond_stereo({a1, a2}) is not None:
            continue

        nbrs1 = tuple(a for a in graph.bonded_to(a1) if a != a2)
        nbrs2 = tuple(a for a in graph.bonded_to(a2) if a != a1)

        if len(nbrs1) > len(nbrs2):
            a1, a2 = a2, a1
            nbrs1, nbrs2 = nbrs2, nbrs1

        if len(nbrs1) == 0 or len(nbrs1) == 1:
            continue

        stereo1 = graph.get_atom_stereo(a1)
        stereo2 = graph.get_atom_stereo(a2)

        if isinstance(stereo1, Tetrahedral) and isinstance(stereo2, Tetrahedral):
            eq_cls1 = []
            for a in set(stereo1.atoms[1:6]) - {a2}:
                for e in eq_cls1:
                    if any(a in eq_cls1[b] for b in e):
                        e.add(a)
                        break
                else:  # if for loop is not broken
                    eq_cls1.append({a})

            eq_cls2 = []
            for a in set(stereo2.atoms[1:6]) - {a1}:
                for e in eq_cls2:
                    if any(a in eq_cls2[b] for b in e):
                        e.add(a)
                        break
                else:  # if for loop is not broken
                    eq_cls2.append({a})

            if len(eq_cls2) > len(eq_cls1):
                ...  # exchange 1 and 2

            if len(eq_cls1) == 3 or len(eq_cls1) == 1:
                for a_perm1 in stereo1._perm_atoms():
                    if a_perm1[1] == a2:
                        hb_atoms1 = (*a_perm1[2:6], a1)
                        break
                for a_perm2 in stereo2._perm_atoms():
                    if a_perm2[1] == a1:
                        hb_atoms2 = (a2, *reversed(a_perm2[2:6]))
                parity = (
                    stereo1.parity * stereo2.parity
                    if stereo1.parity is not None and stereo2.parity is not None
                    else None
                )
                s = HinderedBond33(atoms=(*hb_atoms1, *hb_atoms2), parity=parity)
            elif len(eq_cls1) == 2 and len(eq_cls2) == 2:
                ...

        elif len(nbrs1) == 2 and len(nbrs2) == 3:
            s = HinderedBond23(atoms=(*nbrs1, a1, a2, *nbrs2), parity=1)
