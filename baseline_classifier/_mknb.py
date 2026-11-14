# -*- coding: utf-8 -*-
"""Helper: dung .ipynb hop le tu danh sach cell (type, source)."""
import json, sys


def make_notebook(cells, path):
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    for i, (ctype, src) in enumerate(cells):
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": ctype, "id": f"c{i:03d}", "metadata": {}, "source": lines}
        if ctype == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", path, f"({len(cells)} cells)")
