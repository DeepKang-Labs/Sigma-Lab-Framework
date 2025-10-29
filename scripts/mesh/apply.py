#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compatibility shim for legacy path scripts/mesh/apply.py
It simply delegates to scripts/mesh_apply.py
"""

from scripts.mesh_apply import main

if __name__ == "__main__":
    main()
