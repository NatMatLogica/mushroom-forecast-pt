#!/usr/bin/env python3
"""
Build the public/community forest table the MCP serves: every ICNF REFLOA area in mainland
Portugal with an interior point and the tree patches found by sampling the COS 2025 map
inside it (landcover.forest_habitats). Run once, or when REFLOA changes; output is shipped
with the MCP so no request ever walks the map.

Usage: python3 mushroom/build_public_forests.py OUT.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mushroom import landcover as L  # noqa: E402

PT = (36.9, 42.2, -9.6, -6.1)


def main(out):
    rows = []
    for f in L.public_forests():
        if not (PT[0] <= f["lat"] <= PT[1] and PT[2] <= f["lon"] <= PT[3]):
            continue
        geom = f.pop("geometry")
        patches = L.forest_habitats(geom, n=12)
        rows.append({**f, "patches": patches})
        print(f"{len(rows):3d} {f['name'][:60]} {list(patches)}", flush=True)
    with open(out, "w") as fh:
        json.dump({"source": "ICNF REFLOA (Regime Florestal) + DGT COS 2025", "forests": rows}, fh, ensure_ascii=False)
    print(f"wrote {out}: {len(rows)} forests")


if __name__ == "__main__":
    main(sys.argv[1])
