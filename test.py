import os
from pathlib import Path

r = Path("C:/_data/(2021-08-13) PhotonicsDFA Vector [-1, -0.5, 0.5, 1]-normed")
for r1 in os.listdir(r):
    rr1 = Path(r).joinpath(r1)
    for r2 in os.listdir(rr1):
        rr2 = rr1.joinpath(r2)
        if r2.endswith(".svg"):
            os.remove(rr2)
            continue
        for r3 in os.listdir(rr2):
            rr3 = rr2.joinpath(r3)
            if os.path.isfile(rr3):
                os.remove(rr3)
