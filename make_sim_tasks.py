#!/usr/bin/env python3
"""Write sim_tasks.txt: one line per (source, date index, kind) for eddytransform_run_levante_sim.py.

Windows mirror SOURCES in eddytransform_run_levante_sim.py (first/last fully covered daily-mean date).
Usage: make_sim_tasks.py [--freq-days 10] [--sources a,b,c] [-o sim_tasks.txt]
"""
import argparse
from datetime import date

WINDOWS = {
    'tco2559_deepon': (date(2020, 1, 1), date(2020, 6, 30)),
    'tco2559_rcbmf': (date(2020, 1, 1), date(2021, 2, 28)),
    'tco3999_amip': (date(2020, 1, 1), date(2021, 2, 28)),
    'ng5_rcbmf': (date(2020, 1, 1), date(2021, 2, 28)),      # needs --tracks, not in the default set
    'ng5_deepoff': (date(2020, 1, 1), date(2021, 2, 28)),    # needs --tracks, not in the default set
}
DEFAULT = ['tco2559_deepon', 'tco2559_rcbmf', 'tco3999_amip']

ap = argparse.ArgumentParser()
ap.add_argument('--freq-days', type=int, default=10)
ap.add_argument('--sources', default=','.join(DEFAULT))
ap.add_argument('-o', default='sim_tasks.txt')
a = ap.parse_args()

n = 0
with open(a.o, 'w') as f:
    for src in a.sources.split(','):
        s, e = WINDOWS[src]
        ndates = len(range(0, (e - s).days + 1, a.freq_days))
        for kind in ('cyclonic', 'anticyclonic'):
            for i in range(ndates):
                f.write('%s %d %s\n' % (src, i, kind))
                n += 1
        print('%-16s %3d dates x 2 kinds' % (src, ndates))
print('%d tasks -> %s' % (n, a.o))
