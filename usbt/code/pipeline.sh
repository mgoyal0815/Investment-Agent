#!/bin/bash
# Train 2012-2019, test 2020-2026 and 2000-2011. Same search space, seed and objective as the
# first round; the selection rule is fixed before the test windows are scored.
set -e
cd "$(dirname "$0")"
python3 exp1_switches.py > ../results/exp1_switches.txt 2>/dev/null
python3 exp2_search.py 600 > ../results/exp2_log.txt 2>/dev/null
python3 exp2b_local.py > ../results/exp2b_log.txt 2>/dev/null
python3 -c "
import json; d=json.load(open('../results/exp2b_best.json'))
ex=[{'label':'Local-search best','cfg':d['balanced']['cfg']}]
if d['beat_all']['cfg']!=d['balanced']['cfg']: ex.append({'label':'Local-search best (beat-all)','cfg':d['beat_all']['cfg']})
json.dump(ex,open('../results/extra_finalists.json','w'))"
python3 exp3_final.py 4 extra > ../results/exp3_log.txt 2>/dev/null
echo PIPELINE_DONE
