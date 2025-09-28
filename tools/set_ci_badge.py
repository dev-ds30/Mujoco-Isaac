
#!/usr/bin/env python3
import argparse, re
p = argparse.ArgumentParser()
p.add_argument('--owner', required=True)
p.add_argument('--repo', required=True)
p.add_argument('--readme', default='README.md')
a = p.parse_args()
badge = f"[![CI](https://github.com/{a.owner}/{a.repo}/actions/workflows/ci.yml/badge.svg)](https://github.com/{a.owner}/{a.repo}/actions/workflows/ci.yml)"
txt = open(a.readme, 'r', encoding='utf-8').read()
pat = re.compile(r"^\[!\[CI\]\(https://github.com/.+?/actions/workflows/ci.yml/badge.svg\)\]\(.+?\)\s*$", re.M)
if pat.search(txt): txt = pat.sub(badge, txt, count=1)
else: txt = badge + "\n\n" + txt
open(a.readme, 'w', encoding='utf-8').write(txt)
print("Updated badge for", a.owner, "/", a.repo)
