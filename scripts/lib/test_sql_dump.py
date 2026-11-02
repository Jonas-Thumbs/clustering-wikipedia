"""Quick sanity check of the regexes in sql_dump.py against real sample
rows pulled from the actual dump files. Run directly: python3 test_sql_dump.py
"""
from sql_dump import (
    _RE_CATEGORYLINKS,
    _RE_LINKTARGET,
    _RE_PAGE,
    _RE_PAGELINKS,
    _RE_REDIRECT,
    unescape,
)

page_rows = [
    "(10,0,'AccessibleComputing',1,0,0.856935107283,'20260820173259','20260820174850',1367012995,111,'wikitext',NULL),",
    "(12,0,'Anarchism',0,0,0.870480291199,'20260902162526','20260902162607',1372193622,114143,'wikitext',NULL),",
    "(25,0,'Autism_spectrum',1,0,0.626026654267708,'20260901065631','20260830180301',1355089873,182,'wikitext',NULL);",
]
for row in page_rows:
    m = _RE_PAGE.match(row)
    assert m, f"page regex failed on: {row}"
    print("page:", m.groups())

linktarget_rows = [
    "(115058193,-2,'Call_of_Duty'),",
    "(10989162,-1,'BlankPage'),",
    r"(78676871,0,'\'\'DEACON_(rapper)\'\''),",
]
for row in linktarget_rows:
    m = _RE_LINKTARGET.match(row)
    assert m, f"linktarget regex failed on: {row}"
    title = unescape(m.group(3))
    print("linktarget:", m.group(1), m.group(2), repr(title))

cl_rows = [
    r"(10,'*..2NN:,@2.FBHRP:D6ܽ','2026-07-07 10:01:29','','page',1,22896130),",
    r"(12,'*D*L.8:NB*D*L.8:NBܽ','2025-06-16 13:16:29','Anarchism','page',1,27206937),",
]
for row in cl_rows:
    m = _RE_CATEGORYLINKS.match(row)
    assert m, f"categorylinks regex failed on: {row}"
    print("categorylinks:", m.groups())

pagelinks_rows = [
    "(1939,0,2),",
    "(3040,0,2),",
    "(14216,0,2);",
]
for row in pagelinks_rows:
    m = _RE_PAGELINKS.match(row)
    assert m, f"pagelinks regex failed on: {row}"
    print("pagelinks:", m.groups())

redirect_rows = [
    "(10,0,'Computer_accessibility','',''),",
    "(13,0,'History_of_Afghanistan','',''),",
    r"(99,0,'O\'Brien_House',NULL,NULL);",
]
for row in redirect_rows:
    m = _RE_REDIRECT.match(row)
    assert m, f"redirect regex failed on: {row}"
    title = unescape(m.group(3))
    print("redirect:", m.group(1), m.group(2), repr(title))

print("ALL OK")
