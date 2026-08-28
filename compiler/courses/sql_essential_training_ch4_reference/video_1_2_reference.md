# Reference Script: video_1_2

**Title:** Aliases — Speaking Management's Language
**Learning objective:** Use the AS keyword to give query results readable column headers

**Total words:** 397  
**Total beats:** 10

| Beat | Kind | Words | Text | Action |
|------|------|-------|------|--------|
| beat_001 | opening | 43 | In this video, we will use the AS keyword to give our query results readable column headers that match management language. Last lesson we pulled the raw contact list; now we want friendly labels like First Name instead of FirstName in the report. | {"type": "wait", "duration": 1.5} |
| beat_002 | state | 52 | The Execute SQL tab is still open from the last query. The editor shows our previous SELECT statement, and the result pane below still displays the raw headers. We can clear the editor and type a new query that replaces each raw column name with a friendly alias using the AS keyword. | {"type": "wait", "duration": 1.5} |
| beat_003 | explain | 58 | The AS keyword gives a column an alias, which is the name that appears in the result header instead of the raw column name. Aliases make reports easier to read without changing the underlying data or the table structure. They are especially useful when managers want headers in plain English while the database keeps its original column names. | {"type": "wait", "duration": 1.5} |
| beat_004 | state | 43 | Right now the result headers show the raw column names. After we run the aliased query, those same three headers will display with spaces, while the rows underneath stay exactly the same. This is a before-and-after change we can verify at a glance. | {"type": "wait", "duration": 1.5} |
| beat_005 | demo | 16 | We type a comment header and the query that aliases each column to a readable header. | {"type": "type_block", "text": "/*\nCreated By: WSDA Student\nCreate Date: 2026-08-27\nDescription: Readable customer contact headers\n*/\n\nSELECT\n    FirstName AS \"First Name\",\n    LastName AS \"Last Name\",\n    Email AS \"Email Address\"\nFROM Customer;"} |
| beat_006 | demo | 13 | We run the query and the result pane fills with the aliased list. | {"type": "run_query"} |
| beat_007 | validation | 28 | We see 60 rows returned with headers reading First Name, Last Name, and Email Address, confirming the AS aliases took effect and the report is readable for management. | {"type": "verify", "detail": "the Execute SQL tab shows a populated results grid with aliased headers"} |
| beat_008 | explain | 54 | The headers now show the friendly aliases, giving management the readable view they requested. The underlying values did not change; only the labels at the top of each column did. This means the same query can serve both technical users who know the original schema and managers who need a polished report for presentations. | {"type": "wait", "duration": 1.5} |
| beat_009 | explain | 52 | Aliases also help when a column name is long or unclear. A short, descriptive alias keeps the header visible in a narrow spreadsheet column and makes formulas easier to write. Using spaces in quoted aliases is common in reports, but the quotes are required so the database recognizes the whole multi-word header. | {"type": "wait", "duration": 1.5} |
| beat_010 | close | 38 | We have used the AS keyword to create readable headers for the customer contact list. The result now speaks management's language without changing any data. Next, we will sort that list alphabetically by last name using ORDER BY. | {"type": "wait", "duration": 1.5} |
