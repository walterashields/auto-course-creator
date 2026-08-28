# Reference Script: video_1_4

**Title:** Limiting Results with LIMIT
**Learning objective:** Limit the number of returned rows using the LIMIT clause

**Total words:** 384  
**Total beats:** 10

| Beat | Kind | Words | Text | Action |
|------|------|-------|------|--------|
| beat_001 | opening | 39 | In this video, we will limit the sorted customer contact list to a small preview using the LIMIT clause. Last lesson we sorted the full list alphabetically; now management only needs the first few rows for a quick look. | {"type": "wait", "duration": 1.5} |
| beat_002 | state | 46 | The Execute SQL tab is open, and the editor holds our sorted query. The result pane below shows the full alphabetical list by last name. We can add one more clause at the end of the statement to control exactly how many rows the database returns. | {"type": "wait", "duration": 1.5} |
| beat_003 | explain | 55 | LIMIT returns only the requested number of rows from the result set. It is useful when a full result is larger than we need, such as when a manager asks for a quick preview before running the full report. The database still sorts the rows first, because LIMIT comes after ORDER BY in the statement. | {"type": "wait", "duration": 1.5} |
| beat_004 | state | 40 | Right now the result pane shows the full alphabetical list. After we add a LIMIT clause, only the first few rows will remain visible, while the sort order stays the same. This turns a long report into a manageable preview. | {"type": "wait", "duration": 1.5} |
| beat_005 | demo | 14 | We type a comment header and the sorted, aliased query with a LIMIT clause. | {"type": "type_block", "text": "/*\nCreated By: WSDA Student\nCreate Date: 2026-08-27\nDescription: Preview of customer contacts\n*/\n\nSELECT\n    FirstName AS \"First Name\",\n    LastName AS \"Last Name\",\n    Email AS \"Email Address\"\nFROM Customer\nORDER BY LastName\nLIMIT 5;"} |
| beat_006 | demo | 13 | We run the query and the result pane shows only the preview rows. | {"type": "run_query"} |
| beat_007 | validation | 31 | We see exactly 5 rows returned, confirming the LIMIT clause trimmed the sorted result to the requested preview size. The rows still appear in alphabetical order, so the preview is representative. | {"type": "verify", "detail": "the result pane shows exactly five rows from the sorted contact list"} |
| beat_008 | explain | 56 | LIMIT trimmed the sorted result to a small preview, and the rows still appear in alphabetical order by last name. The database stopped after the requested number of rows, so the result loads faster and fits neatly on one screen. This is ideal for summaries and quick checks before a manager requests the full data set. | {"type": "wait", "duration": 1.5} |
| beat_009 | explain | 54 | LIMIT always comes after ORDER BY in a SELECT statement. If it came before the sort, the database would trim the rows first and then sort, which could return the wrong rows for the preview. Keeping the order SELECT, FROM, ORDER BY, LIMIT makes the query predictable and easy for the team to maintain. | {"type": "wait", "duration": 1.5} |
| beat_010 | close | 36 | We have limited the result set with LIMIT, and the preview remains sorted alphabetically by last name. Next, we will recap the chapter with one clean, well-documented query that combines comments, aliases, ORDER BY, and LIMIT. | {"type": "wait", "duration": 1.5} |
