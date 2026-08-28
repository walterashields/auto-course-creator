# Reference Script: video_1_1

**Title:** Your First Query
**Learning objective:** Write and run your first SELECT query to return a customer contact list

**Total words:** 405  
**Total beats:** 10

| Beat | Kind | Words | Text | Action |
|------|------|-------|------|--------|
| beat_001 | opening | 43 | In this video, we will write our first SELECT query to pull a clean customer contact list for WSDA Music management. The goal is to return only the columns we need from the Customer table, so the result is focused and immediately useful. | {"type": "wait", "duration": 1.5} |
| beat_002 | state | 58 | DB Browser for SQLite opens with two tabs above the data view. Browse Data is on the left and shows raw table rows, while Execute SQL on the right opens the SQL editor on top and an empty result pane below. The toolbar sits above the tabs, and the editor is the large white area where we type. | {"type": "wait", "duration": 1.5} |
| beat_003 | explain | 54 | SELECT is the SQL command that chooses which columns to return, and FROM chooses which table holds those columns. Together they form the simplest useful query pattern: ask for specific data from one table. This keeps the result small and fast because the database does not waste time returning columns we do not need. | {"type": "wait", "duration": 1.5} |
| beat_004 | state | 40 | Right now the editor is empty and the result pane below it is blank. When we finish, the editor will hold a comment block followed by a formatted SELECT statement, and the result pane will show the customer contact list. | {"type": "wait", "duration": 1.5} |
| beat_005 | demo | 17 | We type a comment header and the formatted query as one contiguous block in the SQL editor. | {"type": "type_block", "text": "/*\nCreated By: WSDA Student\nCreate Date: 2026-08-27\nDescription: Customer contact list for management\n*/\n\nSELECT\n    FirstName,\n    LastName,\n    Email\nFROM Customer;"} |
| beat_006 | demo | 12 | We press F5 and the result pane fills with the contact list. | {"type": "run_query"} |
| beat_007 | validation | 26 | We see 60 rows returned with FirstName, LastName, and Email for each customer, confirming the contact list is complete and the query ran exactly as intended. | {"type": "verify", "detail": "the Execute SQL tab shows a populated results grid below the query"} |
| beat_008 | explain | 59 | The result pane shows the complete customer contact list, with each customer record appearing exactly once in the order it is stored. Management now has a reliable set of first names, last names, and email addresses without extra columns cluttering the view. Because the database returned only the columns we requested, the output is compact and fast to scan. | {"type": "wait", "duration": 1.5} |
| beat_009 | explain | 54 | Adding a comment header at the top of the query is a professional habit. It records who created the query, when it was written, and what problem it solves, which helps anyone who opens the file later. Good comments turn a quick one-off query into documentation that the whole team can trust and maintain. | {"type": "wait", "duration": 1.5} |
| beat_010 | close | 42 | We have written our first SELECT query and pulled a complete customer contact list. The query returned only the columns we needed and included a clear comment header. Next, we will make those column headers friendlier for management reports by using aliases. | {"type": "wait", "duration": 1.5} |
