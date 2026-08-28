# Reference Script: video_1_5

**Title:** Query Etiquette Recap
**Learning objective:** Combine comment headers, aliases, ORDER BY, and LIMIT in one clean query

**Total words:** 414  
**Total beats:** 10

| Beat | Kind | Words | Text | Action |
|------|------|-------|------|--------|
| beat_001 | opening | 41 | In this video, we will combine comment headers, aliases, ORDER BY, and LIMIT into one clean query. The previous lessons built each skill step by step; now we use them together in a single professional statement that is ready to share. | {"type": "wait", "duration": 1.5} |
| beat_002 | state | 50 | The Execute SQL tab is open, so we can build the final chapter query step by step. The comment block will come first, followed by SELECT with aliases, then FROM, ORDER BY, and finally LIMIT. Each clause has a specific job that we have already practiced in the earlier videos. | {"type": "wait", "duration": 1.5} |
| beat_003 | explain | 53 | A comment header documents the query's purpose for anyone who opens it later. Aliases make the headers readable, ORDER BY sorts the rows alphabetically, and LIMIT keeps the output concise. Together these four techniques turn a raw database query into a polished, shareable report for management, showing both technical precision and professional presentation. | {"type": "wait", "duration": 1.5} |
| beat_004 | state | 45 | Right now the editor is empty. After we finish, it will hold one contiguous comment block followed by a SELECT statement that uses aliases, ORDER BY LastName, and a LIMIT clause. The result pane will then show the documented preview in a single clean view. | {"type": "wait", "duration": 1.5} |
| beat_005 | demo | 16 | We type the commented query that uses aliases, orders by last name, and limits the result. | {"type": "type_block", "text": "/*\nCreated By: WSDA Student\nCreate Date: 2026-08-27\nDescription: Clean, documented customer contact preview\n*/\n\nSELECT\n    FirstName AS \"First Name\",\n    LastName AS \"Last Name\",\n    Email AS \"Email Address\"\nFROM Customer\nORDER BY LastName\nLIMIT 5;"} |
| beat_006 | demo | 12 | We run the query and the result pane shows the documented preview. | {"type": "run_query"} |
| beat_007 | validation | 38 | We see 5 rows returned with readable headers sorted alphabetically by last name, confirming the combined query works as one clean statement. The comment block at the top documents the purpose for anyone who opens the file later. | {"type": "verify", "detail": "the SQL editor contains a comment block followed by a SELECT with aliases, ORDER BY, and LIMIT"} |
| beat_008 | explain | 60 | The result pane shows a professional preview with readable rows sorted by last name, demonstrating all three presentation techniques in one statement. The comment header makes the query's intent clear at a glance, while aliases, ORDER BY, and LIMIT handle the headers, order, and size. This is the kind of query a data analyst can confidently share with a manager. | {"type": "wait", "duration": 1.5} |
| beat_009 | explain | 51 | Clean query etiquette matters when other people will read or maintain the code. A comment header explains intent, aliases remove jargon, ORDER BY removes ambiguity about row order, and LIMIT prevents accidental overload of a report. These habits separate a quick scratch query from production-ready SQL that a team can trust. | {"type": "wait", "duration": 1.5} |
| beat_010 | close | 48 | We have recapped the chapter with a clean, documented query that combines comment headers, aliases, ORDER BY, and LIMIT. The result is a readable, sorted, and appropriately sized preview. Next, we will filter results with WHERE clauses so we can return only the rows that match specific conditions. | {"type": "wait", "duration": 1.5} |
