# Reference Script: video_1_3

**Title:** Sorting Results with ORDER BY
**Learning objective:** Sort query results by a specific column using ORDER BY

**Total words:** 400  
**Total beats:** 10

| Beat | Kind | Words | Text | Action |
|------|------|-------|------|--------|
| beat_001 | opening | 42 | In this video, we will sort the customer contact list alphabetically by last name using the ORDER BY clause. Last lesson we made the headers readable with aliases; now we control the order in which the rows appear in the result pane. | {"type": "wait", "duration": 1.5} |
| beat_002 | state | 49 | The Execute SQL tab is open, and the editor still holds our aliased query. The result pane below shows the readable headers with rows in their stored order. We can change the query to tell the database exactly how to arrange the returned rows before they reach the screen. | {"type": "wait", "duration": 1.5} |
| beat_003 | explain | 58 | ORDER BY tells the database how to sort the returned rows. It does not change which rows are selected or the columns that come back; it only rearranges the order in which they appear. By default, text columns sort alphabetically from A to Z, which is what we want for a contact list that managers can scan quickly. | {"type": "wait", "duration": 1.5} |
| beat_004 | state | 43 | Right now the rows appear in the order they are stored in the table. After we add ORDER BY LastName, the same rows will reorder so the earliest last name appears at the top and the rest follow alphabetically down the result pane. | {"type": "wait", "duration": 1.5} |
| beat_005 | demo | 19 | We type a comment header and the query that aliases each column and orders the results by last name. | {"type": "type_block", "text": "/*\nCreated By: WSDA Student\nCreate Date: 2026-08-27\nDescription: Customer contact list sorted by last name\n*/\n\nSELECT\n    FirstName AS \"First Name\",\n    LastName AS \"Last Name\",\n    Email AS \"Email Address\"\nFROM Customer\nORDER BY LastName;"} |
| beat_006 | demo | 13 | We run the query and the rows reorder alphabetically in the result pane. | {"type": "run_query"} |
| beat_007 | validation | 28 | We see 60 rows returned, with Almeida at the top of the Last Name column, confirming the ascending alphabetical sort is active and the rows follow A-to-Z order. | {"type": "verify", "detail": "the result pane rows are sorted by Last Name in ascending order"} |
| beat_008 | explain | 57 | The rows are now arranged alphabetically by last name, with the earliest last name at the top of the list. This makes it easy to scan contacts from A to Z when looking for a specific person. The selected columns and aliases did not change; only the sequence of rows changed to match the ORDER BY clause. | {"type": "wait", "duration": 1.5} |
| beat_009 | explain | 55 | ORDER BY belongs at the end of the SELECT statement, after the column list and FROM clause. Putting it last is a readability convention that helps the team see the sort rule at a glance. It also makes the query easier to edit when we later add filtering or limiting clauses before the final sort. | {"type": "wait", "duration": 1.5} |
| beat_010 | close | 36 | We have sorted the customer contact list alphabetically by last name with ORDER BY. The rows now appear in A-to-Z order, ready for scanning. Next, we will limit the result to a small preview using LIMIT. | {"type": "wait", "duration": 1.5} |
